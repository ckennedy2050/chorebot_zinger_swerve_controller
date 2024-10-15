# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import math

import numpy as np
import rclpy
from rclpy.clock import Clock, Time
from rclpy.duration import Duration as TimeDuration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from tf2_geometry_msgs import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf2_ros

from builtin_interfaces.msg import Duration as MsgDuration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tf_transformations import quaternion_from_euler

from .control import BodyMotionCommand
from .drive_module import DriveModule
from .geometry import Point
from .profile import SingleVariableLinearProfile, SingleVariableSCurveProfile, TransientVariableProfile
from .states import DriveModuleMeasuredValues
from .steering_controller import DriveModuleDesiredValuesProfilePoint, ModuleFollowsBodySteeringController

class SwerveController(Node):
    def __init__(self):
        super().__init__("publisher_velocity_controller")
        # Declare all parameters
        self.declare_parameter("robot_base_frame", "chassis_link")
        self.declare_parameter("twist_topic", "cmd_vel")

        self.declare_parameter("position_controller_name", "position_controller")
        self.declare_parameter("velocity_controller_name", "velocity_controller")
        self.declare_parameter("cycle_frequency", 10)

        self.declare_parameter("steering_joints",
                               ["joint_actuator_servo_front_right",
                                "joint_actuator_servo_front_left",
                                "joint_actuator_servo_rear_right",
                                "joint_actuator_servo_rear_left"])
        self.declare_parameter("drive_joints",
                               ["joint_actuator_wheel_front_right",
                                "joint_actuator_wheel_front_left",
                                "joint_actuator_wheel_rear_right",
                                "joint_actuator_wheel_rear_left"])


        #self.declare_parameter("motion_estimation_time_span", 1.)
        self.declare_parameter("motion_estimation_time_span", 0.25)
        self.motion_time_span = self.get_parameter("motion_estimation_time_span").value

        self.declare_parameter("new_motion_tolerance", 0.01)
        self.new_motion_tolerance = self.get_parameter("new_motion_tolerance").value

        # CK
        self.declare_parameter("max_body_angular_acceleration", 1.3)

        self.get_logger().info(f'Initializing swerve controller ...')

        self.last_velocity_command: Twist = None

        self.robot_base_link = self.get_parameter("robot_base_frame").value
        """
        # publish the module steering angle
        position_controller_name = self.get_parameter("position_controller_name").value
        steering_angle_publish_topic = "/" + position_controller_name + "/" + "commands"
        self.drive_module_steering_angle_publisher = self.create_publisher(
            Float64MultiArray,
            steering_angle_publish_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))

        self.get_logger().info(
            f'Publishing steering angle changes on topic "{steering_angle_publish_topic}"'
        )

        # publish the module drive velocity
        velocity_controller_name = self.get_parameter("velocity_controller_name").value
        velocity_publish_topic = "/" + velocity_controller_name + "/" + "commands"
        self.drive_module_velocity_publisher = self.create_publisher(
            Float64MultiArray,
            velocity_publish_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))

        self.get_logger().info(
            f'Publishing drive velocity changes on topic "{velocity_publish_topic}"'
        )
        """
        # publish odometry
        odom_topic = "/odom_drive"
        self.odometry_publisher = self.create_publisher(
            Odometry,
            odom_topic,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))
        self.get_logger().info(
            f'Publishing odometry information on topic "{odom_topic}"'
        )

        # Define TF broadcaster 
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize odom TF 
        zero_odometry = Odometry()
        zero_odometry.header.stamp = self.get_clock().now().to_msg()
        zero_odometry.header.frame_id = "odom"
        zero_odometry.child_frame_id = self.robot_base_link
        zero_odometry.pose.pose.position.x = 0.0
        zero_odometry.pose.pose.position.y = 0.0
        zero_odometry.pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        zero_odometry.pose.pose.orientation.x = quat[0]
        zero_odometry.pose.pose.orientation.y = quat[1]
        zero_odometry.pose.pose.orientation.z = quat[2]
        zero_odometry.pose.pose.orientation.w = quat[3]
        self.send_odom_transform(zero_odometry)

        # self.send_static_tf()

        # Create the controller that will determine the correct drive commands for the different drive modules
        # Create the controller before we subscribe to state changes so that the first change that comes in gets
        # registered
        self.get_logger().info(f'Storing drive module information...')
        self.drive_modules = self.get_drive_modules()
        self.controller = ModuleFollowsBodySteeringController(self.drive_modules, self.get_motion_profile,
                                                              self.get_parameter("max_body_angular_acceleration").value,
                                                              self.write_log)

        # initialize the time tracking variables after we get the controller up and running
        # so that we can initialize the controller at the same time.
        self.store_time_and_update_controller_time()
        self.last_control_update_send_at = self.last_recorded_time
        self.last_velocity_command_received_at = self.last_recorded_time

        # keep last position message to avoid inf value in steering angle data
        self.last_position_msg: Float64MultiArray = None

        # Create the timer that is used to ensure that we publish movement data regularly
        self.cycle_time_in_hertz = self.get_parameter("cycle_frequency").value
        self.get_logger().info(
            f'Publishing changes at fequency: "{self.cycle_time_in_hertz}" Hz'
        )

        self.timer = self.create_timer(
            1.0 / self.cycle_time_in_hertz,
            self.timer_callback,
            callback_group=None,
            clock=self.get_clock())
        self.i = 0

        # Listen for state changes in the drive modules
        joint_state_topic = "chassis_joint_states"
        self.state_change_subscription = self.create_subscription(
            JointState,
            joint_state_topic,
            self.joint_states_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10)
        )

        self.get_logger().info(
            f'Listening for drive module state changes on "{joint_state_topic}"'
        )

        # Initialize the drive modules
        self.last_drive_module_state = self.initialize_drive_module_states(self.drive_modules)

        self.joint_command_publisher = self.create_publisher(JointState, "joint_command", 10)
        self.steering_joint_names = self.get_parameter("steering_joints").value
        self.drive_joint_names = self.get_parameter("drive_joints").value
        self.last_steering_angle_values_deg = [0.] * len(self.steering_joint_names)

        # Finally listen to the cmd_vel topic for movement commands. We could have a message incoming
        # at any point after we register so we set this subscription up last.
        twist_topic = self.get_parameter("twist_topic").value
        self.cmd_vel_subscription = self.create_subscription(
            Twist,
            twist_topic,
            self.cmd_vel_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                durability=DurabilityPolicy.VOLATILE,
                depth=10))
        self.get_logger().info(
            f'Listening for movement commands on topic "{twist_topic}"'
        )

    def cmd_vel_callback(self, msg: Twist):
        if msg == None:
            return

        # If this twist message is the same as last time, then we don't need to do anything
        if self.last_velocity_command is not None:

            """
            now_seconds = self.get_clock().now().nanoseconds * 1e-9
            prev_seconds = self.last_velocity_command_received_at.nanoseconds * 1e-9

            # Cap the velocity based on max acceleration value
            norm_msg: Twist = cap_velocity_from_max_acceleration(self.last_velocity_command,
                                                                 msg,
                                                                 0.2,
                                                                 now_seconds - prev_seconds)
            """


            last_array = np.array([self.last_velocity_command.linear.x, self.last_velocity_command.linear.y, self.last_velocity_command.angular.z])
            msg_array = np.array([msg.linear.x, msg.linear.y, msg.angular.z])

            any_new_zero = np.any((msg_array == 0.) & (last_array != 0.))

            if np.allclose(last_array, msg_array, atol=self.new_motion_tolerance) and not any_new_zero:


                # if msg.linear.x == self.last_velocity_command.linear.x and \
                #    msg.linear.y == self.last_velocity_command.linear.y and \
                #    msg.angular.z == self.last_velocity_command.angular.z:

                    # The last command was the same as the current command. So just ignore it and move on.
                    # self.get_logger().info(
                    #     f'Received a Twist message that is the same as the last message. Taking no action. Message was: "{msg}"'
                    # )

                return


        # self.get_logger().info(
        #     f'Received a Twist message that is different from the last command. Processing message: "{msg}"'
        # )

        # When we get a stream of command it is possible that each command is slightly different (looking at you ROS2 nav)
        # This means we reset the starting time of the change profile each time, which starts the process all over
        # Because we don't take the current steering velocity / drive acceleration into account we assume that we
        # start from rest. That is wrong. We should be starting from a place where we have the current
        # steering velocity / drive acceleration.

        self.store_time_and_update_controller_time()
        self.controller.on_desired_state_update(
            BodyMotionCommand(
                self.motion_time_span, # THIS SHOULD REALLY BE CALCULATED SOME HOW
                msg.linear.x,
                msg.linear.y,
                msg.angular.z
            )
        )

        self.last_velocity_command = msg
        self.last_velocity_command_received_at = self.last_recorded_time

    def get_drive_modules(self) -> List[DriveModule]:
        # Get the drive module information from the URDF and turn it into a list of drive modules.
        #
        # For now we don't read the URDF and just hard-code the drive modules
        robot_length = 0.75
        robot_width = 0.3366

        steering_radius = 0.05

        wheel_radius = 0.062
        wheel_width = 0.05

        # Steering motor params are not used, only drive motor
        steering_motor_maximum_velocity = 10.
        steering_motor_minimum_acceleration = 0.02
        steering_motor_maximum_acceleration = 1.0
        drive_motor_maximum_velocity = 0.6
        drive_motor_minimum_acceleration = 0.1
        drive_motor_maximum_acceleration = 1.0

        # store the steering joints
        steering_joint_names = self.get_parameter("steering_joints").value
        steering_joints = []
        for name in steering_joint_names:
            steering_joints.append(name)
            self.get_logger().info(
                f'Discovered steering joint: "{name}"'
            )

        # store the drive joints
        drive_joint_names = self.get_parameter("drive_joints").value
        drive_joints = []
        for name in drive_joint_names:
            drive_joints.append(name)
            self.get_logger().info(
                f'Discovered drive joint: "{name}"'
            )

        drive_modules: List[DriveModule] = []
        drive_module_name = "front_right"
        right_front = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(0.5 * (robot_length - 2 * steering_radius), -0.5 * (robot_width - steering_radius), 0.0),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            steering_motor_maximum_velocity=steering_motor_maximum_velocity,
            steering_motor_minimum_acceleration=steering_motor_minimum_acceleration,
            steering_motor_maximum_acceleration=steering_motor_maximum_acceleration,
            drive_motor_maximum_velocity=drive_motor_maximum_velocity,
            drive_motor_minimum_acceleration=drive_motor_minimum_acceleration,
            drive_motor_maximum_acceleration=drive_motor_maximum_acceleration
        )
        drive_modules.append(right_front)

        self.get_logger().info(
            f'Configured drive module: "{right_front.name}" ' +
            f'with steering link: "{right_front.steering_link_name}" ' +
            f'and drive link: "{right_front.driving_link_name}" ' +
            f'and position: ["{right_front.steering_axis_xy_position.x}", "{right_front.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "front_left"
        left_front = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(0.5 * (robot_length - 2 * steering_radius), 0.5 * (robot_width - steering_radius), 0.0),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            steering_motor_maximum_velocity=steering_motor_maximum_velocity,
            steering_motor_minimum_acceleration=steering_motor_minimum_acceleration,
            steering_motor_maximum_acceleration=steering_motor_maximum_acceleration,
            drive_motor_maximum_velocity=drive_motor_maximum_velocity,
            drive_motor_minimum_acceleration=drive_motor_minimum_acceleration,
            drive_motor_maximum_acceleration=drive_motor_maximum_acceleration
        )
        drive_modules.append(left_front)

        self.get_logger().info(
            f'Configured drive module: "{left_front.name}" ' +
            f'with steering link: "{left_front.steering_link_name}" ' +
            f'and drive link: "{left_front.driving_link_name}" ' +
            f'and position: ["{left_front.steering_axis_xy_position.x}", "{left_front.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "rear_right"
        right_rear = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(-0.5 * (robot_length - 2 * steering_radius), -0.5 * (robot_width - steering_radius), 0.0),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            steering_motor_maximum_velocity=steering_motor_maximum_velocity,
            steering_motor_minimum_acceleration=steering_motor_minimum_acceleration,
            steering_motor_maximum_acceleration=steering_motor_maximum_acceleration,
            drive_motor_maximum_velocity=drive_motor_maximum_velocity,
            drive_motor_minimum_acceleration=drive_motor_minimum_acceleration,
            drive_motor_maximum_acceleration=drive_motor_maximum_acceleration
        )
        drive_modules.append(right_rear)

        self.get_logger().info(
            f'Configured drive module: "{right_rear.name}" ' +
            f'with steering link: "{right_rear.steering_link_name}" ' +
            f'and drive link: "{right_rear.driving_link_name}" ' +
            f'and position: ["{right_rear.steering_axis_xy_position.x}", "{right_rear.steering_axis_xy_position.y}"]'
        )

        drive_module_name = "rear_left"
        left_rear = DriveModule(
            name=drive_module_name,
            steering_link=next((x for x in steering_joints if drive_module_name in x), "joint_steering_{}".format(drive_module_name)),
            drive_link=next((x for x in drive_joints if drive_module_name in x), "joint_drive_{}".format(drive_module_name)),
            steering_axis_xy_position=Point(-0.5 * (robot_length - 2 * steering_radius), 0.5 * (robot_width - steering_radius), 0.0),
            wheel_radius=wheel_radius,
            wheel_width=wheel_width,
            steering_motor_maximum_velocity=steering_motor_maximum_velocity,
            steering_motor_minimum_acceleration=steering_motor_minimum_acceleration,
            steering_motor_maximum_acceleration=steering_motor_maximum_acceleration,
            drive_motor_maximum_velocity=drive_motor_maximum_velocity,
            drive_motor_minimum_acceleration=drive_motor_minimum_acceleration,
            drive_motor_maximum_acceleration=drive_motor_maximum_acceleration
        )
        drive_modules.append(left_rear)

        self.get_logger().info(
            f'Configured drive module: "{left_rear.name}" ' +
            f'with steering link: "{left_rear.steering_link_name}" ' +
            f'and drive link: "{left_rear.driving_link_name}" ' +
            f'and position: ["{left_rear.steering_axis_xy_position.x}", "{left_rear.steering_axis_xy_position.y}"]'
        )

        return drive_modules

    def get_motion_profile(self, start: float, end: float) -> TransientVariableProfile:
        # return SingleVariableSCurveProfile(start, end)

        return SingleVariableLinearProfile(start, end)

    def initialize_drive_module_states(self, drive_modules: List[DriveModule]) -> List[DriveModuleMeasuredValues]:
        measured_drive_states: List[DriveModuleMeasuredValues] = []
        for drive_module in self.drive_modules:

            value = DriveModuleMeasuredValues(
                drive_module.name,
                drive_module.steering_axis_xy_position.x,
                drive_module.steering_axis_xy_position.y,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0
            )
            measured_drive_states.append(value)

            self.get_logger().info(
                f'Initializing drive module state for module: "{drive_module.name}"'
            )

        self.store_time_and_update_controller_time()
        self.controller.on_state_update(measured_drive_states)

        return measured_drive_states

    def joint_states_callback(self, msg: JointState):
        if msg == None:
            return

        # self.get_logger().debug(
        #     f'Received a JointState message: "{msg}"'
        # )

        # It would be better if we stored this message and processed it during our own timer loop. That way
        # we wouldn't be blocking the callback.

        joint_names: List[str] = msg.name
        joint_positions: List[float] = [pos for pos in msg.position]
        joint_velocities: List[float] = [vel for vel in msg.velocity]

        ### NOTE: This could be specific to sim
        for joint_name in joint_names:
            idx = joint_names.index(joint_name)
            # Flip the right wheel velocities on incoming
            if "actuator_wheel_front_right" in joint_name or "actuator_wheel_rear_right" in joint_name:
                joint_velocities[idx] *= -1.
        ###



        measured_drive_states: List[DriveModuleMeasuredValues] = []
        for index, drive_module in enumerate(self.drive_modules):
            if drive_module.steering_link_name in joint_names and drive_module.driving_link_name in joint_names:
                steering_values_index = joint_names.index(drive_module.steering_link_name)
                drive_values_index = joint_names.index(drive_module.driving_link_name)

                value = DriveModuleMeasuredValues(
                    drive_module.name,
                    drive_module.steering_axis_xy_position.x,
                    drive_module.steering_axis_xy_position.y,
                    joint_positions[steering_values_index],
                    joint_velocities[steering_values_index],
                    0.0,
                    0.0,
                    joint_velocities[drive_values_index] * drive_module.wheel_radius,
                    0.0,
                    0.0
                )
                measured_drive_states.append(value)

                # self.get_logger().info(
                #     f'Updating joint states for: "{drive_module.name}" with: ' +
                #     f'[ steering angle: "{value.orientation_in_body_coordinates.z}", ' +
                #     f' steering velocity: "{value.orientation_velocity_in_body_coordinates.z}",' +
                #     f' velocity: "{value.drive_velocity_in_module_coordinates.x}" ] '
                # )
            else:
                # grab the previous state and just assume that's the one
                value = self.last_drive_module_state[index]
                measured_drive_states.append(value)

                # self.get_logger().debug(
                #     f'Updating joint states for: "{drive_module.name}" with: ' +
                #     f'[ steering angle: "{value.orientation_in_body_coordinates.z}", ' +
                #     f' steering velocity: "{value.orientation_velocity_in_body_coordinates.z}",' +
                #     f' velocity: "{value.drive_velocity_in_module_coordinates.x}" ] '
                # )

        # Ideally we would get the time from the message. And then check if we have gotten a more
        # recent message
        self.store_time_and_update_controller_time()
        self.controller.on_state_update(measured_drive_states)
        self.last_drive_module_state = measured_drive_states

    def publish_odometry(self):
        body_state = self.controller.body_state_at_current_time()

        msg = Odometry()
        msg.header.stamp = self.last_recorded_time.to_msg()
        msg.header.frame_id = "odom_drive"
        msg.child_frame_id = self.robot_base_link
        msg.pose.pose.position.x = body_state.position_in_world_coordinates.x
        msg.pose.pose.position.y = body_state.position_in_world_coordinates.y
        msg.pose.pose.position.z = body_state.position_in_world_coordinates.z

        quat = quaternion_from_euler(0.0, 0.0, body_state.orientation_in_world_coordinates.z)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        msg.twist.twist.linear.x = body_state.motion_in_body_coordinates.linear_velocity.x
        msg.twist.twist.linear.y = body_state.motion_in_body_coordinates.linear_velocity.y
        msg.twist.twist.linear.z = body_state.motion_in_body_coordinates.linear_velocity.z

        msg.twist.twist.angular.x = body_state.motion_in_body_coordinates.angular_velocity.x
        msg.twist.twist.angular.y = body_state.motion_in_body_coordinates.angular_velocity.y
        msg.twist.twist.angular.z = body_state.motion_in_body_coordinates.angular_velocity.z

        self.send_odom_transform(msg)

        # For now we ignore the covariances

        # self.get_logger().info(
        #     'Publishing odometry message {}'.format(msg)
        # )

        self.odometry_publisher.publish(msg)

    def send_odom_transform(self, odometry_msg: Odometry):
        transform = TransformStamped()
        transform.header.stamp = odometry_msg.header.stamp
        transform.header.frame_id = "odom_drive"
        transform.child_frame_id = self.robot_base_link
        transform.transform.translation.x = odometry_msg.pose.pose.position.x
        transform.transform.translation.y = odometry_msg.pose.pose.position.y
        transform.transform.translation.z = odometry_msg.pose.pose.position.z
        transform.transform.rotation.x = odometry_msg.pose.pose.orientation.x
        transform.transform.rotation.y = odometry_msg.pose.pose.orientation.y
        transform.transform.rotation.z = odometry_msg.pose.pose.orientation.z
        transform.transform.rotation.w = odometry_msg.pose.pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)

    def send_static_tf(self):
        tf_static_broadcaster = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "odom_drive"
        transform.child_frame_id = self.robot_base_link
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        quat = quaternion_from_euler(0.0, 0.0, 0.0)
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]
        tf_static_broadcaster.sendTransform(transform)

    def store_time_and_update_controller_time(self):
        time: Time = self.get_clock().now()
        seconds = time.nanoseconds * 1e-9
        self.controller.on_tick(seconds)
        self.last_recorded_time = time

    def timer_callback(self):
        self.store_time_and_update_controller_time()

        # always send out the odometry information
        #self.publish_odometry()

        # Check if we actually have a movement profile to send
        current_time = self.get_clock().now()
        trajectory_running_duration: TimeDuration = current_time - self.last_velocity_command_received_at
        # self.get_logger().debug(
        #     'Current trajectory duration {} s. Based on current time {} and sequence start time {}'.format(
        #         trajectory_running_duration,
        #         current_time,
        #         self.last_velocity_command_received_at
        #     )
        # )

        running_duration_as_float: float = trajectory_running_duration.nanoseconds * 1e-9
        # self.get_logger().debug(
        #     'Current trajectory duration {} s'.format(running_duration_as_float)
        # )

        #if running_duration_as_float > self.controller.min_time_for_profile:
        if running_duration_as_float > self.controller.min_time_for_profile + 1. / self.cycle_time_in_hertz and \
                not self.controller.had_illegal_rotation:
            # self.get_logger().debug(
            #     'Trajectory completed waiting for next command.'
            # )
            return

        # CK
        #next_time_step = current_time.nanoseconds * 1e-9 + 1.0  / self.cycle_time_in_hertz
        next_time_step = current_time.nanoseconds * 1e-9 + self.motion_time_span / self.cycle_time_in_hertz
        # self.get_logger().debug(
        #     'Calculating next step in profile at time {} s'.format(next_time_step)
        # )

        drive_module_states = self.controller.drive_module_state_at_future_time(next_time_step)

        # Only publish movement commands if there is a trajectory
        if len(drive_module_states) == 0:
            return

        position_msg = Float64MultiArray()
        steering_angle_values = [a.steering_angle_in_radians for a in drive_module_states]
        position_msg.data = steering_angle_values

        # Note that the controller gives the velocity in meters per second, i.e. the velocity of the wheel at the
        # contact point with the ground. But ROS wants to know the rotational velocity of the wheel
        drive_velocity_values = []
        for a in drive_module_states:
            linear_velocity = a.drive_velocity_in_meters_per_second
            wheel_radius = next((x.wheel_radius for x in self.drive_modules if x.name == a.name), 1.0)
            drive_velocity_values.append(linear_velocity / wheel_radius)

        """
        velocity_msg = Float64MultiArray()
        velocity_msg.data = drive_velocity_values

        # if there are some inf values in data publish last position instead (or update last position message)
        if (any(math.isinf(x) for x in position_msg.data)) and not (self.last_position_msg is None):
            position_msg = self.last_position_msg
        else:
            self.last_position_msg = position_msg

        # Publish the next steering angle and the next velocity sets. Note that
        # The velocity is published (very) shortly after the position data, which means
        # that the velocity could lag in very tight update loops.
        #self.get_logger().info(f'Publishing steering angle data: "{position_msg}"')
        self.drive_module_steering_angle_publisher.publish(position_msg)

        #self.get_logger().info(f'Publishing velocity angle data: "{velocity_msg}"')
        self.drive_module_velocity_publisher.publish(velocity_msg)
        """



        ################################################################################################################
        # CK
        steering_angle_values_deg = [math.degrees(a) for a in steering_angle_values]
        #print(f'steering angles: {steering_angle_values_deg}')
        # Retain last angle if infinite
        for i in range(0, len(steering_angle_values_deg)):
            if not math.isfinite(steering_angle_values_deg[i]):
                steering_angle_values_deg[i] = self.last_steering_angle_values_deg[i]

        self.last_steering_angle_values_deg = steering_angle_values_deg

        # Scale the outgoing velocity values
        vel_scalar = 100.
        drive_velocity_values = [v * vel_scalar for v in drive_velocity_values]


        # Flip the right wheels on outgoing
        ### NOTE: This might only be for sim
        drive_velocity_values[0] *= -1.
        drive_velocity_values[2] *= -1.
        ###


        quad_zero = [0.]*4
        self.publish_joint_command(self.steering_joint_names + self.drive_joint_names, positions=steering_angle_values_deg + quad_zero,
                                   velocities=quad_zero + drive_velocity_values, effort=[float(100.)]*8)

        ################################################################################################################
        self.last_control_update_send_at = self.last_recorded_time

    def write_log(self, text: str):
        self.get_logger().info(text)


    def publish_joint_command(self, joint_names, positions=None, velocities=None, effort=None):
        joint_state = JointState()
        joint_state.name = joint_names
        joint_state.header.stamp = self.get_clock().now().to_msg()

        if positions is not None:
            joint_state.position = positions
        if velocities is not None:
            joint_state.velocity = velocities
        if effort is not None:
            joint_state.effort = effort

        # Publish the message to the topic
        self.joint_command_publisher.publish(joint_state)


def cap_velocity_from_max_acceleration(prev_msg: Twist, curr_msg: Twist, max_accel: float, dt: float) -> Twist:
    """
    Limits the velocity components of the current Twist message to a maximum acceleration.

    :param prev_msg: Previous Twist message containing the last commanded velocities.
    :param curr_msg: Current Twist message containing the desired velocities.
    :param max_accel: Maximum allowable acceleration (linear and angular).
    :param dt: Time elapsed between the previous and current message (in seconds).
    :return: New Twist message with limited velocities based on the maximum acceleration.
    """
    # Create a new Twist message for the limited velocities
    limited_msg = Twist()

    # Calculate the maximum change in velocity allowed (acceleration * time)
    max_delta_v = max_accel * dt

    # Limit linear velocity in X
    delta_vx = curr_msg.linear.x - prev_msg.linear.x
    if abs(delta_vx) > max_delta_v:
        limited_msg.linear.x = prev_msg.linear.x + math.copysign(max_delta_v, delta_vx)
    else:
        limited_msg.linear.x = curr_msg.linear.x

    # Limit linear velocity in Y
    delta_vy = curr_msg.linear.y - prev_msg.linear.y
    if abs(delta_vy) > max_delta_v:
        limited_msg.linear.y = prev_msg.linear.y + math.copysign(max_delta_v, delta_vy)
    else:
        limited_msg.linear.y = curr_msg.linear.y

    # Limit angular velocity in Z
    delta_vz = curr_msg.angular.z - prev_msg.angular.z
    if abs(delta_vz) > max_delta_v:
        limited_msg.angular.z = prev_msg.angular.z + math.copysign(max_delta_v, delta_vz)
    else:
        limited_msg.angular.z = curr_msg.angular.z

    return limited_msg

def main(args=None):
    rclpy.init(args=args)

    pub = SwerveController()

    rclpy.spin(pub)
    pub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

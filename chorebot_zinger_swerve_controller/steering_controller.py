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
import itertools
import math

from typing import Callable, List

from .profile import TransientVariableProfile
from rclpy.time import Time

# local
from .control import BodyMotionCommand, DriveModuleMotionCommand, InvalidMotionCommandException, MotionCommand
from .control_model import difference_between_angles, SimpleFourWheelSteeringControlModel
from .control_profile import BodyMotionProfile, DriveModuleStateProfile
from .drive_module import DriveModule
from .states import BodyState, DriveModuleDesiredValues, DriveModuleMeasuredValues

class DriveModuleDesiredValuesProfilePoint():

    def __init__(self, time: float, drive_module_states: List[DriveModuleDesiredValues]):
        self.time_since_start_of_profile = time
        self.drive_module_states = drive_module_states


# CK
#
# Constants (should be parameters)
#
# If wheel module steering angle diff > than this value and robot is not stopped, stop robot before commanding angle (radians)
ACCEPTABLE_INDIVIDUAL_ROTATION_DIFF_WHILE_MOVING_THRESHOLD = math.radians(30.)
# If wheel module desired steering angles are opposing, every module must reach this threshold before robot can move again
ACCEPTABLE_OPPOSING_ROTATION_DIFF_THRESHOLD = math.radians(2.)
# Wheel module is considered stopped if abs velocity < this value (m/s)
STOPPED_EPSILON = 0.001
# If either body velocities are <= to these values, don't impose acceleration scaling (m/s)
MIN_LINEAR_VELOCITY_SAFETY_THRESHOLD = 0.15
MIN_ANGULAR_VELOCITY_SAFETY_THRESHOLD = 0.13


### CK
class MovingAverage:
    """
    Track a moving average over a fixed window
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.data_points = []

    def add(self, value):
        """
        Append a point
        :param value: Number or np.array
        :return: New average
        """
        self.data_points.append(value)

        if len(self.data_points) > self.window_size:
            self.data_points = self.data_points[-self.window_size:]

        return self.average()

    def average(self):
        if len(self.data_points) == 0:
            return 0
        return sum(self.data_points) / len(self.data_points)

    def has_data(self):
        return len(self.data_points) > 0

class BodyVelocityStateAverage:
    """
    Average body state over multiple time steps
    """
    # We use 4 because the joint messages come separately and just insert previous value for half the updates,
    # So this effectively averages over two actual sensor readings
    WINDOW_SIZE = 4

    def __init__(self):
        self.linear_vel_x = MovingAverage(BodyVelocityStateAverage.WINDOW_SIZE)
        self.linear_vel_y = MovingAverage(BodyVelocityStateAverage.WINDOW_SIZE)
        self.angular_vel_z = MovingAverage(BodyVelocityStateAverage.WINDOW_SIZE)

    def has_data(self):
        return self.linear_vel_x.has_data() and self.linear_vel_y.has_data() and self.angular_vel_z.has_data()

class ModuleFollowsBodySteeringController():

    def __init__(
            self,
            drive_modules: List[DriveModule],
            motion_profile_func: Callable[[float, float], TransientVariableProfile],
            max_angular_body_acceleration: float,
            logger: Callable[[str], None]):
        # Get the geometry for the robot
        self.modules = drive_modules
        self.motion_profile_func = motion_profile_func
        self.max_angular_body_acceleration = max_angular_body_acceleration
        self.logger = logger

        # Use a simple control model for the time being. Just need something that roughly works
        self.control_model = SimpleFourWheelSteeringControlModel(self.modules)

        # Store the current (estimated) state of the body
        self.body_state: BodyState = BodyState(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        # Store the current (measured) state of the drive modules
        self.module_states: List[DriveModuleMeasuredValues] = [
            DriveModuleMeasuredValues(
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
            ) for drive_module in drive_modules
        ]

        self.previous_module_states: List[DriveModuleMeasuredValues] = [
            DriveModuleMeasuredValues(
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
            ) for drive_module in drive_modules
        ]

        # Profiles
        self.body_profile: BodyMotionProfile = None
        self.module_profile_from_command: DriveModuleStateProfile = None

         # Keep track of our position in time so that we can figure out where on the current
        # profile we should be
        self.current_time_in_seconds = 0.0
        self.profile_was_started_at_time_in_seconds = 0.0
        self.last_state_update_time: Time = None
        self.min_time_for_profile: float = 0.0

        # flags
        self.is_executing_body_profile: bool = False
        self.is_executing_module_profile: bool = False
        # CK
        self.velocity_state_avg: BodyVelocityStateAverage = BodyVelocityStateAverage()
        self.had_illegal_rotation: bool = False
        self.had_illegal_acceleration: bool = False

    def body_state_at_current_time(self) -> BodyState:
        return self.body_state

    def drive_module_states_at_current_time(self) -> List[DriveModuleMeasuredValues]:
        return self.module_states

    def drive_module_state_at_profile_time(self, time_fraction: float) -> List[DriveModuleDesiredValues]:
        # self.logger(
        #     'Determining profile values at time fraction {}'.format(time_fraction)
        # )

        result: List[DriveModuleDesiredValues] = []
        if self.is_executing_body_profile:
            body_state = self.body_profile.body_motion_at(time_fraction)

            ###### CK
            # Logic for imposing acceleration scaling
            if abs(body_state.angular_acceleration.z) > self.max_angular_body_acceleration and \
                    (abs(body_state.linear_velocity.x) > MIN_LINEAR_VELOCITY_SAFETY_THRESHOLD or
                     abs(body_state.linear_velocity.y) > MIN_LINEAR_VELOCITY_SAFETY_THRESHOLD ) and \
                    abs(body_state.angular_velocity.z) > MIN_ANGULAR_VELOCITY_SAFETY_THRESHOLD:
                accel_scalar = abs(self.max_angular_body_acceleration / (2. * body_state.angular_acceleration.z))
                self.logger(f'Body angular acceleration {body_state.angular_acceleration.z} too high, scaling velocity by {accel_scalar}')
                #self.logger(f'  Cur Body: - vx:{body_state.linear_velocity.x}, vy: {body_state.linear_velocity.y}, vo: {body_state.angular_velocity.z}')
                body_state.linear_velocity.x *= accel_scalar
                body_state.linear_velocity.y *= accel_scalar
                body_state.angular_velocity.z *= accel_scalar
                #self.logger(f'  New Body: - vx:{body_state.linear_velocity.x}, vy: {body_state.linear_velocity.y}, vo: {body_state.angular_velocity.z}')


            # If we previously had an illegal rotation, check current velocity of each drive module
            stopped = True
            m: DriveModuleMeasuredValues
            #print('Post illegal module velocities: ' +  str([m.drive_velocity_in_module_coordinates.x for m in self.module_states]))
            for m in self.module_states:
                if not math.isclose(m.drive_velocity_in_module_coordinates.x, 0., abs_tol=STOPPED_EPSILON):
                    stopped = False

            self.had_illegal_rotation = False
            self.had_illegal_acceleration = False
            ######

            drive_module_desired_values = self.control_model.state_of_wheel_modules_from_body_motion(body_state)
            for i in range(len(self.modules)):
                # Wheels are moving. We don't know what kind of movement yet though, so figure out if:
                # - The wheel are moving at some significant velocity, in that case pick the state that most
                #   closely matches the current state, i.e. match the drive velocity and the steering angle as
                #   close as possible
                # - The wheel is moving slowly, in that case we may just be close to the moment where the wheel
                #   stops moving (either just before it does that, or just after). This is where we could potentially
                #   flip directions (or we might just have flipped directions)
                #   - If we have just flipped directions then we should probably continue in the same way (but maybe not)

                #previous_state_for_module = self.previous_module_states[i]

                current_state_for_module = self.module_states[i]
                current_steering_angle = current_state_for_module.orientation_in_body_coordinates.z
                current_velocity = current_state_for_module.drive_velocity_in_module_coordinates.x

                #previous_rotation_difference = current_steering_angle - previous_state_for_module.orientation_in_body_coordinates.z
                #previous_velocity_difference = current_velocity - previous_state_for_module.drive_velocity_in_module_coordinates.x

                states_for_module = drive_module_desired_values[i]

                first_state_rotation_difference = difference_between_angles(current_steering_angle, states_for_module[0].steering_angle_in_radians)
                second_state_rotation_difference = difference_between_angles(current_steering_angle, states_for_module[1].steering_angle_in_radians)

                first_state_velocity_difference = states_for_module[0].drive_velocity_in_meters_per_second - current_velocity
                second_state_velocity_difference = states_for_module[1].drive_velocity_in_meters_per_second - current_velocity

                # print(f'First state diffs (rot,vel): {first_state_rotation_difference} / {first_state_velocity_difference}')
                # print(f'Second state diffs (rot,vel): {second_state_rotation_difference} / {second_state_velocity_difference}')

                # Possibilities:
                # - first velocity change and first orientation change are the smallest -> pick the first state
                # - second velocity change and second orientation change are the smallest -> pick the second state
                # - first velocity change is larger and second orientation change is larger -> Bad state. Pick the one with the least relative change?

                if abs(first_state_rotation_difference) <= abs(second_state_rotation_difference):
                    # CK
                    if abs(first_state_rotation_difference) > ACCEPTABLE_INDIVIDUAL_ROTATION_DIFF_WHILE_MOVING_THRESHOLD:
                        self.had_illegal_rotation = self.had_illegal_rotation or not stopped
                        self.logger(f'Rotation diff {math.degrees(first_state_rotation_difference):.1f} for {states_for_module[0].name} exceeds threshold. Deny forward velocity: {self.had_illegal_rotation}')

                    #print(f'{states_for_module[0].name} Diff: {math.degrees(first_state_rotation_difference)}')

                    if abs(first_state_velocity_difference) <= abs(second_state_velocity_difference):
                        # first rotation and velocity change are the smallest, so take the first state
                        result.append(states_for_module[0])
                        # self.logger(
                        #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 1'.format(
                        #         self.modules[i].name,
                        #         current_steering_angle,
                        #         current_velocity,
                        #         states_for_module[0].steering_angle_in_radians,
                        #         states_for_module[0].drive_velocity_in_meters_per_second,
                        #         states_for_module[1].steering_angle_in_radians,
                        #         states_for_module[1].drive_velocity_in_meters_per_second
                        #     )
                        # )
                    else:
                        if math.isclose(abs(first_state_rotation_difference), abs(second_state_rotation_difference), rel_tol=1e-7, abs_tol=1e-7):
                            # first rotation is equal to the second rotation
                            # first velocity larger than the second velocity.
                            # pick the second state
                            result.append(states_for_module[1])
                            # self.logger(
                            #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 2'.format(
                            #         self.modules[i].name,
                            #         current_steering_angle,
                            #         current_velocity,
                            #         states_for_module[0].steering_angle_in_radians,
                            #         states_for_module[0].drive_velocity_in_meters_per_second,
                            #         states_for_module[1].steering_angle_in_radians,
                            #         states_for_module[1].drive_velocity_in_meters_per_second
                            #     )
                            # )
                        else:
                            # first rotation is the smallest but second velocity is the smallest
                            result.append(states_for_module[0])
                            # self.logger(
                            #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 1'.format(
                            #         self.modules[i].name,
                            #         current_steering_angle,
                            #         current_velocity,
                            #         states_for_module[0].steering_angle_in_radians,
                            #         states_for_module[0].drive_velocity_in_meters_per_second,
                            #         states_for_module[1].steering_angle_in_radians,
                            #         states_for_module[1].drive_velocity_in_meters_per_second
                            #     )
                            # )
                else:
                    # CK
                    if abs(second_state_rotation_difference) > ACCEPTABLE_INDIVIDUAL_ROTATION_DIFF_WHILE_MOVING_THRESHOLD:
                        self.had_illegal_rotation = self.had_illegal_rotation or not stopped
                        self.logger(f'Rotation diff {math.degrees(second_state_rotation_difference):.1f} for {states_for_module[0].name} exceeds threshold. Deny forward velocity: {self.had_illegal_rotation}')
                    #print(f'{states_for_module[1].name}  Diff: {math.degrees(second_state_rotation_difference)}')

                    if abs(second_state_velocity_difference) <= abs(first_state_velocity_difference):
                        # second rotation and velocity change are the smallest, so take the second state
                        result.append(states_for_module[1])
                        # self.logger(
                        #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 2'.format(
                        #         self.modules[i].name,
                        #         current_steering_angle,
                        #         current_velocity,
                        #         states_for_module[0].steering_angle_in_radians,
                        #         states_for_module[0].drive_velocity_in_meters_per_second,
                        #         states_for_module[1].steering_angle_in_radians,
                        #         states_for_module[1].drive_velocity_in_meters_per_second
                        #     )
                        # )
                    else:
                        if math.isclose(abs(first_state_rotation_difference), abs(second_state_rotation_difference), rel_tol=1e-7, abs_tol=1e-7):
                            # second rotation is equal to the first rotation
                            # second velocity larger than the first velocity.
                            # pick the first state
                            result.append(states_for_module[0])
                            # self.logger(
                            #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 1'.format(
                            #         self.modules[i].name,
                            #         current_steering_angle,
                            #         current_velocity,
                            #         states_for_module[0].steering_angle_in_radians,
                            #         states_for_module[0].drive_velocity_in_meters_per_second,
                            #         states_for_module[1].steering_angle_in_radians,
                            #         states_for_module[1].drive_velocity_in_meters_per_second
                            #     )
                            # )
                        else:
                            # second rotation is the smallest but first velocity is the smallest
                            result.append(states_for_module[1])
                            # self.logger(
                            #     'module: {} - current state [{} rad, {} m/s]. Options - 1) [{} rad, {} m/s] - 2) [{} rad, {} m/s]. Selected option 2'.format(
                            #         self.modules[i].name,
                            #         current_steering_angle,
                            #         current_velocity,
                            #         states_for_module[0].steering_angle_in_radians,
                            #         states_for_module[0].drive_velocity_in_meters_per_second,
                            #         states_for_module[1].steering_angle_in_radians,
                            #         states_for_module[1].drive_velocity_in_meters_per_second
                            #     )
                            # )

            #
            # Opposing wheel check
            #
            desired_steering_angles = [r.steering_angle_in_radians for r in result]
            # This is the maximum difference in desired steering angles between all wheels
            max_steering_angle_diff =  max(desired_steering_angles) - min(desired_steering_angles)
            #self.logger(f'MAX STEERING DIFF: {math.degrees(max_steering_angle_diff)}')

            if max_steering_angle_diff > math.pi / 2. and not stopped and not self.had_illegal_rotation:
                # In special case where wheels position themselves at opposing angles, we must remain stopped until all
                # wheels have reached their desired steering angle
                # Check if wheels have reached angle
                for i in range(len(self.modules)):
                    current_state_for_module = self.module_states[i]
                    current_steering_angle = current_state_for_module.orientation_in_body_coordinates.z
                    # Diff between current and desired
                    module_rotation_difference = difference_between_angles(current_steering_angle, result[i].steering_angle_in_radians)
                    if module_rotation_difference > ACCEPTABLE_OPPOSING_ROTATION_DIFF_THRESHOLD:
                        self.logger(f'Opposing wheel steering targets not reached and robot is moving. Stopping robot...')
                        self.had_illegal_rotation = True


            #
            # Handle illegal rotation and scale acceleration
            #
            for i in range(len(self.modules)):
                desired_state = result[i]
                if self.had_illegal_rotation:
                    # If illegal, we command a full stop of the drive motors for this cycle, until they get closer to their desired values at a future timestep
                    #current_steering_angle = self.module_states[i].orientation_in_body_coordinates.z
                    #desired_state.steering_angle_in_radians = current_steering_angle
                    desired_state.drive_velocity_in_meters_per_second = 0.

                #self.logger(f'Target: {desired_state.drive_velocity_in_meters_per_second} current: {self.module_states[i].drive_velocity_in_module_coordinates.x}')

                # Compute acceleration scaling on per module basis
                # TODO: We should technically divide this by the time fraction diff
                module_target_accel = desired_state.drive_velocity_in_meters_per_second - self.module_states[i].drive_velocity_in_module_coordinates.x
                if abs(module_target_accel) > self.modules[i].drive_motor_maximum_acceleration:
                    self.had_illegal_acceleration = True
                    if module_target_accel > 0.:
                        new_target_velocity = self.module_states[i].drive_velocity_in_module_coordinates.x + self.modules[i].drive_motor_maximum_acceleration
                    else:
                        new_target_velocity = self.module_states[i].drive_velocity_in_module_coordinates.x - self.modules[i].drive_motor_maximum_acceleration

                    new_target_velocity = math.copysign(min(abs(new_target_velocity), self.modules[i].drive_motor_maximum_velocity), new_target_velocity)

                    self.logger(f'Drive acceleration for {desired_state.name} exceeds {self.modules[i].drive_motor_maximum_acceleration} threshold at current velocity {self.module_states[i].drive_velocity_in_module_coordinates.x:.3f},\n    scaling target velocity from {desired_state.drive_velocity_in_meters_per_second:.3f} to {new_target_velocity:.3f}')
                    desired_state.drive_velocity_in_meters_per_second = new_target_velocity

        else:
            # CK
            assert False, 'Only body motion commands supported'
            for drive_module in self.modules:
                state = self.module_profile_from_command.value_for_module_at(drive_module.name, time_fraction)
                result.append(DriveModuleDesiredValues(
                    state.name,
                    state.orientation_in_body_coordinates.z,
                    state.drive_velocity_in_module_coordinates.x
                ))

        return result

    # Returns the state of the drive modules to required to match the current profile at the given
    # time.
    def drive_module_state_at_future_time(self, future_time_in_seconds:float) -> List[DriveModuleDesiredValues]:
        if self.body_profile is None and self.module_profile_from_command is None:
            return []

        time_from_start_of_profile = future_time_in_seconds - self.profile_was_started_at_time_in_seconds
        # self.logger(
        #     'Determining profile values at {}'.format(time_from_start_of_profile)
        # )

        profile_time = self.body_profile.time_span() if self.is_executing_body_profile else self.module_profile_from_command.time_span()
        time_fraction = time_from_start_of_profile / profile_time

        result: List[DriveModuleDesiredValues] = self.drive_module_state_at_profile_time(time_fraction)
        return result

    def drive_module_profile_points_from_now_till_end(self, starting_time: float) -> List[DriveModuleDesiredValuesProfilePoint]:
        if self.body_profile is None and self.module_profile_from_command is None:
            return []

        # for now distribute the points equally. But really what we should be doing is putting more points in
        # places where the second or third derivatives change sign

        time_from_start_of_profile = starting_time - self.profile_was_started_at_time_in_seconds

        profile_time = self.body_profile.time_span() if self.is_executing_body_profile else self.module_profile_from_command.time_span()

        division_count = 10
        time_fraction_start = (time_from_start_of_profile / profile_time)
        time_fraction_end = 1.0 * division_count

        # Take time steps of 1/100 of the total profile time. Find the next time step we should take and
        # then find the number of steps we have left to take in the current
        next_time_step = int(self.round_up(time_fraction_start, 1.0 / division_count) * division_count)

        result: List[DriveModuleDesiredValuesProfilePoint] = []
        for step in range(next_time_step, time_fraction_end, 1):
            time_fraction = (float(step)) / division_count
            time = profile_time * time_fraction
            states = self.drive_module_state_at_profile_time(time_fraction)

            point = DriveModuleDesiredValuesProfilePoint(time, states)
            result.append(point)

        return result

    # Updates the currently stored desired body state. On the next time tick the
    # drive module trajectory will be updated to match the new desired end state.
    def on_desired_state_update(self, desired_motion: MotionCommand):
        if isinstance(desired_motion, BodyMotionCommand):
            trajectory = BodyMotionProfile(
                self.body_state,
                desired_motion.to_body_state(self.control_model),
                desired_motion.time_for_motion(),
                self.motion_profile_func)
            self.body_profile = trajectory

            self.is_executing_body_profile = True
            self.is_executing_module_profile = False

            self.logger(
                'Starting body motion profile with starting state [[x:{}, y:{}, o:{}],[vx:{}, vy:{}, vo:{}]] and desired end state [vx:{}, vy:{},vo:{}]'.format(
                    self.body_state.position_in_world_coordinates.x,
                    self.body_state.position_in_world_coordinates.y,
                    self.body_state.orientation_in_world_coordinates.z,
                    self.body_state.motion_in_body_coordinates.linear_velocity.x,
                    self.body_state.motion_in_body_coordinates.linear_velocity.y,
                    self.body_state.motion_in_body_coordinates.angular_velocity.z,
                    desired_motion.linear_velocity.x,
                    desired_motion.linear_velocity.y,
                    desired_motion.angular_velocity.z,)
            )
        else:
            if isinstance(desired_motion, DriveModuleMotionCommand):
                trajectory = DriveModuleStateProfile(self.modules, desired_motion.time_for_motion(), self.motion_profile_func)
                trajectory.set_current_state(self.module_states)
                trajectory.set_desired_end_state(desired_motion.to_drive_module_state(self.control_model)[0])
                self.module_profile_from_command = trajectory

                self.is_executing_body_profile = False
                self.is_executing_module_profile = True

                self.logger(
                    'Starting module motion profile with starting state {} and desired end state {}'.format(self.body_state, desired_motion)
                )
            else:
                raise InvalidMotionCommandException()

        self.profile_was_started_at_time_in_seconds = self.current_time_in_seconds
        self.min_time_for_profile = desired_motion.time_for_motion()

    # Updates the currently stored drive module state
    def on_state_update(self, current_module_states: List[DriveModuleMeasuredValues], msg_timestamp: Time):
        if current_module_states is None:
            raise TypeError()

        if len(current_module_states) != len(self.modules):
            raise ValueError()

        self.previous_module_states = self.module_states
        self.module_states = current_module_states

         # Calculate the current body state
        body_motion = self.control_model.body_motion_from_wheel_module_states(self.module_states)
        # self.logger(
        #     'Body motion: linear [{}, {}, {}]; rotation [{}, {}, {}]'.format(
        #         body_motion.linear_velocity.x,
        #         body_motion.linear_velocity.y,
        #         body_motion.linear_velocity.z,
        #         body_motion.angular_velocity.x,
        #         body_motion.angular_velocity.y,
        #         body_motion.angular_velocity.z,
        #     )
        # )
        # CK
        self.velocity_state_avg.linear_vel_x.add(body_motion.linear_velocity.x)
        self.velocity_state_avg.linear_vel_y.add(body_motion.linear_velocity.y)
        self.velocity_state_avg.angular_vel_z.add(body_motion.angular_velocity.z)

        # CK
        if msg_timestamp is not None and self.last_state_update_time is not None:
            # Using the node clock time results in inaccurate position data, use message timestamp instead
            #time_step_in_seconds = self.current_time_in_seconds - self.last_state_update_time
            time_step_in_seconds = (msg_timestamp - self.last_state_update_time).nanoseconds / 1e9

            # self.logger(
            #     'Determining body position at {}. Last update at: {}. Time delta: {}'.format(self.current_time_in_seconds, self.last_state_update_time, time_step_in_seconds)
            # )

            # Position
            # local_x_distance = time_step_in_seconds * 0.5 * (self.body_state.motion_in_body_coordinates.linear_velocity.x + body_motion.linear_velocity.x)
            # local_y_distance = time_step_in_seconds * 0.5 * (self.body_state.motion_in_body_coordinates.linear_velocity.y + body_motion.linear_velocity.y)
            local_x_distance = time_step_in_seconds * self.velocity_state_avg.linear_vel_x.average()
            local_y_distance = time_step_in_seconds * self.velocity_state_avg.linear_vel_y.average()

            # Orientation
            #global_orientation = self.body_state.orientation_in_world_coordinates.z + time_step_in_seconds * 0.5 * (self.body_state.motion_in_body_coordinates.angular_velocity.z + body_motion.angular_velocity.z)
            global_orientation = self.body_state.orientation_in_world_coordinates.z + time_step_in_seconds * self.velocity_state_avg.angular_vel_z.average()

            # Acceleration
            local_x_acceleration = 0.0
            local_y_acceleration = 0.0
            orientation_acceleration = 0.0
            if not math.isclose(time_step_in_seconds, 0.0, abs_tol=1e-4, rel_tol=1e-4):
                local_x_acceleration = (body_motion.linear_velocity.x - self.body_state.motion_in_body_coordinates.linear_velocity.x) / time_step_in_seconds
                local_y_acceleration = (body_motion.linear_velocity.y - self.body_state.motion_in_body_coordinates.linear_velocity.y) / time_step_in_seconds
                orientation_acceleration = (body_motion.angular_velocity.z - self.body_state.motion_in_body_coordinates.angular_velocity.z) / time_step_in_seconds

            # print('Body:')
            # print(f'   vel - vx: {body_motion.linear_velocity.x}, vy: {body_motion.linear_velocity.y}, vo: {body_motion.angular_velocity.z}')
            # print(f'   accel - ax: {local_x_acceleration}, ay: {local_y_acceleration}, ao: {orientation_acceleration}')

            # Jerk
            local_x_jerk = 0.0
            local_y_jerk = 0.0
            orientation_jerk = 0.0
            if not math.isclose(time_step_in_seconds, 0.0, abs_tol=1e-4, rel_tol=1e-4):
                local_x_jerk = (local_x_acceleration - self.body_state.motion_in_body_coordinates.linear_acceleration.x) / time_step_in_seconds
                local_y_jerk = (local_y_acceleration - self.body_state.motion_in_body_coordinates.linear_acceleration.y) / time_step_in_seconds
                orientation_jerk = (orientation_acceleration - self.body_state.motion_in_body_coordinates.angular_acceleration.z) / time_step_in_seconds

            self.body_state = BodyState(
                self.body_state.position_in_world_coordinates.x + local_x_distance * math.cos(global_orientation) - local_y_distance * math.sin(global_orientation),
                self.body_state.position_in_world_coordinates.y + local_x_distance * math.sin(global_orientation) + local_y_distance * math.cos(global_orientation),
                global_orientation,
                body_motion.linear_velocity.x,
                body_motion.linear_velocity.y,
                body_motion.angular_velocity.z,
                local_x_acceleration,
                local_y_acceleration,
                orientation_acceleration,
                local_x_jerk,
                local_y_jerk,
                orientation_jerk
            )
        else:
            self.body_state = BodyState(
                self.body_state.position_in_world_coordinates.x,
                self.body_state.position_in_world_coordinates.y,
                self.body_state.orientation_in_world_coordinates.z,
                body_motion.linear_velocity.x,
                body_motion.linear_velocity.y,
                body_motion.angular_velocity.z,
                0.,
                0.,
                0.,
                0.,
                0.,
                0.
            )

        # self.logger(
        #     'position: [{}, {}, {}] orientation [[{}, {}, {}]]'.format(
        #         self.body_state.position_in_world_coordinates.x,
        #         self.body_state.position_in_world_coordinates.y,
        #         self.body_state.position_in_world_coordinates.z,
        #         self.body_state.orientation_in_world_coordinates.x,
        #         self.body_state.orientation_in_world_coordinates.y,
        #         self.body_state.orientation_in_world_coordinates.z,
        #     )
        # )

        #self.last_state_update_time = self.current_time_in_seconds
        if msg_timestamp is not None:
            self.last_state_update_time = msg_timestamp

    # On clock tick, determine if we need to recalculate the trajectories for the drive modules
    def on_tick(self, current_time_in_seconds: float):
        self.current_time_in_seconds = current_time_in_seconds

    def round_down(self, num: float, to: float) -> float:
        if num < 0:
            return -self.round_up(-num, to)
        mod = math.fmod(num, to)

        return num if math.isclose(mod, to) else num - mod

    def round_up(self, num: float, to: float) -> float:
        if num < 0:
            return -self.round_down(-num, to)

        down = self.round_down(num, to)

        return num if num == down else down + to


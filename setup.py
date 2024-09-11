from glob import glob
import os
from setuptools import setup

package_name = 'chorebot_zinger_swerve_controller'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include all the config files
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='petrik',
    maintainer_email='petrikvandervelde@gmail.com',
    description='Defines a four wheel independent steering, aka swerve, controller for the zinger robot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'swerve_controller = chorebot_zinger_swerve_controller.swerve_controller:main',
        ],
    },
)

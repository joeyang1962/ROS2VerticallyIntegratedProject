from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'dcsl_f1tenth'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nick',
    maintainer_email='nickzhang@gatech.edu',
    description='F1TENTH utilities and controllers',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_planner = dcsl_f1tenth.local_planner:main',
            'pure_pursuit_controller = dcsl_f1tenth.pure_pursuit_controller:main',
            'stanley_controller = dcsl_f1tenth.stanley_controller:main',
            'vicon = dcsl_f1tenth.vicon:main',
            'hand_follower = dcsl_f1tenth.hand_follower:main',
            'odometry_validation = dcsl_f1tenth.odometry_validation:main',
        ],
    },
)

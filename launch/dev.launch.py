"""
dev.launch.py

Separate processes (no zero-copy), verbose logging, VIC disabled for easier
debugging.  Do NOT use in production — inter-process copies are expensive.

_____________________________________________________________________________
@file: dev.launch.py
@brief: Launch file for development/debugging: separate processes, verbose logging, VIC disabled.
@author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
@company: Sintez.LLC
@date: 2026-03-01
_____________________________________________________________________________
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg = get_package_share_directory('isaac_ros_object_detection')
    params_file = os.path.join(pkg, 'config', 'params.yaml')

    return LaunchDescription([
        Node(
            package='isaac_ros_object_detection',
            executable='object_detection_node',   # standalone binary from main.cpp
            name='object_detection_node',
            output='screen',
            parameters=[
                params_file
            ],
            arguments=['--ros-args', '--log-level', 'DEBUG'],
        ),
    ])
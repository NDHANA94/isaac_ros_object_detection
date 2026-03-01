"""
annotated_image_publisher.launch.py

Standalone launch file for the AnnotatedImagePublisherNode.
Subscribes to camera image + detections, publishes annotated frames.

_____________________________________________________________________________
@file: annotated_image_publisher.launch.py
@brief: Launch file for the AnnotatedImagePublisherNode composable ROS2 node.
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
    pkg_dir = get_package_share_directory('isaac_ros_object_detection')
    config_file = os.path.join(pkg_dir, 'config', 'annotated_img_publisher_params.yaml')

    return LaunchDescription([
        Node(
            package='isaac_ros_object_detection',
            executable='annotated_image_publisher_node',
            name='annotated_image_publisher_node',
            output='screen',
            parameters=[config_file],
        )
    ])

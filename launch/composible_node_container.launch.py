"""
composable_node_container.launch.py

Pipeline:
    DnnImageEncoder
        └─> ObjectDetectionNode

All parameters are loaded from config/obj_det_params.yaml.
No CLI arguments — edit the params file to change behaviour.

_____________________________________________________________________________
@file: composable_node_container.launch.py
@brief: Launch file for the full object detection pipeline: camera + encoder + detector.
@author: WM Nipun Dhananjaya (nipun.dhananjaya@gmail.com)
@company: Sintez.LLC
@date: 2026-03-01
_____________________________________________________________________________
"""

import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get(d: dict, *keys, default=None):
    """Safely walk nested dict keys; return default on any miss."""
    try:
        for k in keys:
            d = d[k]
        return d
    except Exception:
        return default


def generate_launch_description():
    this_pkg     = get_package_share_directory('isaac_ros_object_detection')

    obj_det_params_file = os.path.join(this_pkg, 'config', 'obj_det_params.yaml')
    annotated_img_pub_params = os.path.join(this_pkg, 'config', 'annotated_img_publisher_params.yaml')
    
    dnn_img_enc_params_file = os.path.join(this_pkg, 'config', 'dnn_img_enc_params.yaml')
    image_input_topic = _get(_load_yaml(dnn_img_enc_params_file), 'dnn_image_encoder_node', 'ros__parameters', 'image_input_topic', default='/image_raw')
    tensor_output_topic = _get(_load_yaml(dnn_img_enc_params_file), 'dnn_image_encoder_node', 'ros__parameters', 'tensor_output_topic', default='/encoded_tensor')

    arducam_pkg = get_package_share_directory('isaac_ros_arducam_b0573')
    arducam_params_file = os.path.join(arducam_pkg, 'config', 'params.yaml')

    publish_annotated_image = _get(_load_yaml(obj_det_params_file), 'object_detection_node', 'ros__parameters', 'publish_annotated_image', default=False)

    return LaunchDescription([

        # ── Composable container: camera + encoder + detector ─────────────────
        ComposableNodeContainer(
            name='isaac_ros_object_detection_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            composable_node_descriptions=[
                
                # ── 1. Camera node ───────────────────────────────────────────────
                # Add your camera node here, e.g. a V4L2 driver or ROS2 wrapper for your camera.
                # Make sure to set the output topic name and QoS in the camera's own params file
                # to match the DnnImageEncoder's subscription parameters in obj_det_params.yaml.
                ComposableNode(
                    package='isaac_ros_arducam_b0573',
                    plugin='nvidia::isaac_ros::arducam::ArducamB0573Node',
                    name='arducam_b0573_node',
                    parameters=[arducam_params_file],
                ),
                

                # ── 2. DnnImageEncoder node ───────────────────────────────────
                # All parameters (encoder type, input topic name, QoS) are read
                # directly from dnn_img_enc_params.yaml — nothing is overridden here.
                ComposableNode(
                    package='isaac_ros_dnn_image_encoder',
                    plugin='nvidia::isaac_ros::dnn_inference::DnnImageEncoderNode',
                    name='dnn_image_encoder_node',
                    parameters=[dnn_img_enc_params_file],
                    remappings=[
                        ('image', image_input_topic),
                        ('encoded_tensor', tensor_output_topic),
                    ],
                ),
        
                
                # ── 3. Object Detection node ───────────────────────────────────
                # All parameters (engine path, thresholds, topic names, QoS) are
                # read directly from obj_det_params.yaml — nothing is overridden here.
                ComposableNode(
                    package='isaac_ros_object_detection',
                    plugin='nvidia::isaac_ros::object_detection::ObjectDetectionNode',
                    name='object_detection_node',
                    parameters=[obj_det_params_file],
                ),

                # ── 4. Annotated Image Publisher node (optional) ──────────────
                # Loaded only when publish_annotated_image=True in obj_det_params.yaml.
                *([ComposableNode(
                    package='isaac_ros_object_detection',
                    plugin='nvidia::isaac_ros::object_detection::AnnotatedImagePublisherNode',
                    name='annotated_image_publisher_node',
                    parameters=[annotated_img_pub_params],
                )] if publish_annotated_image else []),
            ],
            output='screen',
        ),
    ])

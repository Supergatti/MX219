#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    left_camera_id = int(os.environ.get('ARGUS_LEFT_CAMERA_ID', '0'))
    right_camera_id = int(os.environ.get('ARGUS_RIGHT_CAMERA_ID', '1'))
    left_module_id = int(os.environ.get('ARGUS_LEFT_MODULE_ID', '0'))
    right_module_id = int(os.environ.get('ARGUS_RIGHT_MODULE_ID', '1'))
    enable_web_monitor = os.environ.get('ENABLE_WEB_MONITOR', '1') == '1'
    swap_lr = os.environ.get('ARGUS_SWAP_LR', '0') == '1'
    left_camera_info_url = os.environ.get(
        'ARGUS_LEFT_CAMERA_INFO_URL', 'file:///workspace/src/calib/left.yaml'
    )
    right_camera_info_url = os.environ.get(
        'ARGUS_RIGHT_CAMERA_INFO_URL', 'file:///workspace/src/calib/right.yaml'
    )

    if swap_lr:
        left_camera_id, right_camera_id = right_camera_id, left_camera_id
        left_module_id, right_module_id = right_module_id, left_module_id
        left_camera_info_url, right_camera_info_url = right_camera_info_url, left_camera_info_url

    # 双路 IMX219 场景使用两个 ArgusMonoNode，加载固定 camera_info 避免坏标定
    left_argus_node = ComposableNode(
        name='argus_left',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusMonoNode',
        namespace='',
        parameters=[{
            'camera_id': left_camera_id,
            'module_id': left_module_id,
            'camera_info_url': left_camera_info_url,
        }],
        remappings=[
            ('left/image_raw', '/left/image_raw'),
            ('left/camera_info', '/left/camera_info'),
        ],
    )

    right_argus_node = ComposableNode(
        name='argus_right',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusMonoNode',
        namespace='',
        parameters=[{
            'camera_id': right_camera_id,
            'module_id': right_module_id,
            'camera_info_url': right_camera_info_url,
        }],
        remappings=[
            ('left/image_raw', '/right/image_raw'),
            ('left/camera_info', '/right/camera_info'),
        ],
    )

    argus_mono_container = ComposableNodeContainer(
        name='argus_mono_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[left_argus_node, right_argus_node],
        output='screen',
        arguments=['--ros-args', '--log-level', 'info'],
    )

    # Web 监控节点：浏览器查看双目画面和 SLAM 位姿，访问 http://<Jetson-IP>:8090
    # 注意：不要在此处使用 env={...} 覆盖环境变量，否则会导致 PYTHONPATH 丢失找不到 rclpy
    web_monitor = ExecuteProcess(
        cmd=['python3', '/workspace/src/web_monitor.py'],
        output='screen'
    )

    # 启动 Isaac ROS Visual SLAM 组件节点
    # 插件类来自 isaac_ros_visual_slam 包
    visual_slam_node = ComposableNode(
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        name='visual_slam_node',
        parameters=[{
            'enable_imu_fusion': False,
        }],
        remappings=[
            ('/visual_slam/image_0', '/left/image_raw'),
            ('/visual_slam/camera_info_0', '/left/camera_info'),
            ('/visual_slam/image_1', '/right/image_raw'),
            ('/visual_slam/camera_info_1', '/right/camera_info'),
        ]
    )

    container = ComposableNodeContainer(
        name='isaac_vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[visual_slam_node],
        output='screen'
    )

    actions = [
        argus_mono_container,
        TimerAction(period=5.0, actions=[container]),
    ]
    if enable_web_monitor:
        actions.append(web_monitor)

    return LaunchDescription(actions)

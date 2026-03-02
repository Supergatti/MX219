#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description() -> LaunchDescription:
    # 启动纯 Python 双目发布节点（发布 /left|right/image_rect 和 /left|right/camera_info）
    stereo_pub = ExecuteProcess(
        cmd=['python3', '/workspace/src/imx219_stereo_publisher.py'],
        output='screen'
    )

    # Web 监控节点：浏览器查看双目画面和 SLAM 位姿，访问 http://<Jetson-IP>:8080
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
            ('stereo_camera/left/image', '/left/image_rect'),
            ('stereo_camera/left/camera_info', '/left/camera_info'),
            ('stereo_camera/right/image', '/right/image_rect'),
            ('stereo_camera/right/camera_info', '/right/camera_info'),
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

    return LaunchDescription([
        stereo_pub,
        web_monitor,
        container,
    ])

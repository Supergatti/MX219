from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch Isaac ROS Visual SLAM using Argus Camera input.
    """
    return LaunchDescription([
        # 启动 Visual SLAM 节点
        Node(
            package='isaac_ros_visual_slam',
            executable='isaac_ros_visual_slam_node',
            name='visual_slam_node',
            parameters=[{
                'enable_rectified_pose': True,
                'denoise_input_images': False,
                'rectified_images': True,
                'enable_slam_visualization': True,
                'enable_observations_view': True,
                'enable_landmarks_view': True,
                'camera_optical_frame': 'camera_optical_frame',
                'base_frame': 'base_link',
            }],
            remappings=[
                ('visual_slam/image_0', '/camera/left/image_raw'),
                ('visual_slam/camera_info_0', '/camera/left/camera_info'),
                ('visual_slam/image_1', '/camera/right/image_raw'),
                ('visual_slam/camera_info_1', '/camera/right/camera_info'),
            ]
        )
    ])

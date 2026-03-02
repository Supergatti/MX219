import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """
    Launch file for NVIDIA Isaac ROS Visual SLAM on IMX219 Dual.
    
    Subscribes to:
    - /left/image_raw (sensor_msgs/Image)
    - /right/image_raw (sensor_msgs/Image)
    - /left/camera_info (sensor_msgs/CameraInfo)
    - /right/camera_info (sensor_msgs/CameraInfo)
    
    Outputs:
    - /visual_slam/tracking/odometry (nav_msgs/Odometry)
    - TF (odom -> base_link)
    
    How to convert IMX219 GStreamer stream to ROS topics:
    -----------------------------------------------------
    Method A (Argus API - Recommended for Jetson):
    Use the 'isaac_ros_argus_camera' package which directly utilizes the Jetson ISP.
    Example:
    ros2 launch isaac_ros_argus_camera isaac_ros_argus_camera_stereo.launch.py \
        sensor_id_left:=0 sensor_id_right:=1
    
    Method B (GStreamer with v4l2_camera):
    Use 'v4l2_camera' node with a GStreamer backend.
    Example:
    ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:="/dev/video0"
    
    Note: Due to the 0.55x scale error, the camera_info.P[0] and P[5] should be correctly 
    scaled, and the baseline (P[3]) should reflect the 33mm (0.033m) distance.
    """

    visual_slam_node = Node(
        package='isaac_ros_visual_slam',
        executable='isaac_ros_visual_slam_node',
        name='visual_slam_node',
        parameters=[{
            'enable_rectified_pose': True,
            'denoise_input_images': False,
            'rectified_images_flag': True,
            'enable_slam_visualization': True,
            'enable_observations_view': True,
            'enable_landmarks_view': True,
            'base_frame': 'base_link',
            'visual_slam_frame': 'odom',
        }],
        remappings=[
            ('visual_slam/left/image_rect', '/left/image_raw'),
            ('visual_slam/right/image_rect', '/right/image_raw'),
            ('visual_slam/left/camera_info', '/left/camera_info'),
            ('visual_slam/right/camera_info', '/right/camera_info'),
        ]
    )

    return LaunchDescription([
        visual_slam_node
    ])

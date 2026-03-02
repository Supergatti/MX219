# NVIDIA Isaac ROS VSLAM for IMX219 Dual

## Prerequisites

1.  A Jetson device (Orin Nano, etc.) with NVIDIA JetPack.
2.  Isaac ROS development Docker image (`isaac_ros_dev-aarch64`).
3.  IMX219 Dual Camera connected.

## How to Run

### Step 1: Launch the Docker Container
Run the provided script to start the Isaac ROS container with camera access:
```bash
./launch_docker.sh
```

### Step 2: Source ROS2 and Run the Launch File (Inside Docker)
Once inside the container, setup your ROS2 environment and run the SLAM node:
```bash
# Inside Docker
source /opt/ros/humble/setup.bash
# Navigate to the project directory
cd /workspaces/isaac_ros-dev/projects/03_Isaac_ROS_VSLAM
# Use ros2 launch, NOT python3!
ros2 launch imx219_vslam.launch.py
```

## Important Notes

1.  **Launch Method**: Never run `.launch.py` files directly with `python3`. Always use `ros2 launch`.
2.  **Display**: If you are using SSH, ensure you have X11 forwarding enabled (`ssh -X ...`) or a monitor connected to the Jetson (`export DISPLAY=:0`).
3.  **Topics**: This launch file assumes you are already publishing camera topics to `/left/image_raw` and `/right/image_raw`. You can use the `isaac_ros_argus_camera` package as suggested in the launch file comments.

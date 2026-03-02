#!/bin/bash

# Launch Isaac ROS Development Docker Image
# Mounting the current project directory and camera devices (/dev/video0, /dev/video1)

PROJECT_DIR="/home/jetson/Desktop/MX219/03_Isaac_ROS_VSLAM"
IMAGE_NAME="isaac_ros_dev-aarch64"

echo "Starting Isaac ROS Docker Container..."

# Ensure we have access to X server for visualization
xhost +local:root

docker run -it --rm \
    --privileged \
    --network host \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v $PROJECT_DIR:/workspaces/isaac_ros-dev/projects/03_Isaac_ROS_VSLAM \
    $IMAGE_NAME

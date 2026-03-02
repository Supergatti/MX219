#!/bin/bash
set -e  # 遇到错误立即退出

echo "=========================================="
echo "Starting Isaac ROS VSLAM Container..."
echo "=========================================="

# 1. 允许 X11 转发 (如果本地有显示器)
xhost +local:root || echo "Warning: xhost command failed, GUI might not work."

# 2. 定义变量
IMAGE_NAME="isaac_ros_dev-aarch64:latest"
WORK_DIR="/home/jetson/Desktop/MX219/03_Isaac_ROS_VSLAM"

# 3. 检查目录是否存在
if [ ! -d "$WORK_DIR" ]; then
    echo "Error: Directory $WORK_DIR does not exist!"
    exit 1
fi

# 4. 运行 Docker
# 使用 --rm 退出后自动清理容器
# 使用 --net=host 共享网络
# 使用 --privileged 获取所有设备访问权限
echo "Running docker command..."
docker run -it --rm \
    --net=host \
    --privileged \
    --runtime=nvidia \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /tmp/argus_socket:/tmp/argus_socket \
    -v ${WORK_DIR}:/workspaces/isaac_ros-dev/src/vslam_launch \
    -w /workspaces/isaac_ros-dev \
    ${IMAGE_NAME} \
    /bin/bash -c "
        echo 'Container started successfully.'
        source /opt/ros/humble/setup.bash
        
        echo 'Building workspace...'
        colcon build --packages-select isaac_ros_visual_slam --symlink-install || echo 'Build skipped or failed, trying to run anyway...'
        
        source install/setup.bash
        
        echo 'Launching VSLAM...'
        # 这里直接运行 launch 文件，如果不成功会打印错误
        ros2 launch src/vslam_launch/launch_vslam.py
    "

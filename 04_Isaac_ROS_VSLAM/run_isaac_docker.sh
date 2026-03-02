#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ISAAC_IMAGE:-isaac_ros_dev-aarch64:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 允许容器访问本机 X11（若无图形界面可忽略相关告警）
if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

echo "Starting Isaac ROS container: ${IMAGE}"
echo "Workspace mount: ${SCRIPT_DIR} -> /workspace/src"

if [[ ! -S /tmp/argus_socket ]]; then
  echo "[WARN] /tmp/argus_socket not found on host, CSI Argus camera may fail." >&2
fi

# 检查必要的 Jetson 设备节点是否存在，缺失时打印警告而不中断
for dev in /dev/nvmap /dev/nvhost-ctrl /dev/nvhost-ctrl-gpu /dev/nvhost-prof-gpu /dev/nvhost-vic /dev/nvhost-nvdec /dev/nvhost-nvenc; do
  if [[ ! -e "$dev" ]]; then
    echo "[WARN] $dev not found on host, skipping." >&2
  fi
done

# 动态拼接仅存在的设备节点参数
EXTRA_DEVS=()
for dev in /dev/nvmap /dev/nvhost-ctrl /dev/nvhost-ctrl-gpu /dev/nvhost-prof-gpu /dev/nvhost-vic /dev/nvhost-nvdec /dev/nvhost-nvenc; do
  [[ -e "$dev" ]] && EXTRA_DEVS+=("--device" "$dev")
done

docker run -it --rm \
  --runtime nvidia \
  --privileged \
  --entrypoint bash \
  --net=host \
  --ipc=host \
  --pid=host \
  --device /dev/video0 \
  --device /dev/video1 \
  "${EXTRA_DEVS[@]}" \
  -v /tmp/argus_socket:/tmp/argus_socket \
  -e DISPLAY="${DISPLAY:-}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e ROS_DOMAIN_ID=99 \
  -e EGL_PLATFORM=surfaceless \
  -e LIBGL_ALWAYS_SOFTWARE=0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${SCRIPT_DIR}:/workspace/src" \
  -w /workspace/src \
  "${IMAGE}" \
  -c "source /opt/ros/humble/setup.bash && \
      echo '[INFO] Starting VSLAM launch...' && \
      ros2 launch /workspace/src/isaac_vslam_run.launch.py || \
      (echo '[ERROR] Launch failed, dropping to bash for debug'; exec bash)"

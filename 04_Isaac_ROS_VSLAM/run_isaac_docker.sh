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
  -e WEB_MONITOR_REFRESH_MS="${WEB_MONITOR_REFRESH_MS:-33}" \
  -e WEB_MONITOR_MAX_FPS="${WEB_MONITOR_MAX_FPS:-30.0}" \
  -e WEB_MONITOR_SCALE="${WEB_MONITOR_SCALE:-0.5}" \
  -e WEB_MONITOR_JPEG_QUALITY="${WEB_MONITOR_JPEG_QUALITY:-60}" \
  -e WEB_MONITOR_DISPARITY_SCALE="${WEB_MONITOR_DISPARITY_SCALE:-0.4}" \
  -e WEB_MONITOR_ENABLE_DISPARITY="${WEB_MONITOR_ENABLE_DISPARITY:-0}" \
  -e WEB_MONITOR_ENABLE_FEATURES="${WEB_MONITOR_ENABLE_FEATURES:-0}" \
  -e ARGUS_LEFT_CAMERA_ID="${ARGUS_LEFT_CAMERA_ID:-0}" \
  -e ARGUS_RIGHT_CAMERA_ID="${ARGUS_RIGHT_CAMERA_ID:-1}" \
  -e ARGUS_LEFT_MODULE_ID="${ARGUS_LEFT_MODULE_ID:-0}" \
  -e ARGUS_RIGHT_MODULE_ID="${ARGUS_RIGHT_MODULE_ID:-1}" \
  -e ARGUS_MODULE_ID="${ARGUS_MODULE_ID:--1}" \
  -e ARGUS_MODE="${ARGUS_MODE:-4}" \
  -e ARGUS_FSYNC_TYPE="${ARGUS_FSYNC_TYPE:-0}" \
  -e ARGUS_FRAMERATE="${ARGUS_FRAMERATE:-30.0}" \
  -e ARGUS_SWAP_LR="${ARGUS_SWAP_LR:-0}" \
  -e ARGUS_LEFT_CAMERA_INFO_URL="${ARGUS_LEFT_CAMERA_INFO_URL:-file:///workspace/src/calib/left.yaml}" \
  -e ARGUS_RIGHT_CAMERA_INFO_URL="${ARGUS_RIGHT_CAMERA_INFO_URL:-file:///workspace/src/calib/right.yaml}" \
  -e ENABLE_WEB_MONITOR="${ENABLE_WEB_MONITOR:-1}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${SCRIPT_DIR}:/workspace/src" \
  -w /workspace/src \
  "${IMAGE}" \
  -c "python3 - <<'PY' || true
import socket
try:
    s = socket.socket(socket.AF_UNIX)
    s.connect('/tmp/argus_restart_socket')
    s.send(b'RESTART_SERVICE')
    s.close()
except Exception:
    pass
PY
      source /opt/ros/humble/setup.bash && \
      echo '[INFO] Starting VSLAM launch...' && \
      ros2 launch /workspace/src/isaac_vslam_run.launch.py || \
      (echo '[ERROR] Launch failed, dropping to bash for debug'; exec bash)"

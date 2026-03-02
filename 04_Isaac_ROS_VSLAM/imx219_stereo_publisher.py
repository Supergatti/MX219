#!/usr/bin/env python3
import os
# 无头/SSH 环境下让 Argus EGL 使用 surfaceless 模式，不依赖桌面 display
os.environ.setdefault('EGL_PLATFORM', 'surfaceless')

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from builtin_interfaces.msg import Time


class Imx219StereoPublisher(Node):
    def __init__(self) -> None:
        super().__init__('imx219_stereo_publisher')

        self.declare_parameter('width', 1280)
        self.declare_parameter('height', 720)
        self.declare_parameter('fps', 15)
        self.declare_parameter('prefer_argus', True)
        self.declare_parameter('fx', 800.0)
        self.declare_parameter('fy', 800.0)
        self.declare_parameter('cx', 640.0)
        self.declare_parameter('cy', 360.0)
        self.declare_parameter('baseline_m', 0.06 * 0.55)

        self.width = int(self.get_parameter('width').value)
        self.height = int(self.get_parameter('height').value)
        self.fps = int(self.get_parameter('fps').value)
        self.prefer_argus = bool(self.get_parameter('prefer_argus').value)
        self.fx = float(self.get_parameter('fx').value)
        self.fy = float(self.get_parameter('fy').value)
        self.cx = float(self.get_parameter('cx').value)
        self.cy = float(self.get_parameter('cy').value)
        self.baseline_m = float(self.get_parameter('baseline_m').value)

        self.left_image_pub = self.create_publisher(Image, '/left/image_rect', 10)
        self.right_image_pub = self.create_publisher(Image, '/right/image_rect', 10)
        self.left_info_pub = self.create_publisher(CameraInfo, '/left/camera_info', 10)
        self.right_info_pub = self.create_publisher(CameraInfo, '/right/camera_info', 10)

        self.left_cap = None
        self.right_cap = None
        self._open_cameras()

        timer_period = 1.0 / max(self.fps, 1)
        self.timer = self.create_timer(timer_period, self.publish_once)

        self.get_logger().info(
            f'Stereo publisher started: {self.width}x{self.height}@{self.fps}Hz, '
            f'baseline={self.baseline_m:.4f}m'
        )

    def _gstreamer_argus_pipeline(self, sensor_id: int) -> str:
        return (
            f'nvarguscamerasrc sensor-id={sensor_id} ! '
            f'video/x-raw(memory:NVMM), width={self.width}, height={self.height}, '
            f'format=(string)NV12, framerate=(fraction){self.fps}/1 ! '
            'nvvidconv flip-method=0 ! '
            'video/x-raw, format=(string)BGRx ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! '
            'appsink drop=true max-buffers=1 sync=false'
        )

    def _gstreamer_v4l2_pipeline(self, device_id: int) -> str:
        return (
            f'v4l2src device=/dev/video{device_id} ! '
            f'video/x-raw, width={self.width}, height={self.height}, framerate={self.fps}/1 ! '
            'videoconvert ! '
            'video/x-raw, format=(string)BGR ! '
            'appsink drop=true max-buffers=1 sync=false'
        )

    def _open_cameras(self) -> None:
        if self.prefer_argus:
            self.left_cap = cv2.VideoCapture(self._gstreamer_argus_pipeline(sensor_id=0), cv2.CAP_GSTREAMER)
            self.right_cap = cv2.VideoCapture(self._gstreamer_argus_pipeline(sensor_id=1), cv2.CAP_GSTREAMER)
            if self.left_cap.isOpened() and self.right_cap.isOpened():
                self.get_logger().info('Using Argus camera pipeline (nvarguscamerasrc).')
                return
            self.get_logger().warning('Argus pipeline unavailable, fallback to v4l2src.')
            if self.left_cap is not None:
                self.left_cap.release()
            if self.right_cap is not None:
                self.right_cap.release()

        self.left_cap = cv2.VideoCapture(self._gstreamer_v4l2_pipeline(device_id=0), cv2.CAP_GSTREAMER)
        self.right_cap = cv2.VideoCapture(self._gstreamer_v4l2_pipeline(device_id=1), cv2.CAP_GSTREAMER)
        if self.left_cap.isOpened() and self.right_cap.isOpened():
            self.get_logger().info('Using v4l2 camera pipeline (/dev/video0,/dev/video1).')
            return

        raise RuntimeError('Failed to open IMX219 stereo cameras via both Argus and v4l2 pipelines.')

    def publish_once(self) -> None:
        ret_l, frame_l = self.left_cap.read()
        ret_r, frame_r = self.right_cap.read()

        if not ret_l or frame_l is None or not ret_r or frame_r is None:
            self.get_logger().warning('Failed to read stereo frames this cycle.')
            return

        now = self.get_clock().now().to_msg()

        left_image_msg = self._to_image_msg(frame_l, 'left_camera_optical_frame', now)
        right_image_msg = self._to_image_msg(frame_r, 'right_camera_optical_frame', now)

        left_info = self._camera_info(is_right=False, stamp=now)
        right_info = self._camera_info(is_right=True, stamp=now)

        self.left_image_pub.publish(left_image_msg)
        self.right_image_pub.publish(right_image_msg)
        self.left_info_pub.publish(left_info)
        self.right_info_pub.publish(right_info)

    def _to_image_msg(self, frame, frame_id: str, stamp: Time) -> Image:
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * frame.shape[2]
        msg.data = frame.tobytes()
        return msg

    def _camera_info(self, is_right: bool, stamp: Time) -> CameraInfo:
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = 'right_camera_optical_frame' if is_right else 'left_camera_optical_frame'
        info.width = self.width
        info.height = self.height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        info.k = [
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        ]

        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        tx = -self.fx * self.baseline_m if is_right else 0.0
        info.p = [
            self.fx, 0.0, self.cx, tx,
            0.0, self.fy, self.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        return info

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, 'left_cap') and self.left_cap is not None:
                self.left_cap.release()
            if hasattr(self, 'right_cap') and self.right_cap is not None:
                self.right_cap.release()
        finally:
            return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = Imx219StereoPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

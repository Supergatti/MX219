#!/usr/bin/env python3
"""
Web 监控节点：订阅 ROS2 话题，通过浏览器实时查看双目画面和 SLAM 位姿。
访问：http://<Jetson-IP>:8080
"""
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image


# --------------------------------------------------------------------------- #
#  全局状态（ROS 线程写，HTTP 线程读）
# --------------------------------------------------------------------------- #
_lock = threading.Lock()
_state = {
    'left_jpg': None,
    'right_jpg': None,
    'pose': {'x': 0.0, 'y': 0.0, 'z': 0.0,
             'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0},
    'tracking': 'WAITING',
    'frame_count': 0,
}

WEB_PORT = int(os.environ.get('WEB_MONITOR_PORT', '8080'))


def _encode_jpg(frame: np.ndarray, quality: int = 70) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# --------------------------------------------------------------------------- #
#  ROS2 节点
# --------------------------------------------------------------------------- #
class WebMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('web_monitor')
        self.bridge = CvBridge()

        self.create_subscription(Image, '/left/image_rect',
                                 lambda msg: self._on_image(msg, 'left_jpg'), 5)
        self.create_subscription(Image, '/right/image_rect',
                                 lambda msg: self._on_image(msg, 'right_jpg'), 5)

        # Isaac ROS VSLAM 输出话题（兼容两种可能的名字）
        self.create_subscription(
            Odometry, '/visual_slam/tracking/odometry',
            self._on_odom, 5)
        self.create_subscription(
            PoseStamped, '/visual_slam/tracking/slam_pose',
            self._on_pose_stamped, 5)

        self.get_logger().info(
            f'Web monitor started → http://0.0.0.0:{WEB_PORT}')

    def _on_image(self, msg: Image, key: str) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            jpg = _encode_jpg(frame)
            with _lock:
                _state[key] = jpg
                _state['frame_count'] += 1
        except Exception as e:
            self.get_logger().warning(f'Image convert error: {e}')

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with _lock:
            _state['pose'] = {'x': p.x, 'y': p.y, 'z': p.z,
                              'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        with _lock:
            _state['pose'] = {'x': p.x, 'y': p.y, 'z': p.z,
                              'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'


# --------------------------------------------------------------------------- #
#  HTTP 请求处理
# --------------------------------------------------------------------------- #
_PLACEHOLDER = _encode_jpg(
    np.zeros((240, 320, 3), dtype=np.uint8))

_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IMX219 VSLAM Monitor</title>
  <style>
    body {{ background:#111; color:#eee; font-family:monospace; margin:0; padding:16px; }}
    h2 {{ color:#0cf; margin:0 0 12px; }}
    .grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
    img {{ border:2px solid #333; max-width:100%; }}
    #pose {{ background:#1a1a2e; border:1px solid #0cf; padding:12px 20px;
             margin:12px 0; border-radius:6px; font-size:14px; line-height:1.8; }}
    .label {{ color:#0cf; }}
    .ok {{ color:#0f0; }} .warn {{ color:#f80; }} .err {{ color:#f44; }}
  </style>
  <script>
    function refresh() {{
      document.getElementById('left').src='/stream/left?t='+Date.now();
      document.getElementById('right').src='/stream/right?t='+Date.now();
      fetch('/api/pose').then(r=>r.json()).then(d=>{{
        let cls = d.tracking==='TRACKING'?'ok':(d.tracking==='WAITING'?'warn':'err');
        document.getElementById('pose').innerHTML =
          '<span class="label">状态：</span><span class="'+cls+'">'+d.tracking+'</span><br>'+
          '<span class="label">位置：</span>'+
          'X='+d.pose.x.toFixed(3)+'m  '+
          'Y='+d.pose.y.toFixed(3)+'m  '+
          'Z='+d.pose.z.toFixed(3)+'m<br>'+
          '<span class="label">帧计数：</span>'+d.frame_count;
      }});
    }}
    setInterval(refresh, 200);
    window.onload = refresh;
  </script>
</head>
<body>
  <h2>IMX219 双目 VSLAM 实时监控</h2>
  <div id="pose">等待 SLAM 数据...</div>
  <div class="grid">
    <div><div style="color:#0cf">左目 /left/image_rect</div>
      <img id="left" width="640" height="360"></div>
    <div><div style="color:#0cf">右目 /right/image_rect</div>
      <img id="right" width="640" height="360"></div>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 静默 HTTP 日志

    def _send(self, code: int, ct: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == '/' or self.path.startswith('/?'):
            self._send(200, 'text/html; charset=utf-8', _HTML.encode())

        elif self.path.startswith('/stream/left'):
            with _lock:
                jpg = _state['left_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)

        elif self.path.startswith('/stream/right'):
            with _lock:
                jpg = _state['right_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)

        elif self.path.startswith('/api/pose'):
            import json
            with _lock:
                data = {
                    'tracking': _state['tracking'],
                    'pose': _state['pose'],
                    'frame_count': _state['frame_count'],
                }
            self._send(200, 'application/json',
                       json.dumps(data).encode())
        else:
            self._send(404, 'text/plain', b'Not Found')


# --------------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------------- #
def _run_http() -> None:
    server = HTTPServer(('0.0.0.0', WEB_PORT), Handler)
    server.serve_forever()


def main() -> None:
    rclpy.init()
    node = WebMonitorNode()

    t = threading.Thread(target=_run_http, daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

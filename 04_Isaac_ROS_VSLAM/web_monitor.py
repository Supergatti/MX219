#!/usr/bin/env python3
"""
Web 监控节点：订阅 ROS2 话题，通过浏览器实时查看：
  - 纠正方向后的双目画面
  - 双目视差图（深度可视化，TURBO 伪彩色：暖色=近，冷色=远）
  - ORB 特征点叠加（SLAM 候选追踪点可视化）
  - SLAM 位姿数值

访问：http://<Jetson-IP>:8080
"""
import json
import os
import queue
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

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
    'left_jpg':      None,   # 纠正方向后的原图 JPEG bytes
    'right_jpg':     None,
    'disparity_jpg': None,   # StereoSGBM 视差图 JPEG bytes
    'features_jpg':  None,   # ORB 特征点叠加 JPEG bytes
    'left_raw':      None,   # numpy BGR，供后处理用
    'right_raw':     None,
    'pose': {'x': 0.0, 'y': 0.0, 'z': 0.0,
             'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0},
    'tracking':    'WAITING',
    'frame_count': 0,
}

WEB_PORT = int(os.environ.get('WEB_MONITOR_PORT', '8080'))

# StereoBM — 比 SGBM 快 5-10x，适合嵌入式实时
_MIN_DISP   =  0
_NUM_DISP   = 64   # 必须是 16 的倍数
_BLOCK_SIZE = 15   # StereoBM 需要奇数且 >=5
_stereo = cv2.StereoBM_create(numDisparities=_NUM_DISP, blockSize=_BLOCK_SIZE)

# ORB 检测器（减少特征点数以加快速度）
_orb = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=6, edgeThreshold=10)

# 后处理任务队列：maxsize=1 保证只保留最新一帧，旧帧自动丢弃
_proc_queue: queue.Queue = queue.Queue(maxsize=1)


def _encode_jpg(frame: np.ndarray, quality: int = 75) -> bytes:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _compute_disparity(left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
    """StereoBM 视差图 → TURBO 伪彩色 BGR，降采样 0.5x 加速。"""
    try:
        scale = 0.5
        lh = cv2.resize(cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        rh = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        disp = _stereo.compute(lh, rh).astype(np.float32) / 16.0
        vis  = np.zeros_like(disp, dtype=np.uint8)
        valid = disp > _MIN_DISP
        if valid.any():
            d_min, d_max = disp[valid].min(), disp[valid].max()
            if d_max > d_min:
                vis[valid] = ((disp[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        return cv2.resize(colored, (left.shape[1], left.shape[0]))
    except Exception:
        return None


def _compute_features(left: np.ndarray) -> np.ndarray:
    """ORB 特征点叠加，返回带标注的 BGR 图。"""
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    kps  = _orb.detect(gray, None)
    out  = left.copy()
    cv2.drawKeypoints(out, kps, out,
                      color=(0, 255, 0),
                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(out, f'ORB features: {len(kps)}', (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def _proc_worker() -> None:
    """独立线程：从队列取最新帧对，做重计算，绝不阻塞 ROS 回调。"""
    while True:
        left, right = _proc_queue.get()  # 阻塞等待新帧
        try:
            disp_bgr = _compute_disparity(left, right)
            feat_bgr = _compute_features(left)
            with _lock:
                if disp_bgr is not None:
                    _state['disparity_jpg'] = _encode_jpg(disp_bgr)
                _state['features_jpg'] = _encode_jpg(feat_bgr)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  ROS2 节点
# --------------------------------------------------------------------------- #
class WebMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('web_monitor')
        self.bridge = CvBridge()

        self.create_subscription(Image, '/left/image_rect',
                                 lambda msg: self._on_image(msg, 'left'), 5)
        self.create_subscription(Image, '/right/image_rect',
                                 lambda msg: self._on_image(msg, 'right'), 5)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry',
                                 self._on_odom, 5)
        self.create_subscription(PoseStamped, '/visual_slam/tracking/slam_pose',
                                 self._on_pose_stamped, 5)

        # 启动后处理工作线程（守护线程，随主进程退出）
        t = threading.Thread(target=_proc_worker, daemon=True)
        t.start()
        self.get_logger().info(f'Web monitor → http://0.0.0.0:{WEB_PORT}')

    def _on_image(self, msg: Image, side: str) -> None:
        """ROS 回调：只做图像解码和 JPEG 编码，立即返回，绝不做重计算。"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            jpg   = _encode_jpg(frame)
            with _lock:
                _state[f'{side}_jpg'] = jpg
                _state[f'{side}_raw'] = frame
                _state['frame_count'] += 1
                left  = _state['left_raw']
                right = _state['right_raw']

            # 把最新帧对投入队列（非阻塞，满了就丢弃旧帧）
            if left is not None and right is not None:
                try:
                    _proc_queue.put_nowait((left.copy(), right.copy()))
                except queue.Full:
                    pass  # 工作线程还在处理上一帧，直接丢弃，不堵塞
        except Exception as e:
            self.get_logger().warning(f'Image error: {e}')

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with _lock:
            _state['pose']     = {'x': p.x, 'y': p.y, 'z': p.z,
                                  'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'

    def _on_pose_stamped(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        with _lock:
            _state['pose']     = {'x': p.x, 'y': p.y, 'z': p.z,
                                  'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'


# --------------------------------------------------------------------------- #
#  HTTP 服务
# --------------------------------------------------------------------------- #
_PLACEHOLDER = _encode_jpg(np.zeros((360, 640, 3), dtype=np.uint8))

_HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>IMX219 VSLAM Monitor</title>
  <style>
    *    { box-sizing:border-box; margin:0; padding:0; }
    body { background:#0d0d0d; color:#ddd; font-family:monospace; padding:12px; }
    h2   { color:#00ccff; margin-bottom:10px; font-size:18px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .cell { background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:8px; }
    .ct   { color:#00ccff; font-size:12px; margin-bottom:6px; }
    img   { width:100%; border-radius:4px; display:block; }
    #pose-bar {
      background:#111827; border:1px solid #00ccff; border-radius:6px;
      padding:10px 16px; margin-bottom:10px;
      display:flex; gap:24px; align-items:center; flex-wrap:wrap; font-size:13px;
    }
    .ok   { color:#00ff88 } .warn { color:#ffaa00 } .err { color:#ff4444 }
    .val  { color:#fff; font-weight:bold }
    .lbl  { color:#888; font-size:11px }
  </style>
  <script>
    function refresh() {
      var t = Date.now();
      ['left','right','disparity','features'].forEach(function(s) {
        document.getElementById('img_'+s).src = '/stream/'+s+'?t='+t;
      });
      fetch('/api/pose').then(function(r){return r.json();}).then(function(d){
        var cls = d.tracking==='TRACKING'?'ok':(d.tracking==='WAITING'?'warn':'err');
        document.getElementById('status').className = cls;
        document.getElementById('status').textContent = d.tracking;
        document.getElementById('px').textContent = d.pose.x.toFixed(3);
        document.getElementById('py').textContent = d.pose.y.toFixed(3);
        document.getElementById('pz').textContent = d.pose.z.toFixed(3);
        document.getElementById('fc').textContent = d.frame_count;
      });
    }
    setInterval(refresh, 200);
    window.onload = refresh;
  </script>
</head>
<body>
  <h2>IMX219 双目 VSLAM 实时监控</h2>
  <div id="pose-bar">
    <div><div class="lbl">SLAM 状态</div><span id="status" class="warn">WAITING</span></div>
    <div><div class="lbl">X (m)</div><span class="val" id="px">—</span></div>
    <div><div class="lbl">Y (m)</div><span class="val" id="py">—</span></div>
    <div><div class="lbl">Z (m)</div><span class="val" id="pz">—</span></div>
    <div><div class="lbl">帧计数</div><span class="val" id="fc">0</span></div>
  </div>
  <div class="grid">
    <div class="cell">
      <div class="ct">左目原图 /left/image_rect</div>
      <img id="img_left" alt="left">
    </div>
    <div class="cell">
      <div class="ct">右目原图 /right/image_rect</div>
      <img id="img_right" alt="right">
    </div>
    <div class="cell">
      <div class="ct">双目视差图（TURBO 伪彩色：暖色=近，冷色=远）</div>
      <img id="img_disparity" alt="disparity">
    </div>
    <div class="cell">
      <div class="ct">ORB 特征点（绿色圆圈 = SLAM 候选追踪点）</div>
      <img id="img_features" alt="features">
    </div>
  </div>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def _send(self, code: int, ct: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split('?')[0]
        if path == '/':
            self._send(200, 'text/html; charset=utf-8', _HTML.encode('utf-8'))
        elif path == '/stream/left':
            with _lock: jpg = _state['left_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)
        elif path == '/stream/right':
            with _lock: jpg = _state['right_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)
        elif path == '/stream/disparity':
            with _lock: jpg = _state['disparity_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)
        elif path == '/stream/features':
            with _lock: jpg = _state['features_jpg'] or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)
        elif path == '/api/pose':
            with _lock:
                data = {'tracking':    _state['tracking'],
                        'pose':        _state['pose'],
                        'frame_count': _state['frame_count']}
            self._send(200, 'application/json', json.dumps(data).encode())
        else:
            self._send(404, 'text/plain', b'Not Found')


# --------------------------------------------------------------------------- #
#  入口
# --------------------------------------------------------------------------- #
def main() -> None:
    rclpy.init()
    node   = WebMonitorNode()
    server = HTTPServer(('0.0.0.0', WEB_PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        server.shutdown()


if __name__ == '__main__':
    main()

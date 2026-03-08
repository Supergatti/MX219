#!/usr/bin/env python3
"""轻量 Web 监控节点（默认关闭重计算，优先保障实时）。"""
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image

WEB_PORT = int(os.environ.get('WEB_MONITOR_PORT', '8090'))
WEB_REFRESH_MS = int(os.environ.get('WEB_MONITOR_REFRESH_MS', '33'))
MONITOR_MAX_FPS = float(os.environ.get('WEB_MONITOR_MAX_FPS', '30.0'))
MONITOR_SCALE = float(os.environ.get('WEB_MONITOR_SCALE', '0.5'))
MONITOR_JPEG_QUALITY = int(os.environ.get('WEB_MONITOR_JPEG_QUALITY', '60'))
DISPARITY_SCALE = float(os.environ.get('WEB_MONITOR_DISPARITY_SCALE', '0.4'))
ENABLE_DISPARITY = os.environ.get('WEB_MONITOR_ENABLE_DISPARITY', '0') == '1'
ENABLE_FEATURES = os.environ.get('WEB_MONITOR_ENABLE_FEATURES', '0') == '1'
LEFT_IMAGE_TOPIC = os.environ.get('WEB_LEFT_IMAGE_TOPIC', '/left/image_raw')
RIGHT_IMAGE_TOPIC = os.environ.get('WEB_RIGHT_IMAGE_TOPIC', '/right/image_raw')

_lock = threading.Lock()
_proc_queue: queue.Queue = queue.Queue(maxsize=1)
_stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # type: ignore[attr-defined]
_orb = cv2.ORB_create(nfeatures=200, scaleFactor=1.2, nlevels=6, edgeThreshold=10)  # type: ignore[attr-defined]


def _encode_jpg(frame: np.ndarray, quality: int = 75) -> bytes:
    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b''


_PLACEHOLDER = _encode_jpg(np.zeros((360, 640, 3), dtype=np.uint8), 70)

_state = {
    'left_jpg': _PLACEHOLDER,
    'right_jpg': _PLACEHOLDER,
    'disparity_jpg': _PLACEHOLDER,
    'features_jpg': _PLACEHOLDER,
    'left_raw': None,
    'right_raw': None,
    'pose': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0},
    'tracking': 'WAITING',
    'frame_count': 0,
    'left_in_count': 0,
    'right_in_count': 0,
    'proc_count': 0,
    'fps': {'left_in': 0.0, 'right_in': 0.0, 'proc': 0.0},
    'perf': {'last_t': time.monotonic(), 'left_in_last': 0, 'right_in_last': 0, 'proc_last': 0},
}


def _msg_to_bgr(msg: Image) -> Optional[np.ndarray]:
    try:
        if msg.height == 0 or msg.width == 0:
            return None
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding == 'bgr8':
            frame = arr.reshape((msg.height, msg.width, 3))
            return frame
        if msg.encoding == 'rgb8':
            frame = arr.reshape((msg.height, msg.width, 3))
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if msg.encoding == 'mono8':
            frame = arr.reshape((msg.height, msg.width))
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # 回退：按 3 通道尝试，避免编码不一致时直接崩
        frame = arr.reshape((msg.height, msg.width, -1))
        if frame.shape[2] >= 3:
            return frame[:, :, :3]
        return None
    except Exception:
        return None


def _update_fps_locked(now: Optional[float] = None) -> None:
    if now is None:
        now = time.monotonic()
    perf = _state['perf']
    dt = now - perf['last_t']
    if dt < 0.5:
        return
    left_now = _state['left_in_count']
    right_now = _state['right_in_count']
    proc_now = _state['proc_count']
    _state['fps'] = {
        'left_in': (left_now - perf['left_in_last']) / dt,
        'right_in': (right_now - perf['right_in_last']) / dt,
        'proc': (proc_now - perf['proc_last']) / dt,
    }
    perf['last_t'] = now
    perf['left_in_last'] = left_now
    perf['right_in_last'] = right_now
    perf['proc_last'] = proc_now


def _compute_disparity(left: np.ndarray, right: np.ndarray) -> Optional[np.ndarray]:
    try:
        scale = max(0.1, min(DISPARITY_SCALE, 1.0))
        lh = cv2.resize(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        rh = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
        disp = _stereo.compute(lh, rh).astype(np.float32) / 16.0
        vis = np.zeros_like(disp, dtype=np.uint8)
        valid = disp > 0
        if valid.any():
            d_min, d_max = disp[valid].min(), disp[valid].max()
            if d_max > d_min:
                vis[valid] = ((disp[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
        return cv2.resize(colored, (left.shape[1], left.shape[0]))
    except Exception:
        return None


def _compute_features(left: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    kps = _orb.detect(gray, None)
    out = left.copy()
    cv2.drawKeypoints(out, kps, out, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(out, f'ORB features: {len(kps)}', (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return out


def _proc_worker() -> None:
    while True:
        left, right = _proc_queue.get()
        try:
            disp = _compute_disparity(left, right) if ENABLE_DISPARITY else None
            feat = _compute_features(left) if ENABLE_FEATURES else None
            with _lock:
                if disp is not None:
                    _state['disparity_jpg'] = _encode_jpg(disp, MONITOR_JPEG_QUALITY)
                if feat is not None:
                    _state['features_jpg'] = _encode_jpg(feat, MONITOR_JPEG_QUALITY)
                _state['proc_count'] += 1
                _update_fps_locked()
        except Exception:
            pass


class WebMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__('web_monitor')
        self.monitor_period_ns = int(1e9 / max(MONITOR_MAX_FPS, 0.1))
        self._last_frame_ns = {'left': 0, 'right': 0}
        # 使用默认可靠 QoS，避免与发布端 RELIABLE 不兼容导致无帧
        self.create_subscription(Image, LEFT_IMAGE_TOPIC, self._on_left_image, 10)
        self.create_subscription(Image, RIGHT_IMAGE_TOPIC, self._on_right_image, 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self._on_odom, 10)
        self.create_subscription(PoseStamped, '/visual_slam/tracking/slam_pose', self._on_pose, 10)

        t = threading.Thread(target=_proc_worker, daemon=True)
        t.start()
        self.get_logger().info(f'Web monitor -> http://0.0.0.0:{WEB_PORT}')

    def _on_image(self, msg: Image, side: str) -> None:
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_frame_ns[side] < self.monitor_period_ns:
            return
        self._last_frame_ns[side] = now_ns

        frame = _msg_to_bgr(msg)
        if frame is None:
            return
        scale = max(0.1, min(MONITOR_SCALE, 1.0))
        if scale < 0.999:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        jpg = _encode_jpg(frame, MONITOR_JPEG_QUALITY)
        if not jpg:
            return

        with _lock:
            _state[f'{side}_jpg'] = jpg
            _state[f'{side}_raw'] = frame
            _state['frame_count'] += 1
            if side == 'left':
                _state['left_in_count'] += 1
            else:
                _state['right_in_count'] += 1
            _update_fps_locked()
            left = _state['left_raw']
            right = _state['right_raw']

        if side == 'right' and (ENABLE_DISPARITY or ENABLE_FEATURES) and left is not None and right is not None:
            try:
                _proc_queue.put_nowait((left.copy(), right.copy()))
            except queue.Full:
                pass

    def _on_left_image(self, msg: Image) -> None:
        self._on_image(msg, 'left')

    def _on_right_image(self, msg: Image) -> None:
        self._on_image(msg, 'right')

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        with _lock:
            _state['pose'] = {'x': p.x, 'y': p.y, 'z': p.z, 'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'

    def _on_pose(self, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        with _lock:
            _state['pose'] = {'x': p.x, 'y': p.y, 'z': p.z, 'qx': q.x, 'qy': q.y, 'qz': q.z, 'qw': q.w}
            _state['tracking'] = 'TRACKING'


_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>IMX219 VSLAM Monitor</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}body{background:#0d0d0d;color:#ddd;font-family:monospace;padding:12px}
h2{color:#00ccff;margin-bottom:10px;font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.cell{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:8px}.ct{color:#00ccff;font-size:12px;margin-bottom:6px}
img{width:100%;border-radius:4px;display:block}#pose{background:#111827;border:1px solid #00ccff;border-radius:6px;padding:10px 16px;margin-bottom:10px;display:flex;gap:20px;flex-wrap:wrap;font-size:13px}
.ok{color:#00ff88}.warn{color:#ffaa00}.val{color:#fff;font-weight:bold}.lbl{color:#888;font-size:11px}
</style>
<script>
function refresh(){var t=Date.now();['left','right','disparity','features'].forEach(function(s){var e=document.getElementById('img_'+s);if(e){e.src='/stream/'+s+'?t='+t;}});
fetch('/api/pose').then(r=>r.json()).then(function(d){document.getElementById('status').className=d.tracking==='TRACKING'?'ok':'warn';
document.getElementById('status').textContent=d.tracking;document.getElementById('px').textContent=d.pose.x.toFixed(3);document.getElementById('py').textContent=d.pose.y.toFixed(3);document.getElementById('pz').textContent=d.pose.z.toFixed(3);
document.getElementById('fps_left').textContent=d.fps.left_in.toFixed(1);document.getElementById('fps_right').textContent=d.fps.right_in.toFixed(1);document.getElementById('fps_proc').textContent=d.fps.proc.toFixed(1);});
}
setInterval(refresh,__REFRESH_MS__);window.onload=refresh;
</script></head><body>
<h2>IMX219 双目 VSLAM 实时监控</h2>
<div id="pose">
<div><div class="lbl">SLAM 状态</div><span id="status" class="warn">WAITING</span></div>
<div><div class="lbl">X (m)</div><span class="val" id="px">0.000</span></div>
<div><div class="lbl">Y (m)</div><span class="val" id="py">0.000</span></div>
<div><div class="lbl">Z (m)</div><span class="val" id="pz">0.000</span></div>
<div><div class="lbl">左输入 FPS</div><span class="val" id="fps_left">0.0</span></div>
<div><div class="lbl">右输入 FPS</div><span class="val" id="fps_right">0.0</span></div>
<div><div class="lbl">后处理 FPS</div><span class="val" id="fps_proc">0.0</span></div>
</div>
<div class="grid">
<div class="cell"><div class="ct">左目原图 __LEFT_TOPIC__</div><img id="img_left"></div>
<div class="cell"><div class="ct">右目原图 __RIGHT_TOPIC__</div><img id="img_right"></div>
<div class="cell"><div class="ct">双目视差图</div><img id="img_disparity"></div>
<div class="cell"><div class="ct">ORB 特征点</div><img id="img_features"></div>
</div></body></html>""".replace('__REFRESH_MS__', str(WEB_REFRESH_MS)).replace('__LEFT_TOPIC__', LEFT_IMAGE_TOPIC).replace('__RIGHT_TOPIC__', RIGHT_IMAGE_TOPIC)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        return

    def _send(self, code: int, ct: str, data: bytes) -> None:
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        p = self.path.split('?')[0]
        if p == '/':
            self._send(200, 'text/html; charset=utf-8', _HTML.encode())
            return
        if p.startswith('/stream/'):
            key = p.split('/')[-1] + '_jpg'
            with _lock:
                jpg = _state.get(key, _PLACEHOLDER) or _PLACEHOLDER
            self._send(200, 'image/jpeg', jpg)
            return
        if p == '/api/pose':
            with _lock:
                _update_fps_locked()
                data = {
                    'tracking': _state['tracking'],
                    'pose': _state['pose'],
                    'fps': _state['fps'],
                    'heavy_enabled': {'disparity': ENABLE_DISPARITY, 'features': ENABLE_FEATURES},
                }
            self._send(200, 'application/json', json.dumps(data).encode())
            return
        self._send(404, 'text/plain', b'Not Found')


def main() -> None:
    rclpy.init()
    node = WebMonitorNode()
    server = ThreadingHTTPServer(('0.0.0.0', WEB_PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        rclpy.spin(node)
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        server.shutdown()


if __name__ == '__main__':
    main()

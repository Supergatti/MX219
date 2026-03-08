#!/usr/bin/env python3
import argparse
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


@dataclass
class ServerConfig:
    source: str
    cam0: str
    cam1: str
    width: int
    height: int
    fps: int
    flip_method: int
    swap_lr: bool
    jpeg_quality: int
    overlay: bool


def build_argus_pipeline(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> str:
    fm = int(flip_method)
    if fm not in (0, 1, 2, 3, 4, 5, 6, 7):
        fm = 2
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={fm} ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def build_v4l2_pipeline(device: str, width: int, height: int, fps: int) -> str:
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def open_camera(source: str, camera: str, width: int, height: int, fps: int, flip_method: int) -> cv2.VideoCapture:
    if source == "argus":
        pipeline = build_argus_pipeline(int(camera), width, height, fps, flip_method)
    else:
        pipeline = build_v4l2_pipeline(camera, width, height, fps)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: source={source}, camera={camera}, pipeline={pipeline}")
    return cap


def open_camera_with_fallback(source: str, camera: str, width: int, height: int, fps: int, flip_method: int) -> cv2.VideoCapture:
    if source != "argus":
        return open_camera(source, camera, width, height, fps, flip_method)

    profiles = [
        (width, height, fps),
        (1920, 1080, min(fps, 30)),
        (1640, 1232, min(fps, 30)),
        (1280, 720, min(fps, 30)),
        (640, 480, min(fps, 30)),
    ]
    seen = set()
    last_error: Optional[Exception] = None
    for w, h, f in profiles:
        profile = (w, h, f)
        if profile in seen:
            continue
        seen.add(profile)
        try:
            print(f"[INFO] 尝试打开相机 {camera}: {w}x{h}@{f}")
            return open_camera(source, camera, w, h, f, flip_method)
        except Exception as exc:
            last_error = exc
            print(f"[WARN] 相机 {camera} 打开失败: {exc}")
            time.sleep(0.3)

    raise RuntimeError(f"相机 {camera} 所有候选分辨率均失败，最后错误: {last_error}")


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def annotate(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def add_center_zoom_preview(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    crop_w = max(40, int(w * 0.28))
    crop_h = max(40, int(h * 0.28))
    x1 = w // 2 - crop_w // 2
    y1 = h // 2 - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    roi = out[y1:y2, x1:x2]
    if roi.size == 0:
        return out

    zoom_w = min(int(crop_w * 3.0), w // 3)
    zoom_h = min(int(crop_h * 3.0), h // 3)
    zoom = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    px1, py1 = 10, h - zoom_h - 10
    px2, py2 = px1 + zoom_w, py1 + zoom_h
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    out[py1:py2, px1:px2] = zoom
    cv2.rectangle(out, (px1, py1), (px2, py2), (255, 255, 0), 2)
    return out


def draw_focus_bar(frame: np.ndarray, score: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_w = int(w * 0.35)
    bar_h = 16
    x1, y1 = w - bar_w - 20, 18
    x2, y2 = x1 + bar_w, y1 + bar_h
    cv2.rectangle(out, (x1, y1), (x2, y2), (220, 220, 220), 1)
    ratio = max(0.0, min(1.0, score / 1000.0))
    fill = int(bar_w * ratio)
    color = (0, 0, 255) if ratio < 0.35 else (0, 255, 255) if ratio < 0.7 else (0, 255, 0)
    if fill > 0:
        cv2.rectangle(out, (x1 + 1, y1 + 1), (x1 + fill - 1, y2 - 1), color, -1)
    cv2.putText(out, "focus", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


class FrameHub:
    def __init__(self, config: ServerConfig):
        self.config = config
        self._lock = threading.Lock()
        self._running = False
        self._cap0: Optional[cv2.VideoCapture] = None
        self._cap1: Optional[cv2.VideoCapture] = None
        self._jpeg0: Optional[bytes] = None
        self._jpeg1: Optional[bytes] = None

    def start(self) -> None:
        self._cap0 = open_camera_with_fallback(
            self.config.source, self.config.cam0, self.config.width, self.config.height, self.config.fps, self.config.flip_method
        )
        time.sleep(0.3)
        self._cap1 = open_camera_with_fallback(
            self.config.source, self.config.cam1, self.config.width, self.config.height, self.config.fps, self.config.flip_method
        )
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True).start()

    def stop(self) -> None:
        self._running = False
        if self._cap0:
            self._cap0.release()
        if self._cap1:
            self._cap1.release()

    def _capture_loop(self) -> None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpeg_quality]
        while self._running:
            ok0, frame0 = self._cap0.read() if self._cap0 else (False, None)
            ok1, frame1 = self._cap1.read() if self._cap1 else (False, None)
            if not ok0 or not ok1:
                time.sleep(0.02)
                continue

            if self.config.source != "argus":
                frame0 = cv2.rotate(frame0, cv2.ROTATE_180)
                frame1 = cv2.rotate(frame1, cv2.ROTATE_180)
            if self.config.swap_lr:
                frame0, frame1 = frame1, frame0

            if self.config.overlay:
                score0 = focus_score(frame0)
                score1 = focus_score(frame1)
                frame0 = annotate(frame0, f"left sharpness={score0:.1f}")
                frame1 = annotate(frame1, f"right sharpness={score1:.1f}")
                frame0 = add_center_zoom_preview(frame0)
                frame1 = add_center_zoom_preview(frame1)
                frame0 = draw_focus_bar(frame0, score0)
                frame1 = draw_focus_bar(frame1, score1)

            ok0, jpg0 = cv2.imencode(".jpg", frame0, encode_param)
            ok1, jpg1 = cv2.imencode(".jpg", frame1, encode_param)
            if not ok0 or not ok1:
                continue

            with self._lock:
                self._jpeg0 = jpg0.tobytes()
                self._jpeg1 = jpg1.tobytes()

    def get_frame(self, side: str) -> Optional[bytes]:
        with self._lock:
            if side == "left":
                return self._jpeg0
            return self._jpeg1


class StereoHandler(BaseHTTPRequestHandler):
    hub: FrameHub = None

    def do_GET(self) -> None:
        if self.path in ["/", "/index.html"]:
            return self._serve_index()
        if self.path == "/stream/left":
            return self._serve_stream("left")
        if self.path == "/stream/right":
            return self._serve_stream("right")

        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()
        self.wfile.write(b"Not Found")

    def log_message(self, format: str, *args) -> None:
        return

    def _serve_index(self) -> None:
        html = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>MX219 Stereo Web Preview</title>
  <style>
    body { margin: 0; font-family: sans-serif; background: #111; color: #ddd; }
    .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }
    .card { background: #1b1b1b; border: 1px solid #333; border-radius: 8px; overflow: hidden; }
    .title { padding: 8px 10px; font-size: 14px; border-bottom: 1px solid #333; }
    img { width: 100%; height: auto; display: block; }
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='card'>
      <div class='title'>Left Camera</div>
      <img src='/stream/left' />
    </div>
    <div class='card'>
      <div class='title'>Right Camera</div>
      <img src='/stream/right' />
    </div>
  </div>
</body>
</html>
"""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_stream(self, side: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        try:
            while True:
                frame = self.hub.get_frame(side)
                if frame is None:
                    time.sleep(0.03)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                time.sleep(0.02)
        except (ConnectionResetError, BrokenPipeError):
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MX219 双目网页预览服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--source", choices=["argus", "v4l2"], default="argus")
    parser.add_argument("--cam0", default="0", help="argus为sensor-id，v4l2为设备路径")
    parser.add_argument("--cam1", default="1", help="argus为sensor-id，v4l2为设备路径")
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=1232)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--flip-method", type=int, default=2)
    parser.add_argument("--swap-lr", action="store_true", default=True)
    parser.add_argument("--no-swap-lr", dest="swap_lr", action="store_false")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--no-overlay", action="store_true", help="关闭清晰度叠加")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ServerConfig(
        source=args.source,
        cam0=args.cam0,
        cam1=args.cam1,
        width=args.width,
        height=args.height,
        fps=args.fps,
        flip_method=args.flip_method,
        swap_lr=args.swap_lr,
        jpeg_quality=max(30, min(95, args.jpeg_quality)),
        overlay=not args.no_overlay,
    )

    hub = FrameHub(config)
    try:
        hub.start()
    except Exception as exc:
        print(f"[ERR] 启动相机失败: {exc}")
        print("[HINT] 可尝试降分辨率: --width 1280 --height 720 --fps 30")
        return 1

    StereoHandler.hub = hub
    server = ThreadingHTTPServer((args.host, args.port), StereoHandler)
    print(f"[INFO] Web 预览已启动: http://{args.host}:{args.port}")
    print("[INFO] 在你的电脑浏览器打开该地址即可看左右画面")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        hub.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
VPI 硬件加速双目深度估计 — 适用于 Jetson Orin Nano Super + 双 MX219

核心思路：Python 只做指挥官，VPI 调 GPU/PVA/OFA 当苦力。
  1) nvarguscamerasrc (CSI) → BGR 帧
  2) 校正（使用预先计算的 rectify map）
  3) VPI stereodisp (CUDA 后端) → 硬件加速视差图
  4) 视差 → 深度 / 伪彩色 → 通过 HTTP 推流让 SSH 也能看

依赖：系统自带的 vpi、opencv (cuda)、numpy — 不需要额外 pip install。
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import vpi

# ── 默认参数 ─────────────────────────────────────────────────
DEFAULT_CALIB = str(Path(__file__).resolve().parent.parent / "MX219-DUAL" / "calib_data" / "stereo_calib.npz")
ROTATE_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


# ── 数据结构 ─────────────────────────────────────────────────
@dataclass
class StereoConfig:
    cam0: int
    cam1: int
    width: int
    height: int
    fps: int
    rotate0: int
    rotate1: int
    swap_lr: bool
    calib_path: str
    no_rectify: bool  # 跳过校正（标定有问题时用）
    # VPI 参数
    max_disparity: int
    vpi_backend: str
    downscale: int  # 处理缩放因子 (1=原始, 2=半分辨率)
    # 服务器
    host: str
    port: int
    jpeg_quality: int


@dataclass
class CalibData:
    """从 .npz 加载的标定信息"""
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    Q: np.ndarray
    roi1: np.ndarray
    roi2: np.ndarray
    baseline_mm: float
    focal_px: float


# ── GStreamer 管线 ───────────────────────────────────────────
def _argus_pipeline(sensor_id: int, w: int, h: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def _open_cam(sensor_id: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(_argus_pipeline(sensor_id, w, h, fps), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机 sensor-id={sensor_id}")
    # 等几帧让 ISP 稳定
    for _ in range(15):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.04)
    return cap


def _open_stereo(cam0: int, cam1: int, w: int, h: int, fps: int) -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
    profiles = [(w, h, fps), (1280, 720, min(fps, 30)), (640, 480, min(fps, 30))]
    seen = set()
    for pw, ph, pf in profiles:
        if (pw, ph, pf) in seen:
            continue
        seen.add((pw, ph, pf))
        try:
            print(f"[INFO] 尝试 {pw}x{ph}@{pf} ...")
            c0 = _open_cam(cam0, pw, ph, pf)
            time.sleep(0.2)
            c1 = _open_cam(cam1, pw, ph, pf)
            print(f"[OK]   双目打开成功: {pw}x{ph}@{pf}")
            return c0, c1
        except Exception as e:
            print(f"[WARN] {pw}x{ph}@{pf} 失败: {e}")
    raise RuntimeError("所有分辨率都无法打开双目相机")


# ── 标定数据加载 ─────────────────────────────────────────────
def load_calib(path: str) -> Optional[CalibData]:
    if not Path(path).exists():
        print(f"[CALIB] 标定文件不存在: {path}")
        return None
    d = np.load(path)
    T = d["T"]
    baseline_mm = float(np.linalg.norm(T)) * 1000  # 米→毫米
    P1 = d["P1"]
    focal_px = float(P1[0, 0])
    stereo_err = float(d["reproj_error_stereo"].item()) if "reproj_error_stereo" in d else -1
    print(f"[CALIB] 基线: {baseline_mm:.1f} mm, 焦距: {focal_px:.1f} px")
    print(f"[CALIB] 标定图像: {int(d['image_width'])}x{int(d['image_height'])}")
    print(f"[CALIB] 双目重投影误差: {stereo_err:.2f}")
    if stereo_err > 2.0:
        print(f"[CALIB] ⚠️  误差 {stereo_err:.1f} 远超正常值 (< 1.0)，标定质量极差！")
        print(f"[CALIB] ⚠️  建议重新标定，或使用 --no-rectify 先看原始画面")
    if focal_px > 10000:
        print(f"[CALIB] ⚠️  焦距 {focal_px:.0f} px 异常偏大，标定数据可能有误")
    return CalibData(
        map1x=d["map1x"], map1y=d["map1y"],
        map2x=d["map2x"], map2y=d["map2y"],
        Q=d["Q"], roi1=d["roi1"], roi2=d["roi2"],
        baseline_mm=baseline_mm, focal_px=focal_px,
    )


# ── VPI 加速核心 ─────────────────────────────────────────────
class VPIStereoEngine:
    """封装 VPI stereodisp 的加速引擎（预分配缓冲区，避免显存泄漏）"""

    def __init__(self, max_disparity: int, backend_name: str, height: int, width: int):
        self.max_disparity = max_disparity
        bk = getattr(vpi.Backend, backend_name.upper(), None)
        if bk is None:
            print(f"[WARN] VPI 后端 '{backend_name}' 不存在，回退到 CUDA")
            bk = vpi.Backend.CUDA
        self.backend = bk
        self._stream = vpi.Stream()

        # 预分配 VPI 图像缓冲区（整个生命周期只分配一次）
        self._vpi_left = vpi.Image((width, height), vpi.Format.U16)
        self._vpi_right = vpi.Image((width, height), vpi.Format.U16)
        # numpy 缓冲区也预分配
        self._buf_left = np.empty((height, width), dtype=np.uint16)
        self._buf_right = np.empty((height, width), dtype=np.uint16)

        # 预热一次，让 VPI 分配内部工作缓冲
        with self.backend:
            _ = vpi.stereodisp(
                self._vpi_left, self._vpi_right,
                maxdisp=self.max_disparity, window=5,
                stream=self._stream,
            )
        self._stream.sync()

        import gc; gc.collect()
        print(f"[VPI]  后端: {self.backend}, max_disparity: {max_disparity}, 缓冲: {width}x{height}")

    def compute_disparity(self, gray_left: np.ndarray, gray_right: np.ndarray) -> np.ndarray:
        """输入: uint8 灰度图; 输出: float32 视差图 (像素单位)"""
        # 就地转到预分配的 uint16 缓冲
        np.copyto(self._buf_left, gray_left, casting='unsafe')
        np.copyto(self._buf_right, gray_right, casting='unsafe')

        # 写入已有的 VPI 图像（不重新分配显存）
        self._vpi_left.cpu()[:] = self._buf_left
        self._vpi_right.cpu()[:] = self._buf_right

        with self.backend:
            vpi_disp = vpi.stereodisp(
                self._vpi_left, self._vpi_right,
                maxdisp=self.max_disparity,
                window=5,
                stream=self._stream,
            )
        self._stream.sync()

        # 结果是 S16 (Q10.5 定点)，除以 32 得到浮点像素视差
        disp_s16 = vpi_disp.cpu().astype(np.float32) / 32.0
        disp_s16[disp_s16 < 0] = 0
        return disp_s16


# ── 帧处理管线 ───────────────────────────────────────────────
class StereoPipeline:
    def __init__(self, config: StereoConfig):
        self.cfg = config
        self.calib: Optional[CalibData] = None
        self._do_rectify = not config.no_rectify

        if self._do_rectify:
            self.calib = load_calib(config.calib_path)
            if self.calib is None:
                print("[PIPE] 无标定数据，自动切换为 --no-rectify 模式")
                self._do_rectify = False
        else:
            print("[PIPE] --no-rectify 模式：跳过校正，直接输入原始图像到 VPI")

        # 计算 VPI 处理分辨率（可缩小以节省显存）
        self._proc_h = config.height // config.downscale
        self._proc_w = config.width // config.downscale
        print(f"[PIPE] 处理分辨率: {self._proc_w}x{self._proc_h} (downscale={config.downscale})")
        self.engine = VPIStereoEngine(
            config.max_disparity, config.vpi_backend,
            self._proc_h, self._proc_w,
        )

        self._cap0: Optional[cv2.VideoCapture] = None
        self._cap1: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()

        # 输出 JPEG 缓存 (6路)
        self._jpeg_raw_left: Optional[bytes] = None
        self._jpeg_raw_right: Optional[bytes] = None
        self._jpeg_left: Optional[bytes] = None
        self._jpeg_right: Optional[bytes] = None
        self._jpeg_disp: Optional[bytes] = None
        self._jpeg_anaglyph: Optional[bytes] = None

        # 性能统计
        self._fps = 0.0
        self._latency_ms = 0.0

    def start(self):
        self._cap0, self._cap1 = _open_stereo(
            self.cfg.cam0, self.cfg.cam1,
            self.cfg.width, self.cfg.height, self.cfg.fps,
        )
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[PIPE] 管线就绪，开始推流")

    def stop(self):
        self._running = False
        if self._cap0: self._cap0.release()
        if self._cap1: self._cap1.release()

    def get_frame(self, channel: str) -> Optional[bytes]:
        with self._lock:
            return {
                "raw_left":  self._jpeg_raw_left,
                "raw_right": self._jpeg_raw_right,
                "left":      self._jpeg_left,
                "right":     self._jpeg_right,
                "disparity": self._jpeg_disp,
                "anaglyph":  self._jpeg_anaglyph,
            }.get(channel)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def _loop(self):
        enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
        colormap = cv2.COLORMAP_INFERNO
        t_prev = time.monotonic()
        alpha = 0.9

        while self._running:
            t0 = time.monotonic()

            ok0, frame0 = self._cap0.read()
            ok1, frame1 = self._cap1.read()
            if not ok0 or not ok1:
                time.sleep(0.005)
                continue

            # 旋转
            if self.cfg.rotate0:
                rc = ROTATE_CODE.get(self.cfg.rotate0)
                if rc is not None:
                    frame0 = cv2.rotate(frame0, rc)
            if self.cfg.rotate1:
                rc = ROTATE_CODE.get(self.cfg.rotate1)
                if rc is not None:
                    frame1 = cv2.rotate(frame1, rc)

            # 交换左右
            if self.cfg.swap_lr:
                frame0, frame1 = frame1, frame0

            left_bgr, right_bgr = frame0, frame1

            # 校正 (remap) — 如果标定可用
            if self._do_rectify and self.calib is not None:
                rect_left = cv2.remap(left_bgr, self.calib.map1x, self.calib.map1y, cv2.INTER_LINEAR)
                rect_right = cv2.remap(right_bgr, self.calib.map2x, self.calib.map2y, cv2.INTER_LINEAR)
            else:
                rect_left = left_bgr
                rect_right = right_bgr

            # 灰度 + 缩放（节省显存）
            gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
            if self.cfg.downscale > 1:
                gray_l = cv2.resize(gray_l, (self._proc_w, self._proc_h), interpolation=cv2.INTER_AREA)
                gray_r = cv2.resize(gray_r, (self._proc_w, self._proc_h), interpolation=cv2.INTER_AREA)

            # VPI 加速视差
            disp = self.engine.compute_disparity(gray_l, gray_r)

            # 如果缩放了，把视差图放大回原始尺寸以便可视化
            if self.cfg.downscale > 1:
                disp = cv2.resize(disp, (self.cfg.width, self.cfg.height), interpolation=cv2.INTER_LINEAR)
                disp *= self.cfg.downscale  # 视差值也要按比例还原

            # 伪彩色可视化
            disp_vis = (disp / self.cfg.max_disparity * 255).clip(0, 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_vis, colormap)

            # 红蓝立体图 (anaglyph)
            anaglyph = np.zeros_like(rect_left)
            anaglyph[:, :, 2] = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)   # Red = left
            anaglyph[:, :, 0] = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)  # Blue = right
            anaglyph[:, :, 1] = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)   # Green = left

            # OSD
            t1 = time.monotonic()
            dt = t1 - t_prev
            t_prev = t1
            inst_fps = 1.0 / max(dt, 1e-6)
            self._fps = alpha * self._fps + (1 - alpha) * inst_fps
            self._latency_ms = alpha * self._latency_ms + (1 - alpha) * ((t1 - t0) * 1000)

            rectify_tag = "RECT" if self._do_rectify else "RAW"
            info = f"VPI({self.cfg.vpi_backend.upper()}) {self._fps:.1f}fps {self._latency_ms:.0f}ms d={self.cfg.max_disparity} {rectify_tag}"
            cv2.putText(disp_color, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # 原始画面加标签
            raw_l_vis = left_bgr.copy()
            raw_r_vis = right_bgr.copy()
            cv2.putText(raw_l_vis, "RAW LEFT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(raw_r_vis, "RAW RIGHT", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 编码
            _, j_rl = cv2.imencode(".jpg", raw_l_vis, enc_param)
            _, j_rr = cv2.imencode(".jpg", raw_r_vis, enc_param)
            _, j_l = cv2.imencode(".jpg", rect_left, enc_param)
            _, j_r = cv2.imencode(".jpg", rect_right, enc_param)
            _, j_d = cv2.imencode(".jpg", disp_color, enc_param)
            _, j_a = cv2.imencode(".jpg", anaglyph, enc_param)

            with self._lock:
                self._jpeg_raw_left = j_rl.tobytes()
                self._jpeg_raw_right = j_rr.tobytes()
                self._jpeg_left = j_l.tobytes()
                self._jpeg_right = j_r.tobytes()
                self._jpeg_disp = j_d.tobytes()
                self._jpeg_anaglyph = j_a.tobytes()


# ── HTTP 服务 ────────────────────────────────────────────────
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jetson VPI Stereo Depth</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#eee;font-family:system-ui,sans-serif}
header{padding:14px 20px;background:#111;border-bottom:1px solid #222;
       display:flex;align-items:center;justify-content:space-between}
header h1{font-size:18px;font-weight:600}
header span{font-size:13px;color:#888}
.section{padding:4px 10px 0}
.section h2{font-size:14px;color:#666;margin:6px 0 4px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:0 10px 10px}
.card{background:#151515;border:1px solid #262626;border-radius:8px;overflow:hidden}
.card .label{padding:5px 12px;font-size:12px;color:#aaa;border-bottom:1px solid #222;display:flex;align-items:center;gap:6px}
.card .label .dot{width:7px;height:7px;border-radius:50%;background:#4ade80}
.card img{width:100%;height:auto;display:block}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>&#x1f4f7; Jetson VPI Stereo Depth</h1>
  <span>Orin Nano Super &middot; VPI 3.2 &middot; CUDA</span>
</header>
<div class="section"><h2>&#x1f4f9; RAW (旋转后，未校正)</h2></div>
<div class="grid">
  <div class="card">
    <div class="label"><span class="dot" style="background:#22d3ee"></span>Raw Left</div>
    <img src="/stream/raw_left">
  </div>
  <div class="card">
    <div class="label"><span class="dot" style="background:#22d3ee"></span>Raw Right</div>
    <img src="/stream/raw_right">
  </div>
</div>
<div class="section"><h2>&#x1f9ee; STEREO (校正 + VPI 深度)</h2></div>
<div class="grid">
  <div class="card">
    <div class="label"><span class="dot"></span>Left (rectified)</div>
    <img src="/stream/left">
  </div>
  <div class="card">
    <div class="label"><span class="dot"></span>Right (rectified)</div>
    <img src="/stream/right">
  </div>
  <div class="card">
    <div class="label"><span class="dot" style="background:#f59e0b"></span>Disparity (VPI)</div>
    <img src="/stream/disparity">
  </div>
  <div class="card">
    <div class="label"><span class="dot" style="background:#ef4444"></span>Anaglyph 3D</div>
    <img src="/stream/anaglyph">
  </div>
</div>
</body>
</html>"""


class StreamHandler(BaseHTTPRequestHandler):
    pipeline: StereoPipeline = None  # type: ignore

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._html(INDEX_HTML.encode())
        if self.path.startswith("/stream/"):
            channel = self.path.split("/stream/", 1)[1].rstrip("/")
            if channel in ("raw_left", "raw_right", "left", "right", "disparity", "anaglyph"):
                return self._mjpeg(channel)
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, *_):
        pass

    def _html(self, body: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, channel: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache,private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpg = self.pipeline.get_frame(channel)
                if jpg is None:
                    time.sleep(0.03)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.016)  # ~60 fps cap for bandwidth
        except (ConnectionResetError, BrokenPipeError):
            pass


# ── CLI ─────────────────────────────────────────────────────
def parse_args() -> StereoConfig:
    p = argparse.ArgumentParser(
        description="VPI 硬件加速双目深度估计 — Jetson Orin Nano + MX219",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cam = p.add_argument_group("相机")
    cam.add_argument("--cam0", type=int, default=0, help="左目 sensor-id")
    cam.add_argument("--cam1", type=int, default=1, help="右目 sensor-id")
    cam.add_argument("--width", type=int, default=1280)
    cam.add_argument("--height", type=int, default=720)
    cam.add_argument("--fps", type=int, default=30)
    cam.add_argument("--rotate0", type=int, default=180, help="左目旋转 (0/90/180/270)")
    cam.add_argument("--rotate1", type=int, default=180, help="右目旋转")
    cam.add_argument("--swap-lr", action="store_true", default=True, help="交换左右")
    cam.add_argument("--no-swap-lr", dest="swap_lr", action="store_false")

    cal = p.add_argument_group("标定")
    cal.add_argument("--calib", default=DEFAULT_CALIB, help="stereo_calib.npz 路径")
    cal.add_argument("--no-rectify", action="store_true", default=False,
                     help="跳过校正（标定坏了或只想看原始画面时用）")

    vp = p.add_argument_group("VPI")
    vp.add_argument("--max-disparity", type=int, default=64, help="最大视差 (必须是 16 的倍数)")
    vp.add_argument("--vpi-backend", default="CUDA", choices=["CUDA", "PVA", "OFA", "CPU"],
                     help="VPI 加速后端")
    vp.add_argument("--downscale", type=int, default=2, choices=[1, 2, 4],
                     help="处理缩放因子 (2=半分辨率，省显存)")

    srv = p.add_argument_group("HTTP 服务")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--jpeg-quality", type=int, default=80)

    a = p.parse_args()
    return StereoConfig(
        cam0=a.cam0, cam1=a.cam1, width=a.width, height=a.height, fps=a.fps,
        rotate0=a.rotate0, rotate1=a.rotate1, swap_lr=a.swap_lr,
        calib_path=a.calib, no_rectify=a.no_rectify,
        max_disparity=a.max_disparity, vpi_backend=a.vpi_backend,
        downscale=a.downscale,
        host=a.host, port=a.port, jpeg_quality=max(30, min(95, a.jpeg_quality)),
    )


def main() -> int:
    cfg = parse_args()
    print("=" * 60)
    print("  Jetson VPI Stereo Depth")
    print(f"  相机: sensor {cfg.cam0} / {cfg.cam1}  {cfg.width}x{cfg.height}@{cfg.fps}")
    print(f"  标定: {cfg.calib_path}")
    print(f"  VPI:  {cfg.vpi_backend}  max_disp={cfg.max_disparity}")
    print(f"  HTTP: http://{cfg.host}:{cfg.port}")
    print("=" * 60)

    pipe = StereoPipeline(cfg)
    try:
        pipe.start()
    except Exception as e:
        print(f"[ERR] 启动失败: {e}")
        return 1

    StreamHandler.pipeline = pipe
    server = ThreadingHTTPServer((cfg.host, cfg.port), StreamHandler)

    def _shutdown(sig, frame):
        print("\n[INFO] 正在关闭...")
        pipe.stop()
        server.server_close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"[INFO] 浏览器打开 http://<jetson-ip>:{cfg.port} 查看双目深度")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

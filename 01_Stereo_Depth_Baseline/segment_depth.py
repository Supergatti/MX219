#!/usr/bin/env python3
"""
实例分割 + 双目深度估计  —  多进程 + SharedMemory 架构
======================================================
进程结构 (彻底绕过 GIL):
  主进程   : 解析参数 / 加载标定 / 分配 SharedMemory / 启动子进程
  进程 A   : CameraWorker  — GStreamer 读帧 → 校正 → SharedMemory
  进程 B   : InferWorker   — VPI 视差(CUDA) + YOLO 分割(TensorRT) → 渲染 → SharedMemory
  进程 C   : WebWorker     — JPEG 编码 + HTTP MJPEG 推流

关键:
  - 所有 CUDA 库 (vpi / torch / ultralytics) 仅在 InferWorker 内 import
  - 使用 SharedMemory + Event 做零拷贝帧传递
  - 不再每帧 torch.cuda.empty_cache(), 依赖 TensorRT 显存管理
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import Event, Process, Value
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
import threading
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_mem(tag: str = ""):
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0].rstrip(':') in ('MemTotal', 'MemAvailable', 'SwapFree'):
                    info[parts[0].rstrip(':')] = int(parts[1]) // 1024
        total = info.get('MemTotal', 0)
        avail = info.get('MemAvailable', 0)
        swap  = info.get('SwapFree', 0)
        label = f"[MEM {tag}]" if tag else "[MEM]"
        print(f"{label} 总: {total}MB  可用: {avail}MB  Swap余: {swap}MB", flush=True)
    except Exception:
        pass

def _proc_title(name: str):
    """尝试设置进程标题 (方便 htop 辨识)"""
    try:
        import importlib
        setproctitle = importlib.import_module("setproctitle")
        setproctitle.setproctitle(name)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Config  (纯数据, 跨进程 pickle 安全)
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    cam0: int = 0
    cam1: int = 1
    width: int = 1280
    height: int = 720
    fps: int = 30
    # 标准方式：在 GStreamer 中做 180 度旋转
    flip_method: int = 2
    swap_lr: bool = True
    calib_path: str = str(Path(__file__).resolve().parent / "calib_data" / "stereo_calib.npz")
    no_rectify: bool = False
    max_disparity: int = 64
    vpi_backend: str = "CUDA"
    downscale: int = 2
    model: str = "yolo11n-seg-dyn.engine"
    conf: float = 0.4
    seg_size: int = 256
    max_det: int = 20
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 70
    preview_width: int = 640


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(value)))


def _apply_sharpen(img: np.ndarray, level: int) -> np.ndarray:
    lvl = _clamp_int(level, 0, 3)
    if lvl == 0:
        return img
    sigma = 1.0 + 0.4 * (lvl - 1)
    amount = 0.9 + 0.5 * lvl
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

# ---------------------------------------------------------------------------
# Calibration  (CPU numpy only, 主进程加载)
# ---------------------------------------------------------------------------

@dataclass
class CalibData:
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    Q: np.ndarray
    baseline_m: float
    focal_px: float

def load_calib(path: str) -> Optional[CalibData]:
    p = Path(path)
    if not p.exists():
        print(f"[CALIB] 文件不存在: {path}")
        return None
    d = np.load(str(p))
    T = d["T"]
    baseline_m = float(np.linalg.norm(T))
    focal_px = float(d["P1"][0, 0])
    err = float(d["reproj_error_stereo"].item()) if "reproj_error_stereo" in d else -1
    print(f"[CALIB] 基线: {baseline_m*1000:.1f}mm  焦距: {focal_px:.1f}px  误差: {err:.3f}")
    return CalibData(
        map1x=d["map1x"], map1y=d["map1y"],
        map2x=d["map2x"], map2y=d["map2y"],
        Q=d["Q"], baseline_m=baseline_m, focal_px=focal_px,
    )

# ---------------------------------------------------------------------------
# SharedMemory Frame Buffer
# ---------------------------------------------------------------------------

class ShmFrameBuffer:
    """
    管理一块 SharedMemory, 用于传递固定大小的图像帧.
    支持多帧 (N 张图拼接在一起), 通过 frame_counter 做无锁同步.
    """

    def __init__(self, name: str, shape: Tuple[int, ...], dtype=np.uint8,
                 n_images: int = 1, create: bool = False):
        self.name = name
        self.shape = shape          # 单张图的 shape, e.g. (h, w, 3)
        self.dtype = np.dtype(dtype)
        self.n_images = n_images
        single_size = int(np.prod(shape)) * self.dtype.itemsize
        self.total_size = single_size * n_images
        self._shm: Optional[SharedMemory] = None
        if create:
            self._shm = SharedMemory(name=name, create=True, size=self.total_size)
        else:
            self._shm = SharedMemory(name=name, create=False)

    def write(self, arrays: List[np.ndarray]):
        """将 n_images 个 ndarray 写入 SharedMemory"""
        assert self._shm is not None
        buf = np.ndarray(
            (self.n_images, *self.shape), dtype=self.dtype, buffer=self._shm.buf
        )
        for i, arr in enumerate(arrays):
            np.copyto(buf[i], arr)

    def read(self) -> List[np.ndarray]:
        """从 SharedMemory 读取 n_images 个 ndarray (零拷贝视图, 调用者按需 .copy())"""
        assert self._shm is not None
        buf = np.ndarray(
            (self.n_images, *self.shape), dtype=self.dtype, buffer=self._shm.buf
        )
        return [buf[i] for i in range(self.n_images)]

    def close(self):
        if self._shm:
            self._shm.close()

    def unlink(self):
        if self._shm:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

# ---------------------------------------------------------------------------
# Camera constants
# ---------------------------------------------------------------------------

ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def _argus_pipeline(sensor_id: int, w: int, h: int, fps: int, flip_method: int) -> str:
    fm = int(flip_method)
    if fm not in (0, 1, 2, 3, 4, 5, 6, 7):
        fm = 2
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={fm} ! video/x-raw, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )

# ---------------------------------------------------------------------------
# Depth colormap  (纯函数, 无 CUDA 依赖)
# ---------------------------------------------------------------------------

DEPTH_COLORS = [
    (0.0,  (60, 60, 255)),
    (0.5,  (30, 120, 255)),
    (1.0,  (0, 180, 255)),
    (1.5,  (0, 230, 230)),
    (2.0,  (0, 220, 150)),
    (3.0,  (0, 200, 80)),
    (5.0,  (200, 200, 0)),
    (8.0,  (255, 150, 0)),
    (15.0, (200, 80, 200)),
]

def depth_to_color(depth_m: float) -> Tuple[int, int, int]:
    if depth_m <= DEPTH_COLORS[0][0]:
        return DEPTH_COLORS[0][1]
    if depth_m >= DEPTH_COLORS[-1][0]:
        return DEPTH_COLORS[-1][1]
    for i in range(len(DEPTH_COLORS) - 1):
        d0, c0 = DEPTH_COLORS[i]
        d1, c1 = DEPTH_COLORS[i + 1]
        if d0 <= depth_m <= d1:
            t = (depth_m - d0) / (d1 - d0)
            return (
                int(c0[0] + t * (c1[0] - c0[0])),
                int(c0[1] + t * (c1[1] - c0[1])),
                int(c0[2] + t * (c1[2] - c0[2])),
            )
    return DEPTH_COLORS[-1][1]

# ═══════════════════════════════════════════════════════════════════════════
#  进程 A: CameraWorker
# ═══════════════════════════════════════════════════════════════════════════

def camera_worker(
    cfg: AppConfig,
    calib_maps: Optional[Dict[str, np.ndarray]],
    shm_name: str,
    frame_shape: Tuple[int, int, int],
    frame_counter: Any,
    frame_ready: Any,
    stop_event: Any,
):
    """
    读取双目 GStreamer 流 → 旋转 / swap / 畸变校正 → 写入 SharedMemory.
    不 import 任何 CUDA 库.
    """
    _proc_title("seg-camera")
    print("[CamW] 进程启动", flush=True)
    cap0 = cap1 = None
    shm_buf = None

    try:
        # 打开 SharedMemory (attach, 不创建)
        shm_buf = ShmFrameBuffer(shm_name, frame_shape, n_images=2, create=False)

        map1x = map1y = map2x = map2y = None
        if calib_maps:
            map1x = calib_maps.get("map1x")
            map1y = calib_maps.get("map1y")
            map2x = calib_maps.get("map2x")
            map2y = calib_maps.get("map2y")
        do_rectify = (
            (not cfg.no_rectify)
            and map1x is not None and map1y is not None
            and map2x is not None and map2y is not None
        )
        debug_saved = False
        frame_idx = 0

        profiles = [
            (cfg.width, cfg.height, cfg.fps),
            (1280, 720, min(cfg.fps, 30)),
            (640, 480, min(cfg.fps, 30)),
        ]
        opened = False
        for pw, ph, pf in profiles:
            try:
                cap0 = cv2.VideoCapture(
                    _argus_pipeline(cfg.cam0, pw, ph, pf, cfg.flip_method), cv2.CAP_GSTREAMER
                )
                time.sleep(0.2)
                cap1 = cv2.VideoCapture(
                    _argus_pipeline(cfg.cam1, pw, ph, pf, cfg.flip_method), cv2.CAP_GSTREAMER
                )
                if cap0.isOpened() and cap1.isOpened():
                    opened = True
                    break
                if cap0:
                    cap0.release()
                if cap1:
                    cap1.release()
                cap0 = cap1 = None
            except Exception:
                pass

        if not opened or cap0 is None or cap1 is None:
            raise RuntimeError("无法打开双目相机")

        while not stop_event.is_set():
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            if not ok0 or not ok1:
                continue

            # 标准方式：方向在 GStreamer 完成，左右通过 swap_lr 处理
            if cfg.swap_lr:
                f0, f1 = f1, f0

            # 畸变校正
            if do_rectify and map1x is not None and map1y is not None and map2x is not None and map2y is not None:
                rect_l = cv2.remap(f0, map1x, map1y, cv2.INTER_LINEAR)
                rect_r = cv2.remap(f1, map2x, map2y, cv2.INTER_LINEAR)
            else:
                rect_l, rect_r = f0, f1

            # 保存第一帧校正图 (调试)
            if not debug_saved:
                cv2.imwrite("/tmp/debug_rect_left.jpg", rect_l)
                cv2.imwrite("/tmp/debug_rect_right.jpg", rect_r)
                print("[CamW] 已保存校正图: /tmp/debug_rect_left.jpg, /tmp/debug_rect_right.jpg", flush=True)
                debug_saved = True

            # 确保 shape 匹配 SharedMemory
            if rect_l.shape != tuple(frame_shape):
                rect_l = cv2.resize(rect_l, (frame_shape[1], frame_shape[0]))
            if rect_r.shape != tuple(frame_shape):
                rect_r = cv2.resize(rect_r, (frame_shape[1], frame_shape[0]))

            # 写入 SharedMemory
            shm_buf.write([rect_l, rect_r])
            with frame_counter.get_lock():
                frame_counter.value += 1
            frame_ready.set()

            frame_idx += 1

    except Exception as e:
        print(f"[CamW] 异常: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        print("[CamW] 正在退出...", flush=True)
        if cap0:
            cap0.release()
        if cap1:
            cap1.release()
        if shm_buf:
            shm_buf.close()
        print("[CamW] 已退出", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  进程 B: InferWorker
# ═══════════════════════════════════════════════════════════════════════════

def infer_worker(
    cfg: AppConfig,
    calib_baseline_m: float,
    calib_focal_px: float,
    shm_cam_name: str,
    frame_shape: Tuple[int, int, int],
    frame_counter: Any,
    frame_ready: Any,
    shm_out_name: str,
    out_shape: Tuple[int, int, int],
    out_counter: Any,
    out_ready: Any,
    yolo_size_value: Any,
    sharpen_value: Any,
    stop_event: Any,
):
    """
    VPI 视差(CUDA) + YOLO 分割(TensorRT) → 渲染 → 写入输出 SharedMemory.
    所有 CUDA 库在此进程内 import, 防止 fork 后 CUDA context 崩溃.
    """
    _proc_title("seg-infer")
    print("[InfW] 进程启动, 初始化 CUDA 子系统...", flush=True)

    shm_cam = None
    shm_out = None

    try:
        # ---- 进程内 import CUDA 库 ----
        import vpi as _vpi
        gc.collect()
        _print_mem("InfW VPI导入后")

        # ---- VPI Stereo Engine (在进程内构建) ----
        ds = max(1, cfg.downscale)
        proc_h = frame_shape[0] // ds
        proc_w = frame_shape[1] // ds
        max_disp = cfg.max_disparity
        stereo_window = 7

        bk = getattr(_vpi.Backend, cfg.vpi_backend.upper(), None) or _vpi.Backend.CUDA
        vpi_stream = _vpi.Stream()
        # VPI warmup
        dummy_l = _vpi.asimage(np.zeros((proc_h, proc_w), dtype=np.uint8))
        dummy_r = _vpi.asimage(np.zeros((proc_h, proc_w), dtype=np.uint8))
        with bk:
            _ = _vpi.stereodisp(dummy_l, dummy_r, maxdisp=max_disp, window=stereo_window, stream=vpi_stream)
        vpi_stream.sync()
        del dummy_l, dummy_r
        gc.collect()
        print(f"[InfW] VPI 就绪: backend={bk} max_disp={max_disp} buf={proc_w}x{proc_h}", flush=True)
        _print_mem("InfW VPI就绪")

        # ---- YOLO (TensorRT) ----
        yolo_disabled = str(cfg.model).lower() == "none"
        yolo_model = None
        yolo_is_engine = False
        yolo_device = 0
        yolo_half = False
        yolo_fail_count = 0
        script_dir = Path(__file__).resolve().parent

        def _pick_fallback_model(primary_model: str) -> Optional[str]:
            primary = Path(primary_model)
            cands: List[Path] = []
            if primary.suffix.lower() == ".engine":
                cands.append(primary.with_suffix(".pt"))
                cands.append(script_dir / "yolo11n-seg.pt")
            for p in cands:
                if p.exists():
                    return str(p)
            return None

        if not yolo_disabled:
            from ultralytics import YOLO
            gc.collect()
            try:
                selected_model = cfg.model
                yolo_model = YOLO(selected_model)
                yolo_is_engine = selected_model.lower().endswith(".engine")
                yolo_device = 0 if yolo_is_engine else "cuda"
                yolo_half = yolo_is_engine

                # warmup
                warmup_sz = cfg.seg_size
                dummy = np.zeros((warmup_sz, warmup_sz, 3), dtype=np.uint8)
                yolo_model.predict(dummy, imgsz=warmup_sz, conf=cfg.conf, verbose=False,
                                   device=yolo_device, half=yolo_half, max_det=cfg.max_det)
                del dummy
                gc.collect()
                print(f"[InfW] YOLO 就绪: model={Path(selected_model).name} engine={yolo_is_engine} imgsz={cfg.seg_size} max_det={cfg.max_det}", flush=True)
            except Exception as we:
                print(f"[InfW] YOLO 初始化失败: {we}", flush=True)
                fallback_model = _pick_fallback_model(cfg.model)
                if fallback_model is not None and Path(fallback_model).resolve() != Path(cfg.model).resolve():
                    try:
                        print(f"[InfW] 尝试回退模型: {Path(fallback_model).name}", flush=True)
                        yolo_model = YOLO(fallback_model)
                        yolo_is_engine = fallback_model.lower().endswith(".engine")
                        yolo_device = 0 if yolo_is_engine else "cuda"
                        yolo_half = yolo_is_engine
                        warmup_sz = cfg.seg_size
                        dummy = np.zeros((warmup_sz, warmup_sz, 3), dtype=np.uint8)
                        yolo_model.predict(dummy, imgsz=warmup_sz, conf=cfg.conf, verbose=False,
                                           device=yolo_device, half=yolo_half, max_det=cfg.max_det)
                        del dummy
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        print(f"[InfW] YOLO 回退成功: model={Path(fallback_model).name}", flush=True)
                    except Exception as fe:
                        print(f"[InfW] YOLO 回退失败, 禁用分割: {fe}", flush=True)
                        yolo_model = None
                else:
                    print("[InfW] 无可用回退模型, 禁用分割", flush=True)
                    yolo_model = None
        else:
            print("[InfW] YOLO 已禁用", flush=True)

        _print_mem("InfW 全部就绪")

        # ---- SharedMemory ----
        shm_cam = ShmFrameBuffer(shm_cam_name, frame_shape, n_images=2, create=False)
        shm_out = ShmFrameBuffer(shm_out_name, out_shape, n_images=4, create=False)

        # ---- 推理参数 ----
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        do_rectify = not cfg.no_rectify
        if do_rectify and calib_baseline_m > 0:
            bf = calib_baseline_m * calib_focal_px
        else:
            bf = 0.06 * 800

        out_h, out_w = out_shape[0], out_shape[1]

        last_counter = -1
        diag_count = 0
        debug_frame_saved = False
        fps_alpha = 0.9
        fps_ema = 0.0
        lat_ema = 0.0
        t_prev = time.monotonic()
        perf_count = 0
        last_yolo_size = cfg.seg_size
        last_sharpen = 0

        print("[InfW] 开始推理循环", flush=True)

        while not stop_event.is_set():
            # 等待新帧 (最多 100ms, 避免阻塞检查 stop)
            if not frame_ready.wait(timeout=0.1):
                continue
            frame_ready.clear()

            cur = frame_counter.value
            if cur == last_counter:
                continue
            last_counter = cur

            t0 = time.monotonic()

            # ---- 读取校正帧 ----
            frames = shm_cam.read()
            rect_l = frames[0].copy()
            rect_r = frames[1].copy()

            cur_sharpen = _clamp_int(int(sharpen_value.value), 0, 3)
            cur_yolo_size = _clamp_int(int(yolo_size_value.value), 160, 1280)

            if cur_sharpen != last_sharpen or cur_yolo_size != last_yolo_size:
                print(f"[InfW] 运行时参数更新: sharpen={cur_sharpen} yolo_imgsz={cur_yolo_size}", flush=True)
                last_sharpen = cur_sharpen
                last_yolo_size = cur_yolo_size

            yolo_input = _apply_sharpen(rect_l, cur_sharpen)

            # ---- VPI 视差 ----
            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY)

            if ds > 1:
                gray_l_s = cv2.resize(gray_l, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
                gray_r_s = cv2.resize(gray_r, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            else:
                gray_l_s, gray_r_s = gray_l, gray_r

            vpi_l = _vpi.asimage(gray_l_s.astype(np.uint8))
            vpi_r = _vpi.asimage(gray_r_s.astype(np.uint8))
            with bk:
                vpi_out = _vpi.stereodisp(vpi_l, vpi_r, maxdisp=max_disp, window=stereo_window, stream=vpi_stream)
            vpi_stream.sync()
            disp = vpi_out.cpu().astype(np.float32) / 32.0
            disp[disp < 0] = 0

            # 前5帧诊断
            if diag_count < 5:
                nz = np.count_nonzero(disp > 0.5)
                total_px = disp.size
                print(f"[DIAG] disp: min={disp.min():.1f} max={disp.max():.1f} "
                      f"mean={disp.mean():.2f} nonzero={nz}/{total_px} ({100*nz/total_px:.1f}%)", flush=True)
                diag_count += 1

            # 上采样 + 裁剪异常视差
            if ds > 1:
                disp = cv2.resize(disp, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                disp *= ds
            disp = np.clip(disp, 0, max_disp * max(1, ds))

            # ---- 深度图 ----
            depth_map = np.zeros_like(disp, dtype=np.float32)
            valid = disp > 1.0
            depth_map[valid] = bf / disp[valid]
            depth_map = np.clip(depth_map, 0, 30.0)

            # ---- YOLO 分割 ----
            instances = []
            if yolo_model is not None:
                try:
                    results = yolo_model.predict(
                        yolo_input, imgsz=cur_yolo_size, conf=cfg.conf, verbose=False,
                        half=yolo_half, device=yolo_device, max_det=cfg.max_det,
                    )
                    if results:
                        r = results[0]
                        if getattr(r, "masks", None) is not None and getattr(r, "boxes", None) is not None:
                            masks_obj = cast(Any, r.masks)
                            boxes_obj = cast(Any, r.boxes)
                            masks_data = masks_obj.data
                            if hasattr(masks_data, "cpu"):
                                masks_data = masks_data.cpu().numpy()
                            else:
                                masks_data = np.asarray(masks_data)
                            boxes_data = boxes_obj.data
                            if hasattr(boxes_data, "cpu"):
                                boxes_data = boxes_data.cpu().numpy()
                            else:
                                boxes_data = np.asarray(boxes_data)
                            # 使用归一化坐标防止纵向压缩
                            for i, seg_norm in enumerate(masks_obj.xyn):
                                if len(seg_norm) == 0: continue
                                pts = (seg_norm * np.array([frame_w, frame_h])).astype(np.int32).reshape((-1, 1, 2))
                                
                                mask_full = np.zeros((frame_h, frame_w), dtype=bool)
                                cv2.fillPoly(mask_full.view(np.uint8), [pts], 1)
                                
                                box = boxes_data[i]
                                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                                conf_val = float(box[4])
                                cls_id = int(box[5])
                                
                                instances.append({
                                    "mask": mask_full,
                                    "class": names.get(cls_id, str(cls_id)),
                                    "conf": conf_val,
                                    "box": (x1, y1, x2, y2),
                                    "pts": pts
                                })
                    yolo_fail_count = 0
                except Exception as pe:
                    err = str(pe)
                    err_l = err.lower()
                    is_size_mismatch = (
                        "not equal to max model size" in err_l
                        or ("input size" in err_l and "not equal" in err_l)
                        or "dimension" in err_l and "mismatch" in err_l
                    )
                    if is_size_mismatch:
                        cur_yolo_size = cfg.seg_size
                        last_yolo_size = cfg.seg_size
                        with yolo_size_value.get_lock():
                            yolo_size_value.value = cfg.seg_size
                        print("[InfW] 当前引擎不支持动态尺寸，强制锁死在默认尺寸", flush=True)
                        continue

                    yolo_fail_count += 1
                    if yolo_fail_count <= 3:
                        print(f"[InfW] YOLO 推理失败({yolo_fail_count}/3): {pe}", flush=True)
                    if yolo_fail_count >= 3:
                        print("[InfW] YOLO 连续失败，已自动禁用分割，仅保留深度输出", flush=True)
                        yolo_model = None

            # ---- 渲染叠加 ----
            overlay = yolo_input.copy()
            seg_depth_vis = yolo_input.copy()

            for inst in instances:
                mask_u8 = inst["mask"].astype(np.uint8) * 255
                mask_u8 = cv2.blur(mask_u8, (3, 3))
                mask = mask_u8 > 127
                mask_u8 = mask.astype(np.uint8) * 255
                cls_name = inst["class"]
                box = inst["box"]

                mask_area = int(np.count_nonzero(mask))
                
                # 提取原始视差（调试用）
                disp_in_mask = disp[mask]
                disp_median = float(np.median(disp_in_mask)) if len(disp_in_mask) > 0 else 0.0
                
                # 严格过滤深度：剔除0、inf、nan、以及超出0.2-8.0m的值
                depths_in_mask = depth_map[mask]
                valid_depths = depths_in_mask[
                    (depths_in_mask > 0.2) & 
                    (depths_in_mask < 8.0) & 
                    np.isfinite(depths_in_mask) &
                    (depths_in_mask != 0.0)
                ]
                
                # 防呆保护：有效深度像素数必须 >= mask总像素的5%
                min_valid_pixels = max(1, int(mask_area * 0.05))
                if len(valid_depths) >= min_valid_pixels:
                    median_depth = float(np.median(valid_depths))
                else:
                    median_depth = -1.0  # 标记为无效
                
                # 硬核调试日志
                x1, y1, x2, y2 = box
                print(f"[DEBUG-DEPTH] 类别: {cls_name}, BBox:[{x1},{y1},{x2},{y2}], Mask总像素: {mask_area}, "
                      f"有效深度像素: {len(valid_depths)}, 原始视差中位数: {disp_median:.2f}, "
                      f"最终输出距离: {median_depth:.2f}m", flush=True)
                
                color = depth_to_color(median_depth) if median_depth > 0 else (128, 128, 128)

                if "pts" in inst:
                    cv2.polylines(overlay, [inst["pts"]], True, color, 2)
                
                if median_depth < 0:
                    label = f"{cls_name} 距离无效 D:{disp_median:.0f}"
                elif median_depth > 0.1:
                    label = f"{cls_name} {median_depth:.1f}m D:{disp_median:.0f}"
                else:
                    label = f"{cls_name} ?m D:{disp_median:.0f}"
                label2 = f"[{x1},{y1}]"
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(overlay, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # 左下角显示bbox坐标
                cv2.putText(overlay, label2, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                seg_depth_vis[mask_u8 > 0] = color

            # ---- 视差可视化（调试用，直观显示匹配质量）----
            # 视差范围 0-128，越大越近（红），越小越远（蓝）
            disp_normed = np.clip(disp / 128.0, 0, 1)
            disp_u8 = (disp_normed * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)  # 红=近 蓝=远
            if not np.all(valid):
                invalid_3c = (~valid)[:, :, np.newaxis]
                depth_color = np.where(invalid_3c, np.uint8(20), depth_color)
            # 添加视差图标注
            cv2.putText(depth_color, "Disparity Map (Red=Near Blue=Far)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # ---- 缩放至输出分辨率 ----
            overlay_s = cv2.resize(overlay, (out_w, out_h), interpolation=cv2.INTER_AREA)
            seg_s = cv2.resize(seg_depth_vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
            depth_s = cv2.resize(depth_color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            raw_s = cv2.resize(yolo_input, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if cur_sharpen > 0:
                overlay_s = _apply_sharpen(overlay_s, cur_sharpen)
                raw_s = _apply_sharpen(raw_s, cur_sharpen)

            # ---- 调试帧保存（首次检测到物体时） ----
            if not debug_frame_saved and len(instances) > 0:
                cv2.imwrite("/tmp/debug_overlay.jpg", overlay_s)
                cv2.imwrite("/tmp/debug_disp.jpg", depth_s)
                cv2.imwrite("/tmp/debug_seg.jpg", seg_s)
                print(f"[DEBUG-SAVE] 已保存调试帧: /tmp/debug_overlay.jpg (检测到{len(instances)}个物体)", flush=True)
                debug_frame_saved = True

            # ---- 写入输出 SharedMemory ----
            shm_out.write([overlay_s, seg_s, depth_s, raw_s])
            with out_counter.get_lock():
                out_counter.value += 1
            out_ready.set()

            # ---- FPS 统计 ----
            t1 = time.monotonic()
            dt = t1 - t_prev
            t_prev = t1
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = fps_alpha * fps_ema + (1 - fps_alpha) * inst_fps
            lat_ema = fps_alpha * lat_ema + (1 - fps_alpha) * ((t1 - t0) * 1000)
            perf_count += 1

            if perf_count % 30 == 0:
                print(f"[PERF] FPS: {fps_ema:.1f}  延迟: {lat_ema:.0f}ms", flush=True)

            # 低频 GC (不 empty_cache)
            if perf_count % 500 == 0:
                gc.collect()

    except Exception as e:
        print(f"[InfW] 异常: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        print("[InfW] 正在退出...", flush=True)
        if shm_cam:
            shm_cam.close()
        if shm_out:
            shm_out.close()
        print("[InfW] 已退出", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  进程 C: WebWorker
# ═══════════════════════════════════════════════════════════════════════════

class _JpegCache:
    """线程安全的 JPEG 缓存, 供 HTTP handler 读取"""
    def __init__(self):
        self._lock = threading.Lock()
        self._frames: Dict[str, bytes] = {}
        self.fps = 0.0
        self.latency_ms = 0.0

    def update(self, channel: str, data: bytes):
        with self._lock:
            self._frames[channel] = data

    def get(self, channel: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(channel)


# 全局引用, 供 StreamHandler 使用 (进程内可见)
_jpeg_cache: Optional[_JpegCache] = None
_runtime_ctrl: Optional[Dict[str, Any]] = None


def _ctrl_state() -> Dict[str, int]:
    if _runtime_ctrl is None:
        return {"yolo_size": 256, "sharpen": 0, "jpeg_quality": 70}
    return {
        "yolo_size": int(_runtime_ctrl["yolo_size"].value),
        "sharpen": int(_runtime_ctrl["sharpen"].value),
        "jpeg_quality": int(_runtime_ctrl["jpeg_quality"].value),
    }


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            return self._index()

        if parsed.path == "/api/state":
            return self._json(_ctrl_state())

        if parsed.path == "/api/set":
            return self._set_control(parsed.query)

        ch_map = {"/overlay": "overlay", "/seg_depth": "seg_depth",
                  "/depth": "depth", "/raw": "raw"}
        ch = ch_map.get(parsed.path)
        if ch:
            return self._mjpeg(ch)
        self.send_error(404)

    def _json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _set_control(self, query: str):
        if _runtime_ctrl is None:
            return self._json({"ok": 0, "error": "runtime_ctrl_not_ready"}, HTTPStatus.SERVICE_UNAVAILABLE)

        q = parse_qs(query)
        if "yolo" in q and q["yolo"]:
            y = _clamp_int(int(q["yolo"][0]), 160, 1280)
            with _runtime_ctrl["yolo_size"].get_lock():
                _runtime_ctrl["yolo_size"].value = y

        if "sharp" in q and q["sharp"]:
            s = _clamp_int(int(q["sharp"][0]), 0, 3)
            with _runtime_ctrl["sharpen"].get_lock():
                _runtime_ctrl["sharpen"].value = s

        if "jpeg" in q and q["jpeg"]:
            j = _clamp_int(int(q["jpeg"][0]), 40, 98)
            with _runtime_ctrl["jpeg_quality"].get_lock():
                _runtime_ctrl["jpeg_quality"].value = j

        return self._json({"ok": 1, **_ctrl_state()})

    def _index(self):
        html = """\
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>实例分割 + 双目测距</title>
<style>
body{background:#111;color:#eee;font-family:sans-serif;margin:20px}
h1{color:#4fc3f7}h3{color:#aaa;margin:18px 0 4px}
img{border:1px solid #333;border-radius:6px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:1320px}
.ctrl{margin:12px 0 8px;padding:10px;background:#1a1a1a;border:1px solid #333;border-radius:8px;max-width:1320px}
.ctrl .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:7px 0}
.ctrl button{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}
.ctrl button:hover{background:#363636}
.state{color:#9bd;font-size:14px}
</style></head>
<body>
<h1>&#x1F4F7; 实例分割 + 双目测距 (多进程)</h1>
<div class="ctrl">
  <div class="row"><b>画质调节（可接受降帧）</b></div>
    <div class="row">YOLO 输入分辨率 (TensorRT engine 下固定为训练尺寸):
    <button onclick="setCtl('yolo',256)">256</button>
    <button onclick="setCtl('yolo',384)">384</button>
    <button onclick="setCtl('yolo',512)">512</button>
    <button onclick="setCtl('yolo',640)">640</button>
  </div>
  <div class="row">锐化强度:
    <button onclick="setCtl('sharp',0)">关</button>
    <button onclick="setCtl('sharp',1)">低</button>
    <button onclick="setCtl('sharp',2)">中</button>
    <button onclick="setCtl('sharp',3)">高</button>
  </div>
  <div class="row">JPEG 质量:
    <button onclick="setCtl('jpeg',70)">70</button>
    <button onclick="setCtl('jpeg',80)">80</button>
    <button onclick="setCtl('jpeg',90)">90</button>
    <button onclick="setCtl('jpeg',95)">95</button>
  </div>
  <div class="row state" id="st">状态加载中...</div>
</div>
<div class="grid">
<div><h3>分割 + 距离标签</h3><img src="/overlay" width="640"/></div>
<div><h3>分割区域按距离填色</h3><img src="/seg_depth" width="640"/></div>
<div><h3>深度图</h3><img src="/depth" width="640"/></div>
<div><h3>原图</h3><img src="/raw" width="640"/></div>
</div>
<script>
function renderState(s){
  document.getElementById('st').textContent =
    `YOLO=${s.yolo_size}  |  锐化=${s.sharpen}  |  JPEG=${s.jpeg_quality}`;
}
function refreshState(){fetch('/api/state').then(r=>r.json()).then(renderState).catch(()=>{});}
function setCtl(k,v){fetch(`/api/set?${k}=${v}`).then(r=>r.json()).then(renderState).catch(()=>{});}
setInterval(refreshState, 1000);
refreshState();
</script>
</body></html>"""
        data = html.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _mjpeg(self, channel: str):
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache,private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpg = _jpeg_cache.get(channel) if _jpeg_cache else None
                if jpg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def log_message(self, format, *args):
        pass


def web_worker(
    cfg: AppConfig,
    shm_out_name: str,
    out_shape: Tuple[int, int, int],
    out_counter: Any,
    out_ready: Any,
    yolo_size_value: Any,
    sharpen_value: Any,
    jpeg_quality_value: Any,
    stop_event: Any,
):
    """
    从输出 SharedMemory 读取渲染帧 → JPEG 编码 → HTTP MJPEG 推流.
    不 import 任何 CUDA 库.
    """
    global _jpeg_cache, _runtime_ctrl
    _proc_title("seg-web")
    print("[WebW] 进程启动", flush=True)

    shm_out = None

    try:
        shm_out = ShmFrameBuffer(shm_out_name, out_shape, n_images=4, create=False)
        _jpeg_cache = _JpegCache()
        _runtime_ctrl = {
            "yolo_size": yolo_size_value,
            "sharpen": sharpen_value,
            "jpeg_quality": jpeg_quality_value,
        }

        # 启动 HTTP 服务器 (在线程中, 本进程内无 GIL 竞争问题)
        server = ThreadingHTTPServer((cfg.host, cfg.port), StreamHandler)
        srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
        srv_thread.start()
        print(f"[WebW] HTTP 服务: http://{cfg.host}:{cfg.port}", flush=True)

        channels = ["overlay", "seg_depth", "depth", "raw"]
        last_counter = -1

        while not stop_event.is_set():
            if not out_ready.wait(timeout=0.1):
                continue
            out_ready.clear()

            cur = out_counter.value
            if cur == last_counter:
                continue
            last_counter = cur

            # 读取 4 张渲染图
            panels = shm_out.read()
            cur_q = _clamp_int(int(jpeg_quality_value.value), 40, 98)
            enc = [int(cv2.IMWRITE_JPEG_QUALITY), cur_q]

            # JPEG 编码 (CPU, 此进程专用核)
            for i, ch in enumerate(channels):
                ok, jpg_buf = cv2.imencode(".jpg", panels[i].copy(), enc)
                if ok:
                    _jpeg_cache.update(ch, jpg_buf.tobytes())

    except Exception as e:
        print(f"[WebW] 异常: {e}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        print("[WebW] 正在退出...", flush=True)
        if shm_out:
            shm_out.close()
        print("[WebW] 已退出", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CLI 解析
# ═══════════════════════════════════════════════════════════════════════════

def _find_model(script_dir: Path) -> str:
    dyn_engine = script_dir / "yolo11n-seg-dyn.engine"
    if dyn_engine.exists():
        return str(dyn_engine)
    engine = script_dir / "yolo11n-seg.engine"
    if engine.exists():
        return str(engine)
    pt = script_dir / "yolo11n-seg.pt"
    if pt.exists():
        return str(pt)
    return "yolo11n-seg.pt"

def parse_args() -> AppConfig:
    script_dir = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    cam = p.add_argument_group("相机")
    cam.add_argument("--cam0", type=int, default=0)
    cam.add_argument("--cam1", type=int, default=1)
    cam.add_argument("--width", type=int, default=1280)
    cam.add_argument("--height", type=int, default=720)
    cam.add_argument("--fps", type=int, default=30)
    cam.add_argument("--flip-method", type=int, default=2)
    cam.add_argument("--swap-lr", action="store_true", default=True)
    cam.add_argument("--no-swap-lr", dest="swap_lr", action="store_false")

    cal = p.add_argument_group("标定")
    cal.add_argument("--calib", default=str(script_dir / "calib_data" / "stereo_calib.npz"))
    cal.add_argument("--no-rectify", action="store_true")

    vi = p.add_argument_group("VPI")
    vi.add_argument("--max-disparity", type=int, default=64)
    vi.add_argument("--vpi-backend", default="CUDA", choices=["CUDA", "PVA", "OFA", "CPU"])
    vi.add_argument("--downscale", type=int, default=2)

    seg = p.add_argument_group("分割")
    seg.add_argument("--model", default=_find_model(script_dir))
    seg.add_argument("--conf", type=float, default=0.4)
    seg.add_argument("--seg-size", type=int, default=256)
    seg.add_argument("--max-det", type=int, default=20)

    srv = p.add_argument_group("服务器")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--jpeg-quality", type=int, default=70)
    srv.add_argument("--preview-width", type=int, default=640)

    a = p.parse_args()
    return AppConfig(
        cam0=a.cam0, cam1=a.cam1, width=a.width, height=a.height, fps=a.fps,
        flip_method=a.flip_method, swap_lr=a.swap_lr,
        calib_path=a.calib, no_rectify=a.no_rectify,
        max_disparity=a.max_disparity, vpi_backend=a.vpi_backend, downscale=a.downscale,
        model=a.model, conf=a.conf, seg_size=a.seg_size, max_det=a.max_det,
        host=a.host, port=a.port, jpeg_quality=a.jpeg_quality,
        preview_width=a.preview_width,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main  (主进程: 仅分配资源 + 管理子进程, 不碰 CUDA)
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    cfg = parse_args()

    # ---- 标定 (CPU numpy) ----
    calib: Optional[CalibData] = None
    if not cfg.no_rectify:
        calib = load_calib(cfg.calib_path)
        if calib is not None:
            calib_h, calib_w = calib.map1x.shape[:2]
            cfg.width = calib_w
            cfg.height = calib_h
            print(f"[MAIN] 检测到标定分辨率: {calib_w}x{calib_h}")

    is_engine = cfg.model.endswith(".engine")
    print("=" * 60)
    print("  实例分割 + 双目深度估计  (多进程架构)")
    print(f"  YOLO: {Path(cfg.model).name}  {'(TensorRT FP16)' if is_engine else '(PyTorch)'}")
    print(f"  VPI: {cfg.vpi_backend}  downscale: {cfg.downscale}")
    print(f"  分辨率: {cfg.width}x{cfg.height}@{cfg.fps}fps")
    print(f"  seg-size: {cfg.seg_size}  max-det: {cfg.max_det}")
    print(f"  浏览器: http://<jetson-ip>:{cfg.port}")
    print("=" * 60)
    _print_mem("启动")

    # ---- 计算帧 shape ----
    frame_shape = (cfg.height, cfg.width, 3)
    preview_w = _clamp_int(cfg.preview_width, 320, 1280)
    preview_h = int(preview_w * cfg.height / cfg.width)
    out_shape = (preview_h, preview_w, 3)

    # ---- 分配 SharedMemory ----
    shm_cam_name = "seg_cam_frames"
    shm_out_name = "seg_out_panels"

    # 清理可能残留的旧 SharedMemory
    for sname in [shm_cam_name, shm_out_name]:
        try:
            old = SharedMemory(name=sname, create=False)
            old.close()
            old.unlink()
            print(f"[MAIN] 清理残留 SharedMemory: {sname}")
        except FileNotFoundError:
            pass

    shm_cam = ShmFrameBuffer(shm_cam_name, frame_shape, n_images=2, create=True)
    shm_out = ShmFrameBuffer(shm_out_name, out_shape, n_images=4, create=True)
    print(f"[MAIN] SharedMemory 已分配: cam={shm_cam.total_size//1024}KB  out={shm_out.total_size//1024}KB")

    # ---- 同步原语 ----
    frame_counter = Value(ctypes.c_int64, 0)
    frame_ready = Event()
    out_counter = Value(ctypes.c_int64, 0)
    out_ready = Event()
    yolo_size_value = Value(ctypes.c_int32, _clamp_int(cfg.seg_size, 160, 1280))
    sharpen_value = Value(ctypes.c_int32, 0)
    jpeg_quality_value = Value(ctypes.c_int32, _clamp_int(cfg.jpeg_quality, 40, 98))
    stop_event = Event()

    # ---- 准备传给 CameraWorker 的校正映射 (纯 numpy) ----
    calib_maps = None
    if calib is not None:
        calib_maps = {
            "map1x": calib.map1x, "map1y": calib.map1y,
            "map2x": calib.map2x, "map2y": calib.map2y,
        }

    calib_baseline = calib.baseline_m if calib else 0.06
    calib_focal = calib.focal_px if calib else 800.0

    # ---- 启动子进程 ----
    procs: List[Process] = []

    p_cam = Process(
        target=camera_worker, name="CameraWorker",
        args=(cfg, calib_maps, shm_cam_name, frame_shape,
              frame_counter, frame_ready, stop_event),
    )
    procs.append(p_cam)

    p_inf = Process(
        target=infer_worker, name="InferWorker",
        args=(cfg, calib_baseline, calib_focal,
              shm_cam_name, frame_shape, frame_counter, frame_ready,
              shm_out_name, out_shape, out_counter, out_ready,
              yolo_size_value, sharpen_value, stop_event),
    )
    procs.append(p_inf)

    p_web = Process(
        target=web_worker, name="WebWorker",
        args=(cfg, shm_out_name, out_shape, out_counter, out_ready,
              yolo_size_value, sharpen_value, jpeg_quality_value, stop_event),
    )
    procs.append(p_web)

    for p in procs:
        p.start()
        print(f"[MAIN] 已启动 {p.name} (pid={p.pid})")

    _print_mem("运行中")

    # ---- 信号处理 + 等待 ----
    def _shutdown(sig=None, frame=None):
        print("\n[MAIN] 收到终止信号, 正在关闭...", flush=True)
        stop_event.set()
        # 唤醒可能在等待的进程
        frame_ready.set()
        out_ready.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[MAIN] 强制终止 {p.name}", flush=True)
                p.terminate()
                p.join(timeout=2)
        # 清理 SharedMemory
        shm_cam.close()
        shm_cam.unlink()
        shm_out.close()
        shm_out.unlink()
        print("[MAIN] SharedMemory 已清理", flush=True)
        print("[MAIN] 已退出", flush=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            # 监控子进程健康
            for p in procs:
                if not p.is_alive():
                    print(f"[MAIN] {p.name} 已退出 (exitcode={p.exitcode})", flush=True)
                    _shutdown()
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

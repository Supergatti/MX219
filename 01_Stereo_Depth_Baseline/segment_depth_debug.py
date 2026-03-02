#!/usr/bin/env python3
"""
深度计算调试版本 - 可切换多种算法方案
==========================================
基于 segment_depth.py，增加了深度计算方案切换功能

调试方案：
  0: 原始公式 (depth = bf / disp)
  1: 反转视差 (depth = bf / (max_disp - disp))
  2: 负bf + 绝对值 (depth = abs(-bf / disp))
  3: 反转视差 + 负bf
  4: 直接用视差 (depth = disp / 10) - 纯测试
  5: 平方根修正 (depth = bf / sqrt(disp))
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
    rotate0: int = 0
    rotate1: int = 0
    swap_lr: bool = False
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
    preview_width: int = 480
    force_pt: bool = False  # 🔥 强制使用PyTorch模型，避免TensorRT不兼容


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

def _argus_pipeline(sensor_id: int, w: int, h: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
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

# 🔥 新增：深度计算方案
DEPTH_SCHEMES = {
    0: "原始公式 depth=bf/disp",
    1: "反转视差 depth=bf/(max_disp-disp)",
    2: "负bf+绝对值 depth=abs(-bf/disp)",
    3: "反转视差+负bf depth=abs(-bf/(max_disp-disp))",
    4: "直接视差 depth=disp/10 (测试)",
    5: "平方根修正 depth=bf/sqrt(disp)",
}

def compute_depth(disp: np.ndarray, bf: float, max_disp: float, scheme: int, scale_factor: float = 1.0) -> np.ndarray:
    """
    根据方案编号计算深度图
    
    Args:
        disp: 视差图 (原始VPI输出)
        bf: baseline * focal
        max_disp: 最大视差值
        scheme: 方案编号 (0-5)
        scale_factor: 距离校正系数 (默认1.0)
    
    Returns:
        深度图 (单位: 米)
    """
    depth_map = np.zeros_like(disp, dtype=np.float32)
    
    if scheme == 0:
        # 方案0: 原始公式
        valid = disp > 1.0
        depth_map[valid] = bf / disp[valid]
        
    elif scheme == 1:
        # 方案1: 反转视差
        disp_inv = max_disp - disp
        disp_inv = np.clip(disp_inv, 0, max_disp)
        valid = disp_inv > 1.0
        depth_map[valid] = bf / disp_inv[valid]
        
    elif scheme == 2:
        # 方案2: 负bf + 绝对值
        valid = disp > 1.0
        depth_map[valid] = np.abs(-bf / disp[valid])
        
    elif scheme == 3:
        # 方案3: 反转视差 + 负bf
        disp_inv = max_disp - disp
        disp_inv = np.clip(disp_inv, 0, max_disp)
        valid = disp_inv > 1.0
        depth_map[valid] = np.abs(-bf / disp_inv[valid])
        
    elif scheme == 4:
        # 方案4: 直接用视差（测试用）
        depth_map = disp / 10.0
        
    elif scheme == 5:
        # 方案5: 平方根修正
        valid = disp > 1.0
        depth_map[valid] = bf / np.sqrt(disp[valid])
    
    # 应用距离校正系数
    depth_map = depth_map * scale_factor
    depth_map = np.clip(depth_map, 0, 30.0)
    return depth_map

# ═══════════════════════════════════════════════════════════════════════════
#  进程 A: CameraWorker (与原版相同)
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
    _proc_title("seg-camera-debug")
    print("[CamW] 进程启动 [调试模式]", flush=True)
    cap_l = cap_r = None
    shm_buf = None

    try:
        shm_buf = ShmFrameBuffer(shm_name, frame_shape, n_images=2, create=False)

        # 核心逻辑修正：
        # 1. 用户反馈：遮住物理右侧，rawleft被遮。说明 cam0 是物理右。
        # 2. 摄像头倒装，需要旋转 180 度。
        # 3. 180 度旋转会交换左右。为了让旋转后的左图(f0)对应场景左侧，
        #    我们需要把物理右(cam0)分配给 f0，旋转后它就变成了场景左。
        def get_pipe(sid, w, h, f):
            return (f"nvarguscamerasrc sensor-id={sid} ! video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
                    f"format=(string)NV12, framerate=(fraction){f}/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! "
                    "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1 max-buffers=1")

        profiles = [(cfg.width, cfg.height, cfg.fps), (1280, 720, 30)]
        opened = False
        for pw, ph, pf in profiles:
            cap_l = cv2.VideoCapture(get_pipe(cfg.cam0, pw, ph, pf), cv2.CAP_GSTREAMER)
            cap_r = cv2.VideoCapture(get_pipe(cfg.cam1, pw, ph, pf), cv2.CAP_GSTREAMER)
            if cap_l.isOpened() and cap_r.isOpened():
                opened = True; break
            if cap_l: cap_l.release()
            if cap_r: cap_r.release()

        while not stop_event.is_set():
            ok0, f0 = cap_l.read(); ok1, f1 = cap_r.read()
            if not ok0 or not ok1: continue
            
            # 🔥 彻底修复方向：180 度旋转
            f0 = cv2.rotate(f0, cv2.ROTATE_180); f1 = cv2.rotate(f1, cv2.ROTATE_180)

            if calib_maps:
                f0 = cv2.remap(f0, calib_maps["map1x"], calib_maps["map1y"], cv2.INTER_LINEAR)
                f1 = cv2.remap(f1, calib_maps["map2x"], calib_maps["map2y"], cv2.INTER_LINEAR)

            shm_buf.write([cv2.resize(f0, (frame_shape[1], frame_shape[0])), 
                           cv2.resize(f1, (frame_shape[1], frame_shape[0]))])
            with frame_counter.get_lock(): frame_counter.value += 1
            frame_ready.set()
    except Exception as e: print(f"[CamW] Error: {e}")
    finally:
        if cap_l: cap_l.release()
        if cap_r: cap_r.release()


# ═══════════════════════════════════════════════════════════════════════════
#  进程 B: InferWorker (修改版，支持方案切换)
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
    depth_scheme_value: Any,  # 🔥 新增：深度方案选择
    dist_scale_value: Any,  # 🔥 新增：距离校正系数
    stop_event: Any,
):
    _proc_title("seg-infer-debug")
    print("[InfW] 进程启动 [调试模式], 初始化 CUDA 子系统...", flush=True)

    shm_cam = None
    shm_out = None

    try:
        import vpi as _vpi
        gc.collect()
        _print_mem("InfW VPI导入后")

        ds = max(1, cfg.downscale)
        proc_h = frame_shape[0] // ds
        proc_w = frame_shape[1] // ds
        max_disp = cfg.max_disparity
        stereo_window = 7

        bk = getattr(_vpi.Backend, cfg.vpi_backend.upper(), None) or _vpi.Backend.CUDA
        vpi_stream = _vpi.Stream()
        dummy_l = _vpi.asimage(np.zeros((proc_h, proc_w), dtype=np.uint8))
        dummy_r = _vpi.asimage(np.zeros((proc_h, proc_w), dtype=np.uint8))
        with bk:
            _ = _vpi.stereodisp(dummy_l, dummy_r, maxdisp=max_disp, window=stereo_window, stream=vpi_stream)
        vpi_stream.sync()
        del dummy_l, dummy_r
        gc.collect()
        print(f"[InfW] VPI 就绪: backend={bk} max_disp={max_disp} buf={proc_w}x{proc_h}", flush=True)
        _print_mem("InfW VPI就绪")

        # 强制内存清理，为YOLO腾出空间
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        time.sleep(0.5)  # 给系统时间释放内存

        yolo_disabled = str(cfg.model).lower() == "none"
        yolo_model = None
        yolo_is_engine = False
        yolo_device = 0
        yolo_half = False
        yolo_fail_count = 0
        yolo_fixed_size = None  # 🔥 记录引擎固定尺寸
        yolo_corruption_count = 0  # 🔥 检测到垃圾输出的次数
        yolo_current_model_path = None  # 🔥 当前使用的模型路径
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
                
                # 🔥 强制使用.pt模型，避免TensorRT兼容性问题
                if selected_model.lower().endswith(".engine"):
                    pt_model = str(Path(selected_model).with_suffix(".pt"))
                    if Path(pt_model).exists():
                        print(f"[InfW] ⚠️ 检测到.engine文件，强制使用.pt模型避免兼容性问题", flush=True)
                        selected_model = pt_model
                    else:
                        print(f"[InfW] ⚠️ 找不到.pt文件，将尝试使用.engine（可能不兼容）", flush=True)
                
                print(f"[InfW] 🚀 加载模型: {Path(selected_model).name}", flush=True)
                yolo_model = YOLO(selected_model)
                yolo_is_engine = selected_model.lower().endswith(".engine")
                yolo_device = 0 if yolo_is_engine else "cuda"
                yolo_half = yolo_is_engine
                yolo_current_model_path = selected_model
                
                # 检查是否为动态引擎
                if yolo_is_engine and "-dyn" not in selected_model.lower():
                    yolo_fixed_size = cfg.seg_size  # 固定尺寸引擎
                    print(f"[InfW] ⚠️ 固定尺寸引擎，锁定输入尺寸为 {yolo_fixed_size}px", flush=True)
                else:
                    print(f"[InfW] ✅ 模型支持动态分辨率切换", flush=True)

                print(f"[InfW] ✅ YOLO就绪: {Path(selected_model).name}", flush=True)
                gc.collect()
            except Exception as we:
                print(f"[InfW] YOLO 初始化失败: {we}", flush=True)
                fallback_model = _pick_fallback_model(cfg.model)
                if fallback_model is not None:
                    try:
                        print(f"[InfW] 尝试回退模型: {Path(fallback_model).name}", flush=True)
                        yolo_model = YOLO(fallback_model)
                        yolo_is_engine = fallback_model.lower().endswith(".engine")
                        yolo_device = 0 if yolo_is_engine else "cuda"
                        yolo_half = yolo_is_engine
                        # 跳过warmup，节省内存
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        print(f"[InfW] YOLO 回退成功: model={Path(fallback_model).name} (延迟初始化)", flush=True)
                    except Exception as fe:
                        print(f"[InfW] YOLO 回退失败, 禁用分割: {fe}", flush=True)
                        yolo_model = None
                else:
                    print("[InfW] 无可用回退模型, 禁用分割", flush=True)
                    yolo_model = None
        else:
            print("[InfW] YOLO 已禁用", flush=True)

        _print_mem("InfW 全部就绪")

        shm_cam = ShmFrameBuffer(shm_cam_name, frame_shape, n_images=2, create=False)
        shm_out = ShmFrameBuffer(shm_out_name, out_shape, n_images=4, create=False)

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
        last_scheme = 0
        last_dist_scale = 1.0

        print(f"[InfW] 开始推理循环 [当前方案: {DEPTH_SCHEMES[0]}]", flush=True)

        while not stop_event.is_set():
            if not frame_ready.wait(timeout=0.1):
                continue
            frame_ready.clear()

            cur = frame_counter.value
            if cur == last_counter:
                continue
            last_counter = cur

            t0 = time.monotonic()

            frames = shm_cam.read()
            rect_l = frames[0].copy()
            rect_r = frames[1].copy()

            cur_sharpen = _clamp_int(int(sharpen_value.value), 0, 3)
            cur_yolo_size = _clamp_int(int(yolo_size_value.value), 160, 1280)
            cur_scheme = _clamp_int(int(depth_scheme_value.value), 0, 5)  # 🔥 读取方案
            cur_dist_scale = max(0.1, min(5.0, float(dist_scale_value.value)))  # 🔥 读取距离校正系数
            
            # 🔥 如果是固定尺寸引擎，强制使用固定尺寸
            if yolo_fixed_size is not None:
                cur_yolo_size = yolo_fixed_size

            if cur_scheme != last_scheme:
                print(f"[InfW] 🔥 切换深度方案: [{cur_scheme}] {DEPTH_SCHEMES[cur_scheme]}", flush=True)
                last_scheme = cur_scheme
            
            if cur_dist_scale != last_dist_scale:
                print(f"[InfW] 🔥 距离校正系数: {cur_dist_scale:.2f}", flush=True)
                last_dist_scale = cur_dist_scale

            if cur_sharpen != last_sharpen or cur_yolo_size != last_yolo_size:
                last_sharpen = cur_sharpen
                last_yolo_size = cur_yolo_size

            yolo_input = _apply_sharpen(rect_l, cur_sharpen)

            # VPI 视差
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
                # swap_lr=False 时，rect_l 就是物理左，rect_r 就是物理右，无需交换
                vpi_out = _vpi.stereodisp(vpi_l, vpi_r, maxdisp=max_disp, window=stereo_window, stream=vpi_stream)
            vpi_stream.sync()
            disp = vpi_out.cpu().astype(np.float32) / 32.0
            disp[disp < 0] = 0

            if diag_count < 5:
                nz = np.count_nonzero(disp > 0.5)
                total_px = disp.size
                print(f"[DIAG] disp: min={disp.min():.1f} max={disp.max():.1f} "
                      f"mean={disp.mean():.2f} nonzero={nz}/{total_px} ({100*nz/total_px:.1f}%)", flush=True)
                diag_count += 1

            if ds > 1:
                disp = cv2.resize(disp, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                disp *= ds
            disp = np.clip(disp, 0, max_disp * max(1, ds))

            # 🔥 使用选择的方案计算深度，应用距离校正系数
            max_disp_val = float(max_disp * max(1, ds))
            depth_map = compute_depth(disp, bf, max_disp_val, cur_scheme, cur_dist_scale)
            valid = (depth_map > 0.01) & (depth_map < 30.0)

            # YOLO 分割
            instances = []
            if yolo_model is not None:
                try:
                    # 🔥 对于固定尺寸引擎，先resize输入图像到固定尺寸
                    if yolo_fixed_size is not None:
                        # 保持宽高比resize
                        h, w = yolo_input.shape[:2]
                        scale = yolo_fixed_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        # 确保是32的倍数
                        new_w = (new_w // 32) * 32
                        new_h = (new_h // 32) * 32
                        if new_w > 0 and new_h > 0:
                            yolo_resized = cv2.resize(yolo_input, (new_w, new_h))
                        else:
                            yolo_resized = yolo_input
                        # 记录缩放比例用于坐标还原
                        scale_x = frame_w / new_w
                        scale_y = frame_h / new_h
                    else:
                        yolo_resized = yolo_input
                        scale_x = 1.0
                        scale_y = 1.0
                    
                    results = yolo_model.predict(
                        yolo_resized, imgsz=cur_yolo_size, conf=cfg.conf, verbose=False,
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
                            
                            # 🔥 验证YOLO输出是否包含垃圾数据
                            if len(boxes_data) > 0:
                                # 检查NaN/Inf
                                if np.any(np.isnan(boxes_data)) or np.any(np.isinf(boxes_data)):
                                    yolo_corruption_count += 1
                                    print(f"[InfW] 🚨 检测到NaN/Inf，垃圾输出计数: {yolo_corruption_count}/3", flush=True)
                                    if yolo_corruption_count >= 3:
                                        print(f"[InfW] ❌ YOLO输出持续异常，已禁用分割", flush=True)
                                        yolo_model = None
                                    raise RuntimeError("YOLO输出异常")
                                # 检查坐标是否合理
                                if np.any(np.abs(boxes_data[:, :4]) > max(frame_w, frame_h) * 10):
                                    yolo_corruption_count += 1
                                    print(f"[InfW] 🚨 检测到异常坐标，垃圾输出计数: {yolo_corruption_count}/3", flush=True)
                                    if yolo_corruption_count >= 3:
                                        print(f"[InfW] ❌ YOLO输出持续异常，已禁用分割", flush=True)
                                        yolo_model = None
                                    raise RuntimeError("YOLO输出异常")
                                # 检查置信度范围
                                if np.any(boxes_data[:, 4] < 0) or np.any(boxes_data[:, 4] > 1.5):
                                    yolo_corruption_count += 1
                                    print(f"[InfW] 🚨 检测到异常置信度，垃圾输出计数: {yolo_corruption_count}/3", flush=True)
                                    if yolo_corruption_count >= 3:
                                        print(f"[InfW] ❌ YOLO输出持续异常，已禁用分割", flush=True)
                                        yolo_model = None
                                    raise RuntimeError("YOLO输出异常")
                            
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
                    yolo_corruption_count = max(0, yolo_corruption_count - 1)  # 成功则递减
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
                        print("[InfW] YOLO 连续失败，已自动禁用分割", flush=True)
                        yolo_model = None

            # 渲染叠加
            overlay = yolo_input.copy()
            seg_depth_vis = yolo_input.copy()

            # 形态学腐蚀核（用于剥离边缘飞点）
            _erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

            for inst in instances:
                mask_u8 = inst["mask"].astype(np.uint8) * 255
                mask_u8 = cv2.blur(mask_u8, (3, 3))
                mask = mask_u8 > 127
                mask_u8 = mask.astype(np.uint8) * 255
                cls_name = inst["class"]
                box = inst["box"]

                mask_area = int(np.count_nonzero(mask))

                # --- 原始视差统计（调试用） ---
                disp_in_mask = disp[mask]
                disp_median = float(np.median(disp_in_mask)) if len(disp_in_mask) > 0 else 0.0

                # ============================================================
                # 🔥 改进深度提取：形态学腐蚀 + IQR 统计滤波
                # ============================================================

                # Step 1: 掩码腐蚀 —— 向内收缩，剥离边缘飞点
                eroded_mask_u8 = cv2.erode(mask_u8, _erode_kernel, iterations=1)
                eroded_mask = eroded_mask_u8 > 127
                eroded_area = int(np.count_nonzero(eroded_mask))

                # 如果腐蚀后掩码太小（物体本身很小），回退到原始掩码
                if eroded_area < 10:
                    eroded_mask = mask
                    eroded_area = mask_area

                # Step 2: 初步过滤 —— 提取有效深度，排除死点和异常值
                depths_raw = depth_map[eroded_mask]
                valid_filter = (
                    (depths_raw > 0.2) &
                    (depths_raw < 8.0) &
                    (depths_raw != 0.0) &
                    np.isfinite(depths_raw)
                )
                depths_filtered = depths_raw[valid_filter]
                n_before_iqr = len(depths_filtered)

                # Step 3: IQR 统计滤波 —— 剔除离群噪点
                median_depth = -1.0
                n_after_iqr = 0

                if n_before_iqr >= 10:
                    q1 = float(np.percentile(depths_filtered, 25))
                    q3 = float(np.percentile(depths_filtered, 75))
                    iqr = q3 - q1

                    lower_bound = q1 - 1.0 * iqr
                    upper_bound = q3 + 1.0 * iqr

                    depths_clean = depths_filtered[
                        (depths_filtered >= lower_bound) &
                        (depths_filtered <= upper_bound)
                    ]
                    n_after_iqr = len(depths_clean)

                    # Step 4: 最终判定 —— 有效点数必须达到阈值
                    min_valid_pixels = max(10, int(mask_area * 0.05))
                    if n_after_iqr >= min_valid_pixels:
                        median_depth = float(np.median(depths_clean))
                    else:
                        median_depth = -1.0
                else:
                    # 有效深度像素太少，直接标记无效
                    n_after_iqr = 0

                x1, y1, x2, y2 = box
                print(f"[DEBUG-DEPTH] [方案{cur_scheme}] {cls_name} BBox:[{x1},{y1},{x2},{y2}] "
                      f"Mask:{mask_area}px Eroded:{eroded_area}px "
                      f"有效深度(IQR前):{n_before_iqr} (IQR后):{n_after_iqr} "
                      f"视差中位数:{disp_median:.2f} → 距离:{median_depth:.2f}m", flush=True)
                
                color = depth_to_color(median_depth) if median_depth > 0 else (128, 128, 128)

                if "pts" in inst:
                    cv2.polylines(overlay, [inst["pts"]], True, color, 2)
                    cv2.fillPoly(seg_depth_vis, [inst["pts"]], color)
                
                if median_depth < 0:
                    label = f"{cls_name} 无效 D:{disp_median:.0f}"
                elif median_depth > 0.1:
                    label = f"{cls_name} {median_depth:.1f}m D:{disp_median:.0f}"
                else:
                    label = f"{cls_name} ?m D:{disp_median:.0f}"
                label2 = f"[方案{cur_scheme} x{cur_dist_scale:.2f}]"
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(overlay, (x1, y1 - th_t - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(overlay, label2, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                seg_depth_vis[mask_u8 > 0] = color

            # 视差可视化
            disp_normed = np.clip(disp / 128.0, 0, 1)
            disp_u8 = (disp_normed * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_JET)
            if not np.all(valid):
                invalid_3c = (~valid)[:, :, np.newaxis]
                depth_color = np.where(invalid_3c, np.uint8(20), depth_color)
            cv2.putText(depth_color, f"Scheme {cur_scheme}: {DEPTH_SCHEMES[cur_scheme]} (x{cur_dist_scale:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 缩放至输出分辨率
            overlay_s = cv2.resize(overlay, (out_w, out_h), interpolation=cv2.INTER_AREA)
            seg_s = cv2.resize(seg_depth_vis, (out_w, out_h), interpolation=cv2.INTER_AREA)
            depth_s = cv2.resize(depth_color, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            raw_s = cv2.resize(yolo_input, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if cur_sharpen > 0:
                overlay_s = _apply_sharpen(overlay_s, cur_sharpen)
                raw_s = _apply_sharpen(raw_s, cur_sharpen)

            if not debug_frame_saved and len(instances) > 0:
                cv2.imwrite("/tmp/debug_overlay_scheme.jpg", overlay_s)
                cv2.imwrite("/tmp/debug_disp_scheme.jpg", depth_s)
                cv2.imwrite("/tmp/debug_seg_scheme.jpg", seg_s)
                print(f"[DEBUG-SAVE] 已保存调试帧 (方案{cur_scheme}): /tmp/debug_*_scheme.jpg", flush=True)
                debug_frame_saved = True

            shm_out.write([overlay_s, seg_s, depth_s, raw_s])
            with out_counter.get_lock():
                out_counter.value += 1
            out_ready.set()

            t1 = time.monotonic()
            dt = t1 - t_prev
            t_prev = t1
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_ema = fps_alpha * fps_ema + (1 - fps_alpha) * inst_fps
            lat_ema = fps_alpha * lat_ema + (1 - fps_alpha) * ((t1 - t0) * 1000)
            perf_count += 1

            if perf_count % 30 == 0:
                print(f"[PERF] FPS: {fps_ema:.1f}  延迟: {lat_ema:.0f}ms  方案: {cur_scheme}", flush=True)

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
#  进程 C: WebWorker (修改版，支持方案切换)
# ═══════════════════════════════════════════════════════════════════════════

class _JpegCache:
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


_jpeg_cache: Optional[_JpegCache] = None
_runtime_ctrl: Optional[Dict[str, Any]] = None


def _ctrl_state() -> Dict[str, Any]:
    if _runtime_ctrl is None:
        return {"yolo_size": 256, "sharpen": 0, "jpeg_quality": 70, "depth_scheme": 0, "dist_scale": 1.0}
    return {
        "yolo_size": int(_runtime_ctrl["yolo_size"].value),
        "sharpen": int(_runtime_ctrl["sharpen"].value),
        "jpeg_quality": int(_runtime_ctrl["jpeg_quality"].value),
        "depth_scheme": int(_runtime_ctrl["depth_scheme"].value),
        "dist_scale": float(_runtime_ctrl["dist_scale"].value),  # 🔥 新增
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

        # 🔥 新增：深度方案切换
        if "scheme" in q and q["scheme"]:
            sc = _clamp_int(int(q["scheme"][0]), 0, 5)
            with _runtime_ctrl["depth_scheme"].get_lock():
                _runtime_ctrl["depth_scheme"].value = sc
        
        # 🔥 新增：距离校正系数
        if "scale" in q and q["scale"]:
            scale_val = max(0.1, min(5.0, float(q["scale"][0])))
            with _runtime_ctrl["dist_scale"].get_lock():
                _runtime_ctrl["dist_scale"].value = scale_val

        return self._json({"ok": 1, **_ctrl_state()})

    def _index(self):
        # 🔥 修改：增加方案切换按钮
        schemes_html = ""
        for i, desc in DEPTH_SCHEMES.items():
            schemes_html += f'<button onclick="setCtl(\'scheme\',{i})" style="margin:3px 2px">[{i}] {desc}</button>'
        
        html = f"""\
<!doctype html>
<html lang="zh">
<head><meta charset="utf-8"><title>双目测距调试工具</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;margin:20px}}
h1{{color:#f44;font-size:24px}}h3{{color:#aaa;margin:18px 0 4px}}
img{{border:1px solid #333;border-radius:6px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:1320px}}
.ctrl{{margin:12px 0 8px;padding:10px;background:#1a1a1a;border:1px solid #333;border-radius:8px;max-width:1320px}}
.ctrl .row{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:7px 0}}
.ctrl button{{background:#2a2a2a;color:#eee;border:1px solid #444;border-radius:6px;padding:6px 10px;cursor:pointer}}
.ctrl button:hover{{background:#363636}}
.state{{color:#9bd;font-size:14px}}
.scheme-desc{{color:#ffcc00;font-size:13px;margin:5px 0}}
.slider-container{{display:flex;align-items:center;gap:10px;margin:7px 0}}
.slider-container input[type=range]{{width:200px}}
.slider-value{{color:#0f0;font-weight:bold;min-width:50px}}
</style></head>
<body>
<h1>🔥 双目测距调试工具 - 深度算法方案切换器</h1>
<div class="ctrl">
  <div class="row"><b style="color:#f44">🔥 深度计算方案切换（实时生效）</b></div>
  <div class="row scheme-desc">点击按钮切换算法，观察距离是否正常</div>
  <div class="row" style="display:flex;flex-wrap:wrap;max-width:1200px">
{schemes_html}
  </div>
  <hr style="border-color:#333;margin:10px 0">
  <div class="row"><b style="color:#0ff">🎯 距离校正系数（用于修正测量偏差）</b></div>
  <div class="slider-container">
    <span>距离倍数:</span>
    <input type="range" id="scaleSlider" min="0.1" max="3.0" step="0.05" value="1.0" 
           oninput="updateScale(this.value)">
    <span class="slider-value" id="scaleValue">1.00x</span>
    <button onclick="resetScale()">重置</button>
  </div>
  <div class="row" style="color:#999;font-size:12px">提示: 如果测量距离偏远，减小倍数；如果偏近，增大倍数</div>
  <hr style="border-color:#333;margin:10px 0">
  <div class="row"><b>画质调节</b></div>
    <div class="row">YOLO 输入分辨率:
    <button onclick="setCtl('yolo',256)">256 (快速)</button>
    <button onclick="setCtl('yolo',384)">384 (平衡)</button>
    <button onclick="setCtl('yolo',512)">512 (推荐)</button>
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
  </div>
  <div class="row state" id="st">状态加载中...</div>
</div>
<div class="grid">
<div><h3>分割 + 距离标签</h3><img src="/overlay" width="640"/></div>
<div><h3>分割区域按距离填色</h3><img src="/seg_depth" width="640"/></div>
<div><h3>视差图 (红=近 蓝=远)</h3><img src="/depth" width="640"/></div>
<div><h3>原图</h3><img src="/raw" width="640"/></div>
</div>
<script>
const SCHEME_NAMES = {json.dumps(DEPTH_SCHEMES)};
function renderState(s){{
  const schemeDesc = SCHEME_NAMES[s.depth_scheme] || "未知";
  const distScale = s.dist_scale || 1.0;
  document.getElementById('st').innerHTML =
    `<b style="color:#ffcc00">当前方案: [${{s.depth_scheme}}] ${{schemeDesc}}</b> | ` +
    `<b style="color:#0ff">距离倍数: ${{distScale.toFixed(2)}}x</b><br>` +
    `YOLO=${{s.yolo_size}} | 锐化=${{s.sharpen}} | JPEG=${{s.jpeg_quality}}`;
  // 更新滑块位置
  const slider = document.getElementById('scaleSlider');
  if (slider && Math.abs(parseFloat(slider.value) - distScale) > 0.001) {{
    slider.value = distScale;
    document.getElementById('scaleValue').textContent = distScale.toFixed(2) + 'x';
  }}
}}
function refreshState(){{fetch('/api/state').then(r=>r.json()).then(renderState).catch(()=>{{}});}}
function setCtl(k,v){{fetch(`/api/set?${{k}}=${{v}}`).then(r=>r.json()).then(renderState).catch(()=>{{}});}}
function updateScale(val){{
  const numVal = parseFloat(val);
  document.getElementById('scaleValue').textContent = numVal.toFixed(2) + 'x';
  setCtl('scale', numVal);
}}
function resetScale(){{
  document.getElementById('scaleSlider').value = 1.0;
  updateScale(1.0);
}}
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
    depth_scheme_value: Any,  # 🔥 新增
    dist_scale_value: Any,  # 🔥 新增
    stop_event: Any,
):
    global _jpeg_cache, _runtime_ctrl
    _proc_title("seg-web-debug")
    print("[WebW] 进程启动 [调试模式]", flush=True)

    shm_out = None

    try:
        shm_out = ShmFrameBuffer(shm_out_name, out_shape, n_images=4, create=False)
        _jpeg_cache = _JpegCache()
        _runtime_ctrl = {
            "yolo_size": yolo_size_value,
            "sharpen": sharpen_value,
            "jpeg_quality": jpeg_quality_value,
            "depth_scheme": depth_scheme_value,  # 🔥 新增
            "dist_scale": dist_scale_value,  # 🔥 新增
        }

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

            panels = shm_out.read()
            cur_q = _clamp_int(int(jpeg_quality_value.value), 40, 98)
            enc = [int(cv2.IMWRITE_JPEG_QUALITY), cur_q]

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
    """优先使用.pt模型，避免TensorRT兼容性问题"""
    pt = script_dir / "yolo11n-seg.pt"
    if pt.exists():
        return str(pt)
    # 只有在没有.pt时才考虑.engine
    dyn_engine = script_dir / "yolo11n-seg-dyn.engine"
    if dyn_engine.exists():
        return str(dyn_engine)
    engine = script_dir / "yolo11n-seg.engine"
    if engine.exists():
        return str(engine)
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
    cam.add_argument("--rotate0", type=int, default=0)
    cam.add_argument("--rotate1", type=int, default=0)
    cam.add_argument("--swap-lr", action="store_true", default=False)
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
    seg.add_argument("--force-pt", action="store_true", help="强制使用PyTorch模型（.pt），避免TensorRT兼容性问题")

    srv = p.add_argument_group("服务器")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--jpeg-quality", type=int, default=70)
    srv.add_argument("--preview-width", type=int, default=640)

    a = p.parse_args()
    return AppConfig(
        cam0=a.cam0, cam1=a.cam1, width=a.width, height=a.height, fps=a.fps,
        rotate0=a.rotate0, rotate1=a.rotate1, swap_lr=a.swap_lr,
        calib_path=a.calib, no_rectify=a.no_rectify,
        max_disparity=a.max_disparity, vpi_backend=a.vpi_backend, downscale=a.downscale,
        model=a.model, conf=a.conf, seg_size=a.seg_size, max_det=a.max_det,
        host=a.host, port=a.port, jpeg_quality=a.jpeg_quality,
        preview_width=a.preview_width, force_pt=a.force_pt,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main  (主进程: 仅分配资源 + 管理子进程, 不碰 CUDA)
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    cfg = parse_args()

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
    print("  🔥 深度计算调试工具 - 方案切换器")
    print(f"  YOLO: {Path(cfg.model).name}  {'(TensorRT FP16)' if is_engine else '(PyTorch)'}")
    if cfg.force_pt and is_engine:
        print(f"  ⚠️  强制使用PyTorch模式 (--force-pt)")
    print(f"  VPI: {cfg.vpi_backend}  downscale: {cfg.downscale}")
    print(f"  分辨率: {cfg.width}x{cfg.height}@{cfg.fps}fps")
    print(f"  浏览器: http://<jetson-ip>:{cfg.port}")
    print("=" * 60)
    _print_mem("启动")

    frame_shape = (cfg.height, cfg.width, 3)
    preview_w = _clamp_int(cfg.preview_width, 320, 1280)
    preview_h = int(preview_w * cfg.height / cfg.width)
    out_shape = (preview_h, preview_w, 3)

    shm_cam_name = "seg_cam_frames_debug"
    shm_out_name = "seg_out_panels_debug"

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

    frame_counter = Value(ctypes.c_int64, 0)
    frame_ready = Event()
    out_counter = Value(ctypes.c_int64, 0)
    out_ready = Event()
    yolo_size_value = Value(ctypes.c_int32, _clamp_int(cfg.seg_size, 160, 1280))
    sharpen_value = Value(ctypes.c_int32, 0)
    jpeg_quality_value = Value(ctypes.c_int32, _clamp_int(cfg.jpeg_quality, 40, 98))
    depth_scheme_value = Value(ctypes.c_int32, 0)  # 🔥 新增：默认方案0
    dist_scale_value = Value(ctypes.c_double, 1.0)  # 🔥 新增：默认距离校正系数1.0
    stop_event = Event()

    calib_maps = None
    if calib is not None:
        calib_maps = {
            "map1x": calib.map1x, "map1y": calib.map1y,
            "map2x": calib.map2x, "map2y": calib.map2y,
        }

    calib_baseline = calib.baseline_m if calib else 0.06
    calib_focal = calib.focal_px if calib else 800.0

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
              yolo_size_value, sharpen_value, depth_scheme_value, dist_scale_value, stop_event),  # 🔥 新增参数
    )
    procs.append(p_inf)

    p_web = Process(
        target=web_worker, name="WebWorker",
        args=(cfg, shm_out_name, out_shape, out_counter, out_ready,
              yolo_size_value, sharpen_value, jpeg_quality_value,
              depth_scheme_value, dist_scale_value, stop_event),  # 🔥 新增参数
    )
    procs.append(p_web)

    for p in procs:
        p.start()
        print(f"[MAIN] 已启动 {p.name} (pid={p.pid})")

    _print_mem("运行中")

    def _shutdown(sig=None, frame=None):
        print("\n[MAIN] 收到终止信号, 正在关闭...", flush=True)
        stop_event.set()
        frame_ready.set()
        out_ready.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[MAIN] 强制终止 {p.name}", flush=True)
                p.terminate()
                p.join(timeout=2)
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

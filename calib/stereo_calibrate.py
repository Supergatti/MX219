#!/usr/bin/env python3
"""
双目标定工具 v2 — 适配 SSH 环境（Web 采集 + 自动检测 + 离群值剔除）

标准流程：
  1) 打印棋盘格（或 ChArUco 板）
  2) python3 stereo_calibrate.py capture   → 在浏览器里采集图像对
  3) python3 stereo_calibrate.py calibrate → 标定（含自动质量过滤）
  4) python3 stereo_calibrate.py verify    → 在浏览器里查看校正效果
  5) 或者一步到位: python3 stereo_calibrate.py all

修复旧版所有问题：
  - Web 界面替代 cv2.imshow，SSH 下可用
  - 自动捕获 + 稳定性检测，不用手动按 's'
  - 标准 5 系数畸变模型（不用 RATIONAL_MODEL 防过拟合）
  - Per-pair 重投影误差过滤，自动剔除离群值
  - 可视化验证：校正图 + 极线叠加
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 常量 & 配置
# ═══════════════════════════════════════════════════════════════
ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass
class Config:
    # 相机
    cam0: int = 0
    cam1: int = 1
    width: int = 1280
    height: int = 720
    fps: int = 30
    # 根据用户描述：面对z轴，y轴指向地面(Down)，x轴水平向左(Left)。
    # 默认在 pipeline 中使用 flip-method=2 (180度旋转)
    flip_method: int = 2
    # 默认启用左右交换，确保输出左右与物理左右一致
    swap_lr: bool = True
    # 标定板
    board_type: str = "chessboard"     # "chessboard" 或 "charuco"
    board_cols: int = 8                # 棋盘格列数 / ChArUco squaresX
    board_rows: int = 5                # 棋盘格行数 / ChArUco squaresY
    square_mm: float = 23.8            # 格子边长 (mm) — iPad Pro 11" 实测
    marker_mm: float = 18.0            # ChArUco marker 边长 (mm)，棋盘格模式下无用
    # 采集
    min_pairs: int = 40              # 最少采集对数
    auto_capture: bool = True        # 自动采集（检测到且稳定则自动保存）
    stability_frames: int = 3        # 连续 N 帧检测到才认为稳定（从8降到3）
    capture_interval: float = 0.8    # 两次采集最小间隔 (秒)（从1.5降到0.8）
    # 标定
    max_reproj_error: float = 1.5    # per-pair 最大重投影误差，超过就剔除
    # 输出
    output_dir: str = "calib_data"
    # 服务器
    host: str = "0.0.0.0"
    port: int = 8095



# ═══════════════════════════════════════════════════════════════
# GStreamer / 相机
# ═══════════════════════════════════════════════════════════════
def _argus_pipeline(sensor_id: int, w: int, h: int, fps: int, flip_method: int) -> str:
    # 默认推荐 flip-method=2 (180 度)，可通过参数覆盖
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


def _open_cam(sensor_id: int, w: int, h: int, fps: int, flip_method: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(_argus_pipeline(sensor_id, w, h, fps, flip_method), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机 sensor-id={sensor_id}")
    for _ in range(15):
        ok, _ = cap.read()
        if ok:
            break
        time.sleep(0.04)
    return cap


def _open_stereo(cam0: int, cam1: int, w: int, h: int, fps: int, flip_method: int
                 ) -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
    profiles = [(w, h, fps), (1280, 720, min(fps, 30)), (640, 480, min(fps, 30))]
    seen = set()
    for pw, ph, pf in profiles:
        if (pw, ph, pf) in seen:
            continue
        seen.add((pw, ph, pf))
        try:
            print(f"[CAM] 尝试 {pw}x{ph}@{pf} ...")
            c0 = _open_cam(cam0, pw, ph, pf, flip_method)
            time.sleep(0.2)
            c1 = _open_cam(cam1, pw, ph, pf, flip_method)
            print(f"[CAM] 双目打开成功: {pw}x{ph}@{pf}")
            return c0, c1
        except Exception as e:
            print(f"[CAM] {pw}x{ph}@{pf} 失败: {e}")
    raise RuntimeError("所有分辨率都无法打开双目相机")


def _rotate(frame: np.ndarray, angle: int) -> np.ndarray:
    rc = ROTATE_CODE.get(angle)
    return cv2.rotate(frame, rc) if rc is not None else frame


# ═══════════════════════════════════════════════════════════════
# 标定板检测
# ═══════════════════════════════════════════════════════════════
class BoardDetector:
    """统一的标定板检测器，支持棋盘格和 ChArUco"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.board_type = cfg.board_type

        if self.board_type == "chessboard":
            self.pattern_size = (cfg.board_cols, cfg.board_rows)
            # 生成三维物体点
            self.obj_template = np.zeros(
                (cfg.board_cols * cfg.board_rows, 3), dtype=np.float32
            )
            self.obj_template[:, :2] = (
                np.mgrid[0:cfg.board_cols, 0:cfg.board_rows].T.reshape(-1, 2)
                * (cfg.square_mm / 1000.0)
            )
            self._criteria_sub = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
            )
        else:  # charuco
            self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self._board = cv2.aruco.CharucoBoard(
                (cfg.board_cols, cfg.board_rows),
                cfg.square_mm / 1000.0, cfg.marker_mm / 1000.0, self._dict,
            )
            self._det_params = cv2.aruco.DetectorParameters()
            self._board_corners = np.array(self._board.getChessboardCorners(), dtype=np.float32)

    def detect(self, gray: np.ndarray
               ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
                         Optional[np.ndarray], np.ndarray]:
        """
        返回: (found, img_points, obj_points, corner_ids, vis_frame)
          - img_points: (N, 2) float32
          - obj_points: (N, 3) float32
          - corner_ids: (N,) int — 角点 ID（charuco 用于左右配对）
          - vis_frame: 带检测标记的 BGR 可视化图
        """
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if self.board_type == "chessboard":
            return self._detect_chessboard(gray, vis)
        else:
            return self._detect_charuco(gray, vis)

    def _detect_chessboard(self, gray: np.ndarray, vis: np.ndarray
                           ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
                                      Optional[np.ndarray], np.ndarray]:
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None, flags)
        if not found or corners is None:
            return False, None, None, None, vis

        # 亚像素精化
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria_sub)
        cv2.drawChessboardCorners(vis, self.pattern_size, corners, True)
        n = corners.shape[0]
        ids = np.arange(n, dtype=np.int32)
        return True, corners.reshape(-1, 2).astype(np.float32), self.obj_template.copy(), ids, vis

    def _detect_charuco(self, gray: np.ndarray, vis: np.ndarray
                        ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray],
                                   Optional[np.ndarray], np.ndarray]:
        mc, mids, _ = cv2.aruco.detectMarkers(gray, self._dict, parameters=self._det_params)
        if mids is None or len(mids) == 0:
            return False, None, None, None, vis

        cv2.aruco.drawDetectedMarkers(vis, mc, mids)
        retval, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, self._board)
        if retval is None or retval < 6 or cc is None or cids is None:
            return False, None, None, None, vis

        cv2.aruco.drawDetectedCornersCharuco(vis, cc, cids, (0, 255, 0))

        ids_flat = cids.flatten().astype(np.int32)
        obj_pts = self._board_corners[ids_flat]
        img_pts = cc.reshape(-1, 2).astype(np.float32)
        return True, img_pts, obj_pts, ids_flat, vis


# ═══════════════════════════════════════════════════════════════
# ★ 阶段 1：Web 采集
# ═══════════════════════════════════════════════════════════════
class CaptureSession:
    """在后台采集线程中运行，通过 HTTP 给浏览器推画面"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = BoardDetector(cfg)
        self._cap0: Optional[cv2.VideoCapture] = None
        self._cap1: Optional[cv2.VideoCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._jpeg_preview: Optional[bytes] = None
        self._capture_count = 0
        self._last_capture = 0.0
        self._stable_l = 0
        self._stable_r = 0
        self._force_capture = False            # 手动触发
        self._status_text = "等待开始..."
        self._left_dir = ""
        self._right_dir = ""

    @property
    def count(self) -> int:
        return self._capture_count

    @property
    def status(self) -> str:
        return self._status_text

    def get_preview(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg_preview

    def trigger_capture(self):
        """手动触发一次采集"""
        self._force_capture = True

    def start(self):
        self._left_dir = os.path.join(self.cfg.output_dir, "images", "left")
        self._right_dir = os.path.join(self.cfg.output_dir, "images", "right")
        os.makedirs(self._left_dir, exist_ok=True)
        os.makedirs(self._right_dir, exist_ok=True)

        self._cap0, self._cap1 = _open_stereo(
            self.cfg.cam0, self.cfg.cam1,
            self.cfg.width, self.cfg.height, self.cfg.fps, self.cfg.flip_method,
        )
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._cap0: self._cap0.release()
        if self._cap1: self._cap1.release()

    def _loop(self):
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), 85]

        assert self._cap0 is not None and self._cap1 is not None
        while self._running:
            ok0, frame0 = self._cap0.read()
            ok1, frame1 = self._cap1.read()
            if not ok0 or not ok1:
                time.sleep(0.01)
                continue

            # 方向已在 GStreamer pipeline 中通过 flip-method 完成
            # 左右交换单独处理，避免与方向修正耦合
            if self.cfg.swap_lr:
                frame0, frame1 = frame1, frame0

            gray_l = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            gray_r = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            found_l, pts_l, _, _, vis_l = self.detector.detect(gray_l)
            found_r, pts_r, _, _, vis_r = self.detector.detect(gray_r)

            # 稳定性计数
            self._stable_l = (self._stable_l + 1) if found_l else 0
            self._stable_r = (self._stable_r + 1) if found_r else 0

            both_stable = (
                self._stable_l >= self.cfg.stability_frames
                and self._stable_r >= self.cfg.stability_frames
            )
            now = time.time()
            can_capture = (now - self._last_capture) >= self.cfg.capture_interval

            should_save = False
            if self._force_capture and found_l and found_r:
                should_save = True
                self._force_capture = False
            elif self.cfg.auto_capture and both_stable and can_capture:
                should_save = True

            if should_save:
                name = f"pair_{self._capture_count:03d}.png"
                cv2.imwrite(os.path.join(self._left_dir, name), frame0)
                cv2.imwrite(os.path.join(self._right_dir, name), frame1)
                self._capture_count += 1
                self._last_capture = now
                self._stable_l = 0
                self._stable_r = 0
                print(f"[CAP] 保存 pair_{self._capture_count - 1:03d} "
                      f"(共 {self._capture_count}/{self.cfg.min_pairs})")

            # 状态
            nl = 0 if pts_l is None else len(pts_l)
            nr = 0 if pts_r is None else len(pts_r)
            if self._capture_count >= self.cfg.min_pairs:
                self._status_text = f"✅ 采集完成: {self._capture_count} 对"
            elif found_l and found_r and both_stable:
                self._status_text = f"🟢 稳定检测中，即将自动保存 ({self._capture_count}/{self.cfg.min_pairs})"
            elif found_l and found_r:
                self._status_text = f"🟡 双目检测到，等待稳定... ({self._capture_count}/{self.cfg.min_pairs})"
            else:
                self._status_text = f"🔴 L:{nl} R:{nr} 请将标定板放入视野 ({self._capture_count}/{self.cfg.min_pairs})"

            # 合成左右预览
            h, w = vis_l.shape[:2]
            combined = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
            combined[:, :w] = vis_l
            combined[:, w + 4:] = vis_r
            combined[:, w:w + 4] = (80, 80, 80)

            # OSD
            color = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
            cv2.putText(combined, f"L: {nl}pts", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(combined, f"R: {nr}pts", (w + 14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(combined, f"Saved: {self._capture_count}/{self.cfg.min_pairs}",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 保存瞬间闪绿框
            if should_save:
                cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1),
                              (0, 255, 0), 6)

            _, jpg = cv2.imencode(".jpg", combined, enc)
            with self._lock:
                self._jpeg_preview = jpg.tobytes()


# ═══════════════════════════════════════════════════════════════
# ★ 阶段 2：标定计算
# ═══════════════════════════════════════════════════════════════
def _collect_pairs(output_dir: str) -> List[Tuple[str, str]]:
    left_glob = os.path.join(output_dir, "images", "left", "pair_*.png")
    pairs = []
    for lp in sorted(glob.glob(left_glob)):
        rp = os.path.join(output_dir, "images", "right", os.path.basename(lp))
        if os.path.exists(rp):
            pairs.append((lp, rp))
    return pairs


def _reproj_error_per_pair(
    obj_pts: np.ndarray, img_pts: np.ndarray,
    mtx: np.ndarray, dist: np.ndarray,
    rvec: np.ndarray, tvec: np.ndarray,
) -> float:
    """计算单对图像的平均重投影误差"""
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, mtx, dist)
    return float(np.mean(np.linalg.norm(img_pts.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)))


def run_calibrate(cfg: Config) -> int:
    print("=" * 60)
    print("  双目标定 — 开始计算")
    print("=" * 60)

    detector = BoardDetector(cfg)
    pairs = _collect_pairs(cfg.output_dir)
    print(f"[CAL] 找到图像对: {len(pairs)}")

    if len(pairs) < 10:
        print("[ERR] 图像对不足 10，请先运行 capture 采集更多数据")
        return 1

    image_size: Optional[Tuple[int, int]] = None
    all_obj: List[np.ndarray] = []
    all_img_l: List[np.ndarray] = []
    all_img_r: List[np.ndarray] = []
    pair_names: List[str] = []

    for lp, rp in pairs:
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, pts_l, obj_l, ids_l, _ = detector.detect(gray_l)
        found_r, pts_r, obj_r, ids_r, _ = detector.detect(gray_r)

        if not (found_l and found_r):
            continue
        if pts_l is None or pts_r is None or obj_l is None or obj_r is None:
            continue
        if ids_l is None or ids_r is None:
            continue

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        if cfg.board_type == "chessboard":
            # 棋盘格：两边检测出的点数和顺序一致
            if len(pts_l) != len(pts_r):
                continue
            all_obj.append(obj_l.astype(np.float32))
            all_img_l.append(pts_l.reshape(-1, 1, 2).astype(np.float32))
            all_img_r.append(pts_r.reshape(-1, 1, 2).astype(np.float32))
        else:
            # ChArUco：左右检测到的角点 ID 可能不同，取交集
            common_ids = np.intersect1d(ids_l, ids_r)
            if len(common_ids) < 12:
                continue
            # 按 ID 建索引，提取对应点
            l_idx = {int(v): i for i, v in enumerate(ids_l)}
            r_idx = {int(v): i for i, v in enumerate(ids_r)}
            board_corners = detector._board_corners
            m_obj, m_l, m_r = [], [], []
            for cid in sorted(common_ids):
                cid = int(cid)
                m_obj.append(board_corners[cid])
                m_l.append(pts_l[l_idx[cid]])
                m_r.append(pts_r[r_idx[cid]])
            all_obj.append(np.array(m_obj, dtype=np.float32))
            all_img_l.append(np.array(m_l, dtype=np.float32).reshape(-1, 1, 2))
            all_img_r.append(np.array(m_r, dtype=np.float32).reshape(-1, 1, 2))

        pair_names.append(os.path.basename(lp))

    if len(all_obj) < 10 or image_size is None:
        print(f"[ERR] 有效图像对仅 {len(all_obj)}，不足 10")
        return 1

    print(f"[CAL] 有效图像对: {len(all_obj)} / {len(pairs)}")

    # ── 第 1 步：单目标定 ────────────────────────────────────
    print("[CAL] 第 1 步: 单目标定 (左)...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    flags_mono = 0  # 标准 5 系数模型，不用 RATIONAL_MODEL

    _init_mtx = np.eye(3, dtype=np.float64)
    _init_dist = np.zeros(5, dtype=np.float64)
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        all_obj, all_img_l, image_size, _init_mtx.copy(), _init_dist.copy(),
        flags=flags_mono, criteria=criteria,
    )
    print(f"      左目 RMS: {ret_l:.4f}")

    print("[CAL] 第 1 步: 单目标定 (右)...")
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        all_obj, all_img_r, image_size, _init_mtx.copy(), _init_dist.copy(),
        flags=flags_mono, criteria=criteria,
    )
    print(f"      右目 RMS: {ret_r:.4f}")

    # 合理性检查
    fx_l = mtx_l[0, 0]
    fx_r = mtx_r[0, 0]
    w = image_size[0]
    print(f"[CAL] 焦距: 左={fx_l:.1f}px 右={fx_r:.1f}px (合理范围 ~{w*0.5:.0f}-{w*2:.0f})")
    if fx_l > w * 5 or fx_r > w * 5:
        print("[WARN] ⚠️ 焦距异常偏大，标定数据可能有问题！")

    # ── 第 2 步：离群值剔除 ──────────────────────────────────
    print("[CAL] 第 2 步: 逐对重投影误差检查...")
    per_pair_err_l = []
    per_pair_err_r = []
    for i in range(len(all_obj)):
        e_l = _reproj_error_per_pair(all_obj[i], all_img_l[i], mtx_l, dist_l, rvecs_l[i], tvecs_l[i])
        e_r = _reproj_error_per_pair(all_obj[i], all_img_r[i], mtx_r, dist_r, rvecs_r[i], tvecs_r[i])
        per_pair_err_l.append(e_l)
        per_pair_err_r.append(e_r)

    # 统计剔除
    keep_mask = []
    removed = []
    for i, (el, er) in enumerate(zip(per_pair_err_l, per_pair_err_r)):
        worse = max(el, er)
        if worse > cfg.max_reproj_error:
            keep_mask.append(False)
            removed.append((pair_names[i], el, er))
            print(f"      ❌ {pair_names[i]}: L={el:.3f} R={er:.3f} → 剔除")
        else:
            keep_mask.append(True)
            print(f"      ✅ {pair_names[i]}: L={el:.3f} R={er:.3f}")

    if removed:
        print(f"[CAL] 剔除 {len(removed)} 对离群值")

    filtered_obj = [all_obj[i] for i in range(len(all_obj)) if keep_mask[i]]
    filtered_l = [all_img_l[i] for i in range(len(all_img_l)) if keep_mask[i]]
    filtered_r = [all_img_r[i] for i in range(len(all_img_r)) if keep_mask[i]]

    if len(filtered_obj) < 8:
        print(f"[ERR] 剔除后仅剩 {len(filtered_obj)} 对，不够标定")
        print("[HINT] 请重新采集，确保标定板清晰、角度多样")
        return 1

    # ── 第 3 步：用过滤后数据重新单目标定 ────────────────────
    print(f"[CAL] 第 3 步: 用 {len(filtered_obj)} 对干净数据重新单目标定...")
    ret_l2, mtx_l, dist_l, _, _ = cv2.calibrateCamera(
        filtered_obj, filtered_l, image_size, _init_mtx.copy(), _init_dist.copy(),
        flags=flags_mono, criteria=criteria,
    )
    ret_r2, mtx_r, dist_r, _, _ = cv2.calibrateCamera(
        filtered_obj, filtered_r, image_size, _init_mtx.copy(), _init_dist.copy(),
        flags=flags_mono, criteria=criteria,
    )
    print(f"      重标定 RMS: 左={ret_l2:.4f} 右={ret_r2:.4f}")

    # ── 第 4 步：双目标定 ────────────────────────────────────
    print("[CAL] 第 4 步: 双目标定...")
    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        filtered_obj, filtered_l, filtered_r,
        mtx_l, dist_l, mtx_r, dist_r,
        image_size,
        criteria=criteria_stereo,
        flags=flags_stereo,
    )
    print(f"      双目 RMS: {ret_s:.4f}")

    if ret_s > 2.0:
        print(f"[WARN] ⚠️ 双目重投影误差 {ret_s:.2f} 偏高 (理想 < 0.5)")
        print("[HINT] 建议重新采集: 标定板需清晰、平整、多角度多距离")
    elif ret_s > 0.8:
        print(f"[INFO] 双目误差 {ret_s:.2f}，尚可，但可以更好")
    else:
        print(f"[INFO] 🎉 双目误差 {ret_s:.4f}，标定质量优秀！")

    # ── 第 5 步：立体校正 ────────────────────────────────────
    print("[CAL] 第 5 步: 立体校正...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1
    )

    # 基线
    baseline_m = float(np.linalg.norm(T))
    focal_px = float(P1[0, 0])

    # ── 保存 ─────────────────────────────────────────────────
    calib_file = os.path.join(cfg.output_dir, "stereo_calib.npz")
    np.savez(
        calib_file,
        image_width=image_size[0], image_height=image_size[1],
        camera_matrix_left=mtx_l, dist_coeffs_left=dist_l,
        camera_matrix_right=mtx_r, dist_coeffs_right=dist_r,
        R=R, T=T, E=E, F=F,
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
        roi1=np.array(roi1), roi2=np.array(roi2),
        reproj_error_left=np.array([ret_l2]),
        reproj_error_right=np.array([ret_r2]),
        reproj_error_stereo=np.array([ret_s]),
    )

    summary: Dict[str, Any] = {
        "pairs_total": len(pairs),
        "pairs_detected": len(all_obj),
        "pairs_rejected": len(removed),
        "pairs_used": len(filtered_obj),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": {
            "type": cfg.board_type,
            "cols": cfg.board_cols,
            "rows": cfg.board_rows,
            "square_mm": cfg.square_mm,
        },
        "focal_left_px": float(mtx_l[0, 0]),
        "focal_right_px": float(mtx_r[0, 0]),
        "baseline_m": baseline_m,
        "baseline_mm": baseline_m * 1000,
        "reproj_error_left": float(ret_l2),
        "reproj_error_right": float(ret_r2),
        "reproj_error_stereo": float(ret_s),
        "quality": "excellent" if ret_s < 0.5 else "good" if ret_s < 1.0 else "acceptable" if ret_s < 2.0 else "poor",
        "output": calib_file,
    }
    summary_file = os.path.join(cfg.output_dir, "calibration_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("  标定结果")
    print(f"  图像对: {len(filtered_obj)} (剔除 {len(removed)})")
    print(f"  图像尺寸: {image_size[0]}x{image_size[1]}")
    print(f"  焦距: L={mtx_l[0,0]:.1f}px R={mtx_r[0,0]:.1f}px")
    print(f"  基线: {baseline_m*1000:.1f}mm")
    print(f"  重投影误差: L={ret_l2:.4f} R={ret_r2:.4f} Stereo={ret_s:.4f}")
    print(f"  质量: {summary['quality'].upper()}")
    print(f"  保存: {calib_file}")
    print("=" * 60)
    return 0


# ═══════════════════════════════════════════════════════════════
# ★ 阶段 3：验证 (Web 可视化)
# ═══════════════════════════════════════════════════════════════
def _make_verify_images(cfg: Config) -> List[bytes]:
    """读取已有图像对 + 标定数据，生成带极线的校正图"""
    calib_file = os.path.join(cfg.output_dir, "stereo_calib.npz")
    if not os.path.exists(calib_file):
        print("[ERR] 标定文件不存在，请先 calibrate")
        return []

    d = np.load(calib_file)
    map1x, map1y = d["map1x"], d["map1y"]
    map2x, map2y = d["map2x"], d["map2y"]

    pairs = _collect_pairs(cfg.output_dir)
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    results: List[bytes] = []

    for lp, rp in pairs[:8]:  # 最多展示 8 对
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            continue

        rect_l = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, map2x, map2y, cv2.INTER_LINEAR)

        h, w = rect_l.shape[:2]
        combined = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
        combined[:, :w] = rect_l
        combined[:, w + 4:] = rect_r
        combined[:, w:w + 4] = (60, 60, 60)

        # 每隔 40px 画一条极线
        for y in range(0, h, 40):
            color = (0, 255, 0) if (y // 40) % 2 == 0 else (0, 200, 255)
            cv2.line(combined, (0, y), (combined.shape[1] - 1, y), color, 1)

        cv2.putText(combined, os.path.basename(lp), (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        _, jpg = cv2.imencode(".jpg", combined, enc)
        results.append(jpg.tobytes())

    return results


# ═══════════════════════════════════════════════════════════════
# HTTP 服务
# ═══════════════════════════════════════════════════════════════
CAPTURE_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双目标定 — 采集</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#eee;font-family:system-ui,sans-serif}
header{padding:14px 20px;background:#111;border-bottom:1px solid #333}
header h1{font-size:18px}
.main{padding:16px;text-align:center}
.status{font-size:16px;padding:10px;margin:8px 0;background:#181818;border-radius:8px;min-height:40px}
img{max-width:100%;border-radius:8px;border:1px solid #333}
.btn{display:inline-block;margin:12px 8px;padding:12px 28px;font-size:15px;
     border:none;border-radius:8px;cursor:pointer;font-weight:600;color:#fff}
.btn-cap{background:#2563eb}.btn-cap:hover{background:#1d4ed8}
.btn-done{background:#16a34a}.btn-done:hover{background:#15803d}
.hint{font-size:13px;color:#888;margin-top:8px}
</style>
</head><body>
<header><h1>📐 双目标定 — 图像采集</h1></header>
<div class="main">
  <div class="status" id="status">加载中...</div>
  <img src="/stream" id="preview">
  <div>
    <button class="btn btn-cap" onclick="capture()">📸 手动采集</button>
    <button class="btn btn-done" onclick="finish()">✅ 完成采集</button>
  </div>
  <div class="hint">
    自动模式: 棋盘格检测到并稳定后自动保存<br>
    手动模式: 点击"手动采集"按钮<br>
    标定板在画面里出现绿色角点 = 检测成功 ✅
  </div>
</div>
<script>
function capture(){fetch('/api/capture').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent=d.status})}
function finish(){fetch('/api/finish').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent='采集结束: '+d.count+' 对';
  setTimeout(()=>window.close(),2000)})}
setInterval(()=>{fetch('/api/status').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent=d.status})},500)
</script></body></html>"""


VERIFY_HTML_TPL = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>双目标定 — 验证</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0a0a0a;color:#eee;font-family:system-ui,sans-serif;padding:20px}}
h1{{font-size:18px;margin-bottom:12px}}
.info{{background:#181818;padding:12px;border-radius:8px;margin-bottom:16px;font-size:14px;line-height:1.8}}
.info b{{color:#4ade80}}
img{{max-width:100%;border-radius:8px;border:1px solid #333;margin-bottom:12px;display:block}}
.hint{{font-size:13px;color:#888;margin-top:4px;margin-bottom:12px}}
</style>
</head><body>
<h1>✅ 双目标定 — 校正验证</h1>
<div class="info">
  <b>重投影误差:</b> {stereo_err} &nbsp;&nbsp;
  <b>质量:</b> {quality} &nbsp;&nbsp;
  <b>焦距:</b> L={focal_l}px R={focal_r}px &nbsp;&nbsp;
  <b>基线:</b> {baseline}mm &nbsp;&nbsp;
  <b>使用图像对:</b> {pairs_used}
</div>
<div class="hint">
  绿色/黄色水平线 = 极线。如果标定正确，左右图像中相同物体应在同一条极线上。
</div>
{images}
</body></html>"""


class CalibHandler(BaseHTTPRequestHandler):
    session: Optional[CaptureSession] = None
    verify_images: List[bytes] = []
    verify_html: bytes = b""
    mode: str = "capture"
    _should_stop = False

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            if self.mode == "capture":
                return self._html(CAPTURE_HTML.encode())
            else:
                return self._html(self.verify_html)
        if path == "/stream" and self.session:
            return self._mjpeg()
        if path == "/api/status" and self.session:
            return self._json({"status": self.session.status, "count": self.session.count})
        if path == "/api/capture" and self.session:
            self.session.trigger_capture()
            time.sleep(0.3)
            return self._json({"status": self.session.status, "count": self.session.count})
        if path == "/api/finish":
            CalibHandler._should_stop = True
            cnt = self.session.count if self.session else 0
            return self._json({"status": "done", "count": cnt})
        if path.startswith("/verify/"):
            try:
                idx = int(path.split("/verify/")[1])
                if 0 <= idx < len(self.verify_images):
                    return self._jpeg_single(self.verify_images[idx])
            except (ValueError, IndexError):
                pass
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
        pass

    def _html(self, body: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _jpeg_single(self, data: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _mjpeg(self):
        self.send_response(HTTPStatus.OK)
        self.send_header("Cache-Control", "no-cache,private")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpg = self.session.get_preview() if self.session else None
                if jpg is None:
                    time.sleep(0.03)
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (ConnectionResetError, BrokenPipeError):
            pass


def _run_server(cfg: Config, mode: str) -> ThreadingHTTPServer:
    CalibHandler.mode = mode
    CalibHandler._should_stop = False
    
    class ReusableServer(ThreadingHTTPServer):
        allow_reuse_address = True

    server = ReusableServer((cfg.host, cfg.port), CalibHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
def parse_args() -> Tuple[str, Config]:
    p = argparse.ArgumentParser(
        description="双目标定工具 v2 — Web 采集 + 自动检测 + 离群值剔除",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("mode", choices=["capture", "calibrate", "verify", "all"],
                   help="capture=采集 | calibrate=标定 | verify=查看校正效果 | all=全套")

    cam = p.add_argument_group("相机")
    cam.add_argument("--cam0", type=int, default=0)
    cam.add_argument("--cam1", type=int, default=1)
    cam.add_argument("--width", type=int, default=1280)
    cam.add_argument("--height", type=int, default=720)
    cam.add_argument("--fps", type=int, default=30)
    cam.add_argument("--flip-method", type=int, default=2, help="GStreamer nvvidconv flip-method，推荐 2（180度）")
    cam.add_argument("--swap-lr", action="store_true", default=True)
    cam.add_argument("--no-swap-lr", dest="swap_lr", action="store_false")

    board = p.add_argument_group("标定板")
    board.add_argument("--board-type", choices=["chessboard", "charuco"], default="chessboard",
                       help="chessboard=普通棋盘格；charuco=ChArUco DICT_4X4")
    board.add_argument("--board-cols", type=int, default=8,
                       help="棋盘格列数 / ChArUco squaresX")
    board.add_argument("--board-rows", type=int, default=5,
                       help="棋盘格行数 / ChArUco squaresY")
    board.add_argument("--square-mm", type=float, default=23.8,
                       help="格子边长 (mm)")
    board.add_argument("--marker-mm", type=float, default=18.0,
                       help="ChArUco marker 边长 (mm)，棋盘格模式无用")

    capt = p.add_argument_group("采集")
    capt.add_argument("--min-pairs", type=int, default=40)
    capt.add_argument("--no-auto", dest="auto_capture", action="store_false", default=True,
                      help="关闭自动采集，仅手动")
    capt.add_argument("--stability-frames", type=int, default=3)
    capt.add_argument("--capture-interval", type=float, default=0.8)

    cal = p.add_argument_group("标定")
    cal.add_argument("--max-reproj-error", type=float, default=1.5,
                     help="Per-pair 最大重投影误差阈值")

    out = p.add_argument_group("输出")
    out.add_argument("--output-dir", default="calib_data")
    out.add_argument("--host", default="0.0.0.0")
    out.add_argument("--port", type=int, default=8095)

    a = p.parse_args()
    cfg = Config(
        cam0=a.cam0, cam1=a.cam1, width=a.width, height=a.height, fps=a.fps,
        flip_method=a.flip_method, swap_lr=a.swap_lr,
        board_type=a.board_type, board_cols=a.board_cols, board_rows=a.board_rows,
        square_mm=a.square_mm, marker_mm=a.marker_mm,
        min_pairs=a.min_pairs, auto_capture=a.auto_capture,
        stability_frames=a.stability_frames, capture_interval=a.capture_interval,
        max_reproj_error=a.max_reproj_error,
        output_dir=a.output_dir, host=a.host, port=a.port,
    )
    return a.mode, cfg


def main() -> int:
    mode, cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── CAPTURE ──────────────────────────────────────────────
    if mode in ("capture", "all"):
        print("=" * 60)
        print("  双目标定 — 采集模式")
        print(f"  标定板: {cfg.board_type} {cfg.board_cols}x{cfg.board_rows}")
        print(f"  格子: {cfg.square_mm:.1f}mm")
        print(f"  目标: {cfg.min_pairs} 对  自动采集: {cfg.auto_capture}")
        print(f"  浏览器打开: http://<jetson-ip>:{cfg.port}")
        print("=" * 60)

        session = CaptureSession(cfg)
        try:
            session.start()
        except Exception as e:
            print(f"[ERR] 相机启动失败: {e}")
            return 1

        CalibHandler.session = session
        server = _run_server(cfg, "capture")

        print(f"[INFO] 采集服务已启动: http://{cfg.host}:{cfg.port}")
        print("[INFO] 在浏览器中操作，或按 Ctrl+C 停止")

        try:
            while not CalibHandler._should_stop:
                time.sleep(0.5)
                if session.count >= cfg.min_pairs:
                    print(f"[INFO] 已达目标 {cfg.min_pairs} 对，自动结束采集")
                    break
        except KeyboardInterrupt:
            print("\n[INFO] 手动停止采集")

        session.stop()
        server.shutdown()
        server.server_close()
        print(f"[INFO] 采集完成: {session.count} 对图像")

        if session.count < 10:
            print("[WARN] 图像对不足 10，标定质量可能很差")
            if mode == "all":
                print("[INFO] 仍然继续标定…")

    # ── CALIBRATE ────────────────────────────────────────────
    if mode in ("calibrate", "all"):
        ret = run_calibrate(cfg)
        if ret != 0:
            return ret

    # ── VERIFY ───────────────────────────────────────────────
    if mode in ("verify", "all"):
        print("[VER] 生成校正验证图...")
        verify_imgs = _make_verify_images(cfg)
        if not verify_imgs:
            print("[ERR] 无法生成验证图，请确认标定文件存在")
            return 1

        # 读取 summary
        summary_file = os.path.join(cfg.output_dir, "calibration_summary.json")
        info: Dict[str, Any] = {}
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                info = json.load(f)

        imgs_html = "\n".join(f'<img src="/verify/{i}">' for i in range(len(verify_imgs)))
        html = VERIFY_HTML_TPL.format(
            stereo_err=f"{info.get('reproj_error_stereo', '?'):.4f}" if isinstance(info.get('reproj_error_stereo'), float) else str(info.get('reproj_error_stereo', '?')),
            quality=info.get("quality", "?").upper(),
            focal_l=f"{info.get('focal_left_px', 0):.1f}",
            focal_r=f"{info.get('focal_right_px', 0):.1f}",
            baseline=f"{info.get('baseline_mm', 0):.1f}",
            pairs_used=info.get("pairs_used", "?"),
            images=imgs_html,
        )

        CalibHandler.verify_images = verify_imgs
        CalibHandler.verify_html = html.encode()
        CalibHandler.session = None  # 不需要相机
        server = _run_server(cfg, "verify")

        print(f"[VER] 验证页面: http://<jetson-ip>:{cfg.port}")
        print("[VER] 绿线 = 极线，左右对应物体应在同一水平线 → 标定正确")
        print("[VER] Ctrl+C 退出")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        server.shutdown()
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

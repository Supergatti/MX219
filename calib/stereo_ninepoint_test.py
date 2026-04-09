#!/usr/bin/env python3
"""
双目棋盘坐标测试工具（WebUI）

功能：
1) 双目相机分别检测棋盘格角点，并标注识别到的全部方格坐标点。
2) 坐标系：以左下角为原点定义棋盘坐标，指定 T0=(t0_x,t0_y)，记录相对 T0 且右上为正方向的坐标。
3) 像素坐标按图像左上角为原点记录。
4) 采集按“组”进行，每组手动触发开始。
5) 数据保存到 calib_data/<会话开始时间>/group_XXX.json（每组一个文件）。
6) 提供 WebUI 监控画面、显示状态和控制按钮。

运行示例：
        python3 stereo_ninepoint_test.py
浏览器访问：
  http://<jetson-ip>:8096
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
@dataclass
class Config:
    cam0: int = 0
    cam1: int = 1
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_method: int = 2
    swap_lr: bool = True

    board_cols: int = 6
    board_rows: int = 9
    t0_x: int = 3
    t0_y: int = 4

    target_groups: int = 10
    output_dir: str = "calib_data"

    preview_scale: float = 0.65
    jpeg_quality: int = 72
    detect_every_n_frames: int = 2
    annotate_all_points: bool = False
    mjpeg_max_fps: int = 12
    status_poll_ms: int = 1000

    host: str = "0.0.0.0"
    port: int = 8096


def _make_placeholder_jpeg(text: str, width: int = 960, height: int = 540) -> bytes:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 20, 20)
    cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 220, 255), 2)
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if ok:
        return jpg.tobytes()
    return b""


# ═══════════════════════════════════════════════════════════════
# 相机
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
# 棋盘检测 + 九点提取
# ═══════════════════════════════════════════════════════════════
class ChessboardPointDetector:
    def __init__(self, board_cols: int, board_rows: int, t0_x: int, t0_y: int, annotate_all_points: bool):
        if board_cols < 2 or board_rows < 2:
            raise ValueError("board-cols 与 board-rows 必须都 >= 2")
        self.board_cols = board_cols
        self.board_rows = board_rows
        self.t0_x = t0_x
        self.t0_y = t0_y
        self.annotate_all_points = annotate_all_points
        self.pattern_size = (board_cols, board_rows)
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def _to_board_relative(self, row_top_to_bottom: int, col_left_to_right: int, rows: int) -> Tuple[int, int]:
        x_left_bottom = col_left_to_right
        y_left_bottom = (rows - 1 - row_top_to_bottom)
        return (x_left_bottom - self.t0_x, y_left_bottom - self.t0_y)

    def detect(self, gray: np.ndarray) -> Tuple[bool, Optional[np.ndarray], np.ndarray, int, int, int]:
        """
        返回: (found, points(Nx2), vis_bgr, total_corner_count, cols, rows)
        """
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        flags_fast = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        flags_strict = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        flags_sb = 0
        if hasattr(cv2, "CALIB_CB_EXHAUSTIVE"):
            flags_sb |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE"))

        corners: Optional[np.ndarray] = None
        used_cols, used_rows = self.board_cols, self.board_rows
        ok, corners = cv2.findChessboardCorners(gray, self.pattern_size, None, flags_fast)
        if (not ok or corners is None) and hasattr(cv2, "findChessboardCornersSB"):
            try:
                ok, corners = cv2.findChessboardCornersSB(gray, self.pattern_size, flags=flags_sb)
            except cv2.error:
                ok, corners = False, None
        if (not ok or corners is None):
            ok, corners = cv2.findChessboardCorners(gray, self.pattern_size, None, flags_strict)

        if not ok or corners is None:
            return False, None, vis, 0, 0, 0

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        cv2.drawChessboardCorners(vis, (used_cols, used_rows), corners, True)

        all_pts = corners.reshape(used_rows, used_cols, 2)
        flat_pts = corners.reshape(-1, 2).astype(np.float32)

        for r in range(used_rows):
            for c in range(used_cols):
                x, y = all_pts[r, c]
                xr, yr = self._to_board_relative(r, c, used_rows)
                is_t0 = (xr == 0 and yr == 0)
                color = (0, 255, 255) if not is_t0 else (0, 255, 0)
                radius = 4 if not is_t0 else 6
                cv2.circle(vis, (int(x), int(y)), radius, color, -1)
                if self.annotate_all_points or is_t0:
                    label = f"({xr},{yr})"
                    cv2.putText(vis, label, (int(x) + 4, int(y) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35 if self.annotate_all_points else 0.5, color, 1)
        cv2.putText(vis, f"Pattern: {used_cols}x{used_rows}", (8, vis.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return True, flat_pts, vis, int(corners.shape[0]), used_cols, used_rows


# ═══════════════════════════════════════════════════════════════
# 采集会话
# ═══════════════════════════════════════════════════════════════
class NinePointSession:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detector = ChessboardPointDetector(
            cfg.board_cols,
            cfg.board_rows,
            cfg.t0_x,
            cfg.t0_y,
            cfg.annotate_all_points,
        )
        self._cap0: Optional[cv2.VideoCapture] = None
        self._cap1: Optional[cv2.VideoCapture] = None

        self._running = False
        self._lock = threading.Lock()

        self._jpeg_preview: Optional[bytes] = _make_placeholder_jpeg("Waiting for camera frames...")
        self._status_text = "等待相机启动..."

        self._session_dir: Optional[Path] = None
        self._session_name: Optional[str] = None
        self._saved_groups = 0
        self._target_groups = cfg.target_groups
        self._pending_triggers = 0
        self._last_saved_file = ""
        self._last_pattern = "-"
        self._last_left_ok = False
        self._last_right_ok = False
        self._last_left_corners = 0
        self._last_right_corners = 0
        self._last_capture_time = "-"
        self._fps = 0.0
        self._frame_counter = 0
        self._fps_last_ts = time.time()
        self._loop_index = 0

    @property
    def stream_interval(self) -> float:
        fps = max(1, int(self.cfg.mjpeg_max_fps))
        return 1.0 / float(fps)

    def start(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._cap0, self._cap1 = _open_stereo(
            self.cfg.cam0, self.cfg.cam1,
            self.cfg.width, self.cfg.height, self.cfg.fps, self.cfg.flip_method,
        )
        self._jpeg_preview = _make_placeholder_jpeg("Camera started. Detecting chessboard...")
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._running = False
        if self._cap0:
            self._cap0.release()
        if self._cap1:
            self._cap1.release()

    def _ensure_session_dir(self):
        if self._session_dir is not None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(self.cfg.output_dir) / f"ninepoint_{ts}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self._session_dir = session_dir
        self._session_name = session_dir.name

    def new_session(self) -> Dict[str, object]:
        with self._lock:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_dir = Path(self.cfg.output_dir) / f"ninepoint_{ts}"
            session_dir.mkdir(parents=True, exist_ok=True)
            self._session_dir = session_dir
            self._session_name = session_dir.name
            self._saved_groups = 0
            self._pending_triggers = 0
            self._last_saved_file = ""
            self._status_text = f"新会话已创建: {self._session_name}"
            return self.get_status()

    def trigger_group(self) -> Dict[str, object]:
        with self._lock:
            self._ensure_session_dir()
            self._pending_triggers += 1
            self._status_text = (
                f"已触发第 {self._saved_groups + 1} 组，等待双目检测到棋盘..."
            )
            return self.get_status()

    def set_target_groups(self, count: int) -> Dict[str, object]:
        with self._lock:
            self._target_groups = max(1, int(count))
            return self.get_status()

    def get_preview(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg_preview

    def get_status(self) -> Dict[str, object]:
        return {
            "status": self._status_text,
            "saved_groups": self._saved_groups,
            "target_groups": self._target_groups,
            "pending_triggers": self._pending_triggers,
            "session": self._session_name,
            "session_dir": str(self._session_dir) if self._session_dir else "",
            "last_saved_file": self._last_saved_file,
            "last_pattern": self._last_pattern,
            "expected_pattern": f"{self.cfg.board_cols}x{self.cfg.board_rows}",
            "left_ok": self._last_left_ok,
            "right_ok": self._last_right_ok,
            "left_corners": self._last_left_corners,
            "right_corners": self._last_right_corners,
            "fps": round(self._fps, 2),
            "last_capture_time": self._last_capture_time,
        }

    def _build_points_payload(self, points: np.ndarray, cols: int, rows: int) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        points_2d = points.reshape(rows, cols, 2)
        for r in range(rows):
            for c in range(cols):
                u, v = points_2d[r, c]
                x_left_bottom = c
                y_left_bottom = rows - 1 - r
                x_rel = x_left_bottom - self.cfg.t0_x
                y_rel = y_left_bottom - self.cfg.t0_y
                payload.append({
                    "id": int(r * cols + c),
                    "board_index_from_top_left": {"col": int(c), "row": int(r)},
                    "board_coord_from_left_bottom": {"x": int(x_left_bottom), "y": int(y_left_bottom)},
                    "board_coord_from_T0": {"x": int(x_rel), "y": int(y_rel)},
                    "pixel_xy": {"x": float(u), "y": float(v)},
                    "pixel_origin": "top_left",
                })
        return payload

    def _save_group(
        self,
        left_pts: np.ndarray,
        right_pts: np.ndarray,
        left_corner_count: int,
        right_corner_count: int,
        cols: int,
        rows: int,
        left_image: np.ndarray,
        right_image: np.ndarray,
    ) -> str:
        assert self._session_dir is not None

        group_idx = self._saved_groups + 1
        now_iso = datetime.now().isoformat(timespec="milliseconds")
        file_name = f"group_{group_idx:03d}.json"
        left_image_name = f"group_{group_idx:03d}_left.jpg"
        right_image_name = f"group_{group_idx:03d}_right.jpg"
        out_path = self._session_dir / file_name
        left_image_path = self._session_dir / left_image_name
        right_image_path = self._session_dir / right_image_name

        cv2.imwrite(str(left_image_path), left_image)
        cv2.imwrite(str(right_image_path), right_image)

        payload = {
            "group_index": group_idx,
            "captured_at": now_iso,
            "board": {
                "cols": cols,
                "rows": rows,
                "t0_from_left_bottom": {"x": self.cfg.t0_x, "y": self.cfg.t0_y},
                "positive_direction": "right_up",
            },
            "left": {
                "corner_count": left_corner_count,
                "points": self._build_points_payload(left_pts, cols, rows),
            },
            "right": {
                "corner_count": right_corner_count,
                "points": self._build_points_payload(right_pts, cols, rows),
            },
            "images": {
                "left": left_image_name,
                "right": right_image_name,
            },
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        self._saved_groups += 1
        self._last_saved_file = str(out_path)
        self._last_capture_time = now_iso
        return str(out_path)

    def _loop(self):
        assert self._cap0 is not None and self._cap1 is not None
        q = int(max(40, min(95, self.cfg.jpeg_quality)))
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), q]

        while self._running:
            try:
                ok0, frame0 = self._cap0.read()
                ok1, frame1 = self._cap1.read()
                if not ok0 or not ok1:
                    with self._lock:
                        self._status_text = "相机帧读取失败，等待恢复..."
                    time.sleep(0.01)
                    continue

                if self.cfg.swap_lr:
                    frame0, frame1 = frame1, frame0

                gray_l = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                gray_r = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

                self._loop_index += 1
                detect_step = max(1, int(self.cfg.detect_every_n_frames))
                force_detect = self._pending_triggers > 0
                run_detect = force_detect or (self._loop_index % detect_step == 0)

                if run_detect:
                    found_l, pts_l, vis_l, cnt_l, cols_l, rows_l = self.detector.detect(gray_l)
                    found_r, pts_r, vis_r, cnt_r, cols_r, rows_r = self.detector.detect(gray_r)
                else:
                    found_l, pts_l, cnt_l, cols_l, rows_l = self._last_left_ok, None, self._last_left_corners, self.cfg.board_cols, self.cfg.board_rows
                    found_r, pts_r, cnt_r, cols_r, rows_r = self._last_right_ok, None, self._last_right_corners, self.cfg.board_cols, self.cfg.board_rows
                    vis_l = frame0.copy()
                    vis_r = frame1.copy()

                self._frame_counter += 1
                now_ts = time.time()
                dt = now_ts - self._fps_last_ts
                if dt >= 1.0:
                    self._fps = self._frame_counter / dt
                    self._frame_counter = 0
                    self._fps_last_ts = now_ts

                same_pattern = found_l and found_r and (cols_l == cols_r) and (rows_l == rows_r)
                if same_pattern:
                    self._last_pattern = f"{cols_l}x{rows_l}"

                self._last_left_ok = bool(found_l)
                self._last_right_ok = bool(found_r)
                self._last_left_corners = int(cnt_l)
                self._last_right_corners = int(cnt_r)

                save_now = False
                with self._lock:
                    if (
                        self._pending_triggers > 0
                        and same_pattern
                        and pts_l is not None
                        and pts_r is not None
                    ):
                        self._ensure_session_dir()
                        saved_path = self._save_group(
                            pts_l,
                            pts_r,
                            cnt_l,
                            cnt_r,
                            cols_l,
                            rows_l,
                            frame0,
                            frame1,
                        )
                        self._pending_triggers -= 1
                        save_now = True

                        if self._saved_groups >= self._target_groups:
                            self._status_text = (
                                f"✅ 已完成目标组数 {self._saved_groups}/{self._target_groups}，可继续手动触发"
                            )
                        else:
                            self._status_text = (
                                f"✅ 已保存第 {self._saved_groups} 组: {os.path.basename(saved_path)}"
                            )
                    else:
                        if found_l and found_r and not same_pattern:
                            self._status_text = (
                                f"左右识别规格异常 L:{cols_l}x{rows_l} R:{cols_r}x{rows_r}（期望 {self.cfg.board_cols}x{self.cfg.board_rows}）"
                            )
                        elif found_l and found_r:
                            if self._pending_triggers > 0:
                                self._status_text = "检测到棋盘，等待写入..."
                            else:
                                self._status_text = (
                                    f"双目检测正常(内角点 {self._last_pattern}) | FPS:{self._fps:.1f} | 点击“开始一组采集” ({self._saved_groups}/{self._target_groups})"
                                )
                        else:
                            self._status_text = (
                                f"请将棋盘放入双目视野 | 期望内角点: {self.cfg.board_cols}x{self.cfg.board_rows} | "
                                f"L:{cnt_l} R:{cnt_r} | FPS:{self._fps:.1f} | "
                                f"已采集 {self._saved_groups}/{self._target_groups}"
                            )

                h, w = vis_l.shape[:2]
                combined = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
                combined[:, :w] = vis_l
                combined[:, w + 4:] = vis_r
                combined[:, w:w + 4] = (90, 90, 90)

                color = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
                cv2.putText(combined, f"Left corners: {cnt_l}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(combined, f"Right corners: {cnt_r}", (w + 16, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(combined, f"Expected: {self.cfg.board_cols}x{self.cfg.board_rows}", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                cv2.putText(combined, f"L:{'OK' if found_l else 'NG'}  R:{'OK' if found_r else 'NG'}  FPS:{self._fps:.1f}",
                            (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                with self._lock:
                    cv2.putText(
                        combined,
                        f"Saved: {self._saved_groups}/{self._target_groups}  Pending: {self._pending_triggers}",
                        (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                if save_now:
                    cv2.rectangle(combined, (0, 0), (combined.shape[1] - 1, combined.shape[0] - 1), (0, 255, 0), 6)

                scale = float(self.cfg.preview_scale)
                if 0.2 <= scale < 0.999:
                    combined = cv2.resize(combined, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                ok, jpg = cv2.imencode(".jpg", combined, enc)
                if ok:
                    with self._lock:
                        self._jpeg_preview = jpg.tobytes()
            except Exception as e:
                with self._lock:
                    self._status_text = f"采集线程异常: {e}"
                    self._jpeg_preview = _make_placeholder_jpeg("Loop exception, retrying...")
                time.sleep(0.05)


# ═══════════════════════════════════════════════════════════════
# Web
# ═══════════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang=\"zh\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>双目棋盘坐标测试</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;color:#eee;font-family:system-ui,sans-serif}
header{padding:14px 20px;background:#111;border-bottom:1px solid #333}
header h1{font-size:18px}
.main{padding:16px;max-width:1200px;margin:0 auto}
.status{font-size:15px;line-height:1.5;padding:12px;margin:10px 0;background:#181818;border-radius:8px;min-height:46px}
img{width:100%;border-radius:8px;border:1px solid #333;display:block}
.controls{margin-top:12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.btn{padding:10px 16px;font-size:14px;border:none;border-radius:8px;cursor:pointer;font-weight:600;color:#fff}
.btn-blue{background:#2563eb}.btn-blue:hover{background:#1d4ed8}
.btn-green{background:#16a34a}.btn-green:hover{background:#15803d}
.btn-orange{background:#ea580c}.btn-orange:hover{background:#c2410c}
.btn-red{background:#dc2626}.btn-red:hover{background:#b91c1c}
input{padding:10px 12px;border:1px solid #444;border-radius:8px;background:#0f0f0f;color:#eee;min-width:120px}
.meta{font-size:13px;color:#9ca3af;margin-top:10px;line-height:1.6}
.code{font-family:ui-monospace,monospace;color:#93c5fd}
</style>
</head><body>
<header><h1>🎯 双目棋盘坐标测试</h1></header>
<div class=\"main\">
  <div id=\"status\" class=\"status\">加载中...</div>
  <img src=\"/stream\" alt=\"preview\">

  <div class=\"controls\">
    <button class=\"btn btn-blue\" onclick=\"newSession()\">新建会话</button>
    <button class=\"btn btn-green\" onclick=\"startGroup()\">开始一组采集</button>
    <input id=\"target\" type=\"number\" min=\"1\" value=\"10\" />
    <button class=\"btn btn-orange\" onclick=\"setTarget()\">设置目标组数</button>
    <button class=\"btn btn-red\" onclick=\"finish()\">结束程序</button>
  </div>

  <div class=\"meta\">
    数据目录: <span id=\"sessionDir\" class=\"code\">-</span><br>
        最近文件: <span id="lastFile" class="code">-</span><br>
        当前识别内角点规格: <span id="pattern" class="code">-</span><br>
        期望内角点规格: <span id="expected" class="code">-</span><br>
        检测状态: L=<span id="lok" class="code">-</span> R=<span id="rok" class="code">-</span><br>
        实时角点数: L=<span id="lcnt" class="code">-</span> R=<span id="rcnt" class="code">-</span><br>
        实时FPS: <span id="fps" class="code">-</span> | 最近写盘时间: <span id="lastTs" class="code">-</span>
  </div>
</div>
<script>
async function getJson(url){const r=await fetch(url);return await r.json();}
async function refresh(){
  const d=await getJson('/api/status');
  document.getElementById('status').textContent =
    `${d.status} | 已保存 ${d.saved_groups}/${d.target_groups} | 待触发 ${d.pending_triggers}`;
  document.getElementById('target').value = d.target_groups;
  document.getElementById('sessionDir').textContent = d.session_dir || '-';
  document.getElementById('lastFile').textContent = d.last_saved_file || '-';
    document.getElementById('pattern').textContent = d.last_pattern || '-';
    document.getElementById('expected').textContent = d.expected_pattern || '-';
    document.getElementById('lok').textContent = d.left_ok ? 'OK' : 'NG';
    document.getElementById('rok').textContent = d.right_ok ? 'OK' : 'NG';
    document.getElementById('lcnt').textContent = d.left_corners ?? '-';
    document.getElementById('rcnt').textContent = d.right_corners ?? '-';
    document.getElementById('fps').textContent = d.fps ?? '-';
    document.getElementById('lastTs').textContent = d.last_capture_time || '-';
}
async function newSession(){ await getJson('/api/new_session'); await refresh(); }
async function startGroup(){ await getJson('/api/start_group'); await refresh(); }
async function setTarget(){
  const n = Number(document.getElementById('target').value || 1);
  await getJson('/api/set_target?count=' + encodeURIComponent(String(n)));
  await refresh();
}
async function finish(){ await getJson('/api/finish'); }
setInterval(refresh, 1000);
refresh();
</script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    session: Optional[NinePointSession] = None
    should_stop = False

    def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)

        if path in ("/", "/index.html"):
            return self._html(HTML.encode("utf-8"))

        if path == "/stream" and self.session:
            return self._mjpeg()

        if path == "/api/status" and self.session:
            return self._json(self.session.get_status())

        if path == "/api/new_session" and self.session:
            return self._json(self.session.new_session())

        if path == "/api/start_group" and self.session:
            return self._json(self.session.trigger_group())

        if path == "/api/set_target" and self.session:
            raw = query.get("count", ["1"])[0]
            try:
                count = int(raw)
            except ValueError:
                count = 1
            return self._json(self.session.set_target_groups(count))

        if path == "/api/finish":
            Handler.should_stop = True
            payload: Dict[str, object]
            if self.session:
                payload = self.session.get_status()
            else:
                payload = {"status": "stopped"}
            return self._json(payload)

        self.send_error(HTTPStatus.NOT_FOUND)

    def _html(self, body: bytes):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Dict[str, object]):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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
                interval = self.session.stream_interval if self.session else 0.05
                time.sleep(interval)
        except (ConnectionResetError, BrokenPipeError):
            pass


class ReusableServer(ThreadingHTTPServer):
    allow_reuse_address = True


def _run_server(cfg: Config) -> ThreadingHTTPServer:
    server = ReusableServer((cfg.host, cfg.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ═══════════════════════════════════════════════════════════════
# 命令行
# ═══════════════════════════════════════════════════════════════
def load_default_config() -> Config:
    return Config()


def main() -> int:
    cfg = load_default_config()

    print("=" * 60)
    print("  双目棋盘坐标测试程序")
    print(f"  棋盘: {cfg.board_cols}x{cfg.board_rows}")
    print(f"  T0(左下角坐标系): ({cfg.t0_x}, {cfg.t0_y})")
    print(f"  目标组数: {cfg.target_groups}")
    print(f"  输出目录: {cfg.output_dir}")
    print(f"  WebUI: http://<jetson-ip>:{cfg.port}")
    print("=" * 60)

    session = NinePointSession(cfg)
    try:
        session.start()
    except Exception as e:
        print(f"[ERR] 相机启动失败: {e}")
        return 1

    Handler.session = session
    Handler.should_stop = False
    server = _run_server(cfg)

    print(f"[INFO] 服务已启动: http://{cfg.host}:{cfg.port}")
    print("[INFO] 在 WebUI 中手动开始每一组采集，Ctrl+C 可退出")

    try:
        while not Handler.should_stop:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C，准备退出")

    session.stop()
    server.shutdown()
    server.server_close()
    print("[INFO] 程序已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

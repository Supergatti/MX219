#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FocusDevice:
    path: str
    min_value: int
    max_value: int
    step: int
    default: int


def run_cmd(command: list[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(command, capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def build_argus_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv ! video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )


def build_v4l2_pipeline(device: str, width: int, height: int, fps: int) -> str:
    return (
        f"v4l2src device={device} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, framerate=(fraction){fps}/1 ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=1"
    )


def open_camera(source: str, camera: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if source == "argus":
        sensor_id = int(camera)
        pipeline = build_argus_pipeline(sensor_id, width, height, fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        pipeline = build_v4l2_pipeline(camera, width, height, fps)
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: source={source}, camera={camera}, pipeline={pipeline}")
    return cap


def open_camera_with_fallback(source: str, camera: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if source != "argus":
        return open_camera(source, camera, width, height, fps)

    fallback_profiles = [
        (width, height, fps),
        (1920, 1080, min(fps, 30)),
        (1640, 1232, min(fps, 30)),
        (1280, 720, min(fps, 30)),
        (640, 480, min(fps, 30)),
    ]

    seen = set()
    last_exc: Optional[Exception] = None
    for w, h, f in fallback_profiles:
        profile = (w, h, f)
        if profile in seen:
            continue
        seen.add(profile)
        try:
            print(f"[INFO] 尝试打开相机 {camera}: {w}x{h}@{f}")
            return open_camera(source, camera, w, h, f)
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] 相机 {camera} 打开失败: {exc}")
            time.sleep(0.4)

    raise RuntimeError(f"相机 {camera} 所有候选分辨率均失败，最后错误: {last_exc}")


def focus_score(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def parse_focus_device(device: str) -> Optional[FocusDevice]:
    if not shutil.which("v4l2-ctl"):
        return None

    ret, out, err = run_cmd(["v4l2-ctl", "-d", device, "-L"])
    if ret != 0:
        print(f"[WARN] 无法读取 {device} 控件: {err}")
        return None

    line = None
    for raw in out.splitlines():
        if "focus_absolute" in raw:
            line = raw
            break

    if line is None:
        return None

    min_match = re.search(r"min=(-?\d+)", line)
    max_match = re.search(r"max=(-?\d+)", line)
    step_match = re.search(r"step=(-?\d+)", line)
    def_match = re.search(r"default=(-?\d+)", line)

    if not (min_match and max_match and step_match and def_match):
        return None

    return FocusDevice(
        path=device,
        min_value=int(min_match.group(1)),
        max_value=int(max_match.group(1)),
        step=max(1, int(step_match.group(1))),
        default=int(def_match.group(1)),
    )


def set_focus_value(device: FocusDevice, value: int) -> bool:
    value = max(device.min_value, min(device.max_value, value))
    # 先尝试关闭自动对焦，再设置手动焦点
    run_cmd(["v4l2-ctl", "-d", device.path, "-c", "focus_auto=0"])
    ret, _, err = run_cmd(["v4l2-ctl", "-d", device.path, "-c", f"focus_absolute={value}"])
    if ret != 0:
        print(f"[WARN] 设置焦点失败 {device.path}={value}: {err}")
        return False
    return True


def autofocus_scan(
    cap: cv2.VideoCapture,
    device: FocusDevice,
    sample_frames: int = 3,
    scan_points: int = 12,
) -> Optional[Tuple[int, float]]:
    values = np.linspace(device.min_value, device.max_value, scan_points).astype(int)
    best_value = None
    best_score = -1.0

    for value in values:
        if not set_focus_value(device, int(value)):
            continue

        local_scores = []
        for _ in range(sample_frames):
            ok, frame = cap.read()
            if not ok:
                continue
            local_scores.append(focus_score(frame))

        if not local_scores:
            continue

        score = float(np.median(local_scores))
        if score > best_score:
            best_score = score
            best_value = int(value)

    if best_value is None:
        return None

    set_focus_value(device, best_value)
    return best_value, best_score


def annotate(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return out


def add_center_zoom_preview(frame: np.ndarray, zoom_ratio: float = 3.0, box_ratio: float = 0.28) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    crop_w = max(40, int(w * box_ratio))
    crop_h = max(40, int(h * box_ratio))
    x1 = w // 2 - crop_w // 2
    y1 = h // 2 - crop_h // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    roi = out[y1:y2, x1:x2]
    if roi.size == 0:
        return out

    zoom_w = min(int(crop_w * zoom_ratio), w // 3)
    zoom_h = min(int(crop_h * zoom_ratio), h // 3)
    zoom = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    px1, py1 = 10, h - zoom_h - 10
    px2, py2 = px1 + zoom_w, py1 + zoom_h

    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
    out[py1:py2, px1:px2] = zoom
    cv2.rectangle(out, (px1, py1), (px2, py2), (255, 255, 0), 2)
    cv2.putText(out, "center zoom", (px1 + 4, py1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    return out


def draw_focus_bar(frame: np.ndarray, score: float, score_ref: float = 1000.0) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_w = int(w * 0.35)
    bar_h = 16
    x1, y1 = w - bar_w - 20, 18
    x2, y2 = x1 + bar_w, y1 + bar_h
    cv2.rectangle(out, (x1, y1), (x2, y2), (220, 220, 220), 1)

    ratio = max(0.0, min(1.0, score / score_ref))
    fill = int(bar_w * ratio)
    color = (0, 0, 255) if ratio < 0.35 else (0, 255, 255) if ratio < 0.7 else (0, 255, 0)
    if fill > 0:
        cv2.rectangle(out, (x1 + 1, y1 + 1), (x1 + fill - 1, y2 - 1), color, -1)
    cv2.putText(out, "focus", (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jetson MX219 双目采集与对焦尝试")
    parser.add_argument("--source", choices=["argus", "v4l2"], default="argus", help="相机源类型")
    parser.add_argument("--cam0", default="0", help="相机0: argus下为sensor-id，v4l2下为设备路径")
    parser.add_argument("--cam1", default="1", help="相机1: argus下为sensor-id，v4l2下为设备路径")
    parser.add_argument("--focus-dev0", default="", help="相机0对焦控制设备，如 /dev/video0")
    parser.add_argument("--focus-dev1", default="", help="相机1对焦控制设备，如 /dev/video1")
    parser.add_argument("--width", type=int, default=1640)
    parser.add_argument("--height", type=int, default=1232)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--view-width", type=int, default=960, help="单路预览窗口宽")
    parser.add_argument("--view-height", type=int, default=720, help="单路预览窗口高")
    parser.add_argument("--no-gui", action="store_true", help="无图形界面模式，仅输出日志")
    parser.add_argument("--try-focus-on-start", action="store_true", help="启动时执行一次扫焦")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.no_gui and (not os.environ.get("DISPLAY")):
        print("[WARN] 未检测到 DISPLAY，可能无法弹出窗口。可在桌面会话运行，或加 --no-gui。")

    try:
        cap0 = open_camera_with_fallback(args.source, args.cam0, args.width, args.height, args.fps)
        time.sleep(0.35)
        cap1 = open_camera_with_fallback(args.source, args.cam1, args.width, args.height, args.fps)
    except Exception as exc:
        print(f"[ERR] {exc}")
        print("[HINT] 可尝试: 1) sudo systemctl restart nvargus-daemon 2) 降分辨率 3) 关闭占用相机进程")
        return 1

    focus0 = parse_focus_device(args.focus_dev0) if args.focus_dev0 else None
    focus1 = parse_focus_device(args.focus_dev1) if args.focus_dev1 else None

    if focus0:
        print(f"[INFO] cam0 支持 focus_absolute: {focus0.min_value}~{focus0.max_value}, step={focus0.step}")
    if focus1:
        print(f"[INFO] cam1 支持 focus_absolute: {focus1.min_value}~{focus1.max_value}, step={focus1.step}")
    if not focus0 and not focus1:
        print("[INFO] 未检测到可编程对焦控件，MX219 常见为固定焦。将仅显示清晰度评分供手动调焦。")

    if args.try_focus_on_start:
        if focus0:
            result = autofocus_scan(cap0, focus0)
            print(f"[INFO] cam0 启动扫焦结果: {result}")
        if focus1:
            result = autofocus_scan(cap1, focus1)
            print(f"[INFO] cam1 启动扫焦结果: {result}")

    print("[INFO] 按键: q 退出, f 对可编程镜头执行扫焦")
    print("[INFO] 显示窗口: Left Camera / Right Camera")

    if not args.no_gui:
        cv2.namedWindow("Left Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Right Camera", cv2.WINDOW_NORMAL)

    while True:
        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()

        if not ok0 or not ok1:
            print("[WARN] 读取帧失败，退出。")
            break

        score0 = focus_score(frame0)
        score1 = focus_score(frame1)

        txt0 = f"cam0 sharpness={score0:.1f}"
        txt1 = f"cam1 sharpness={score1:.1f}"

        frame0 = annotate(frame0, txt0, (0, 255, 0))
        frame1 = annotate(frame1, txt1, (0, 255, 0))
        frame0 = add_center_zoom_preview(frame0)
        frame1 = add_center_zoom_preview(frame1)
        frame0 = draw_focus_bar(frame0, score0)
        frame1 = draw_focus_bar(frame1, score1)

        vis0 = cv2.resize(frame0, (args.view_width, args.view_height))
        vis1 = cv2.resize(frame1, (args.view_width, args.view_height))

        key = 255
        if not args.no_gui:
            cv2.imshow("Left Camera", vis0)
            cv2.imshow("Right Camera", vis1)
            key = cv2.waitKey(1) & 0xFF
        else:
            print(f"[INFO] cam0={score0:.1f}, cam1={score1:.1f}")
            time.sleep(0.12)

        if key == ord("q"):
            break
        if key == ord("f"):
            t0 = time.time()
            if focus0:
                result = autofocus_scan(cap0, focus0)
                print(f"[INFO] cam0 扫焦结果: {result}")
            if focus1:
                result = autofocus_scan(cap1, focus1)
                print(f"[INFO] cam1 扫焦结果: {result}")
            if not focus0 and not focus1:
                print("[INFO] 当前硬件无可编程焦点控件，无法自动扫焦。")
            print(f"[INFO] 扫焦耗时: {time.time() - t0:.2f}s")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import cv2
import numpy as np


ROTATE_CODE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

CAM0 = 0
CAM1 = 1
WIDTH = 1280
HEIGHT = 720
FPS = 30
ROTATE0 = 0
ROTATE1 = 0
SWAP_LR = True
FLIP_METHOD = 2

SQUARES_X = 9
SQUARES_Y = 6
SQUARE_LENGTH = 0.025
MARKER_LENGTH = 0.018
MIN_CORNERS = 6
MIN_COMMON_CORNERS = 6
PAIRS = 20
MIN_SAVE_INTERVAL = 0.8


@dataclass
class CaptureConfig:
    cam0: int
    cam1: int
    width: int
    height: int
    fps: int
    flip_method: int
    rotate0: int
    rotate1: int
    swap_lr: bool


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


def open_camera(sensor_id: int, width: int, height: int, fps: int, flip_method: int) -> cv2.VideoCapture:
    pipeline = build_argus_pipeline(sensor_id, width, height, fps, flip_method)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开相机: sensor-id={sensor_id}, pipeline={pipeline}")
    return cap


def open_stereo_cameras_with_fallback(
    cam0: int,
    cam1: int,
    width: int,
    height: int,
    fps: int,
    flip_method: int,
) -> Tuple[cv2.VideoCapture, cv2.VideoCapture, Tuple[int, int, int]]:
    profiles = [
        (width, height, fps),
        (1280, 720, min(fps, 30)),
        (640, 480, min(fps, 30)),
    ]

    last_error: Optional[Exception] = None
    for w, h, f in dict.fromkeys(profiles):
        profile = (w, h, f)

        cap0 = None
        cap1 = None
        try:
            print(f"[INFO] 成对尝试双目 argus: {w}x{h}@{f}")
            cap0 = open_camera(cam0, w, h, f, flip_method)
            time.sleep(0.25)
            cap1 = open_camera(cam1, w, h, f, flip_method)

            ensure_camera_stream_ready(cap0, f"cam0({cam0})")
            ensure_camera_stream_ready(cap1, f"cam1({cam1})")
            return cap0, cap1, profile
        except Exception as exc:
            last_error = exc
            print(f"[WARN] 双目配置失败 {w}x{h}@{f}: {exc}")
            if cap0 is not None:
                cap0.release()
            if cap1 is not None:
                cap1.release()
            time.sleep(0.4)

    raise RuntimeError(f"双目相机所有候选分辨率均失败，最后错误: {last_error}")


def ensure_camera_stream_ready(cap: cv2.VideoCapture, name: str, retries: int = 25, delay: float = 0.04) -> None:
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return
        time.sleep(delay)
    raise RuntimeError(f"{name} 未能输出有效帧")


def rotate_frame(frame: np.ndarray, angle: int) -> np.ndarray:
    rotate_code = ROTATE_CODE.get(angle)
    return frame if rotate_code is None else cv2.rotate(frame, rotate_code)


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
) -> Tuple[Any, Any]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    except AttributeError:
        board_create = getattr(cv2.aruco, "CharucoBoard_create")
        board = board_create(squares_x, squares_y, square_length, marker_length, dictionary)
    return dictionary, board


def get_board_corners(board: Any) -> np.ndarray:
    corners = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    return cast(np.ndarray, corners).astype(np.float32)


def detect_charuco(
    gray: np.ndarray,
    dictionary: Any,
    board: Any,
    detector_params: Any,
    min_corners: int,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Any, Any]:
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=detector_params)
    if marker_ids is None or len(marker_ids) == 0:
        return False, None, None, marker_corners, marker_ids

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners, marker_ids, gray, board
    )
    if retval is None or retval < min_corners or charuco_ids is None:
        return False, charuco_corners, charuco_ids, marker_corners, marker_ids

    return True, charuco_corners, charuco_ids, marker_corners, marker_ids


def ensure_dirs(base_dir: str) -> Tuple[str, str]:
    left_dir = os.path.join(base_dir, "images", "left")
    right_dir = os.path.join(base_dir, "images", "right")
    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)
    return left_dir, right_dir


def capture_pairs(
    config: CaptureConfig,
    output_dir: str,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    min_corners: int,
    min_common_corners: int,
    min_interval: float,
    required_pairs: int,
) -> int:
    left_dir, right_dir = ensure_dirs(output_dir)
    dictionary, board = create_charuco_board(squares_x, squares_y, square_length, marker_length)
    detector_params = cv2.aruco.DetectorParameters()

    cap0, cap1, used_profile = open_stereo_cameras_with_fallback(
        config.cam0, config.cam1, config.width, config.height, config.fps, config.flip_method
    )

    try:
        cv2.namedWindow("Calib Left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Calib Right", cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        cap0.release()
        cap1.release()
        print("[ERR] 无法初始化图形窗口，当前环境可能没有桌面显示会话。")
        print("[HINT] 请在 Jetson 本机桌面终端运行标定采集，或启用 X11 转发后再试。")
        print(f"[HINT] OpenCV 错误: {exc}")
        return 0

    last_save = 0.0
    count = 0
    print("[INFO] 按键: s 保存一对图像(需左右都检测到 ChArUco 角点), q 退出")
    print(f"[INFO] 当前采集配置: {used_profile[0]}x{used_profile[1]}@{used_profile[2]}")
    print(f"[INFO] 当前旋转配置: flip_method={config.flip_method}, rotate0={config.rotate0}, rotate1={config.rotate1}")
    print(f"[INFO] 当前左右互换: swap_lr={config.swap_lr}")
    print(f"[INFO] 目标采集数量: {required_pairs}")

    try:
        while True:
            ok0, frame0 = cap0.read()
            ok1, frame1 = cap1.read()
            if not ok0 or not ok1:
                print("[WARN] 读取相机帧失败，结束采集。")
                break

            frame0 = rotate_frame(frame0, config.rotate0)
            frame1 = rotate_frame(frame1, config.rotate1)

            if config.swap_lr:
                frame0, frame1 = frame1, frame0

            gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            found0, corners0, ids0, mc0, mid0 = detect_charuco(
                gray0, dictionary, board, detector_params, min_corners
            )
            found1, corners1, ids1, mc1, mid1 = detect_charuco(
                gray1, dictionary, board, detector_params, min_corners
            )

            vis0 = frame0.copy()
            vis1 = frame1.copy()

            if mid0 is not None:
                cv2.aruco.drawDetectedMarkers(vis0, mc0, mid0)
            if mid1 is not None:
                cv2.aruco.drawDetectedMarkers(vis1, mc1, mid1)
            if ids0 is not None and corners0 is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis0, corners0, ids0, (0, 255, 0))
            if ids1 is not None and corners1 is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis1, corners1, ids1, (0, 255, 0))

            m0 = 0 if mid0 is None else int(len(mid0))
            m1 = 0 if mid1 is None else int(len(mid1))
            c0 = 0 if ids0 is None else int(len(ids0))
            c1 = 0 if ids1 is None else int(len(ids1))
            common = 0
            if ids0 is not None and ids1 is not None:
                common = int(len(np.intersect1d(ids0.flatten(), ids1.flatten())))

            ready_left = c0 >= min_corners
            ready_right = c1 >= min_corners
            ready_common = common >= min_common_corners
            ready_save = ready_left and ready_right and ready_common

            status = (
                f"pairs={count} | ML={m0} MR={m1} | L={c0}/{min_corners} R={c1}/{min_corners} "
                f"C={common}/{min_common_corners}"
            )
            color = (0, 255, 0) if ready_save else (0, 200, 255)
            cv2.putText(vis0, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            cv2.putText(vis1, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            cv2.imshow("Calib Left", vis0)
            cv2.imshow("Calib Right", vis1)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                now = time.time()
                if now - last_save < min_interval:
                    print(f"[WARN] 保存太快，请至少间隔 {min_interval:.1f}s。")
                    continue

                if not ready_save:
                    reasons = []
                    if not ready_left:
                        reasons.append(f"左角点不足 L={c0} < {min_corners}")
                    if not ready_right:
                        reasons.append(f"右角点不足 R={c1} < {min_corners}")
                    if not ready_common:
                        reasons.append(f"共同角点不足 C={common} < {min_common_corners}")
                    print("[WARN] 本次未保存： " + "；".join(reasons))
                    if (m0 >= 4 and c0 == 0) or (m1 >= 4 and c1 == 0):
                        print(
                            "[HINT] Marker 已识别但 ChArUco 角点为 0，通常是板参数不匹配。"
                            "请检查 squaresX/squaresY 是否反了，当前图像常见应为 --squares-x 9 --squares-y 6。"
                        )
                    continue

                name = f"pair_{count:03d}.png"
                left_path = os.path.join(left_dir, name)
                right_path = os.path.join(right_dir, name)
                cv2.imwrite(left_path, frame0)
                cv2.imwrite(right_path, frame1)
                count += 1
                last_save = now
                print(f"[INFO] 已保存: {left_path} | {right_path}")

                if count >= required_pairs:
                    print("[INFO] 达到目标采集数量，结束采集。")
                    break
    finally:
        cap0.release()
        cap1.release()
        cv2.destroyAllWindows()

    return count


def collect_image_pairs(base_dir: str) -> List[Tuple[str, str]]:
    left_glob = os.path.join(base_dir, "images", "left", "pair_*.png")
    left_paths = sorted(glob.glob(left_glob))
    pairs: List[Tuple[str, str]] = []
    for left_path in left_paths:
        name = os.path.basename(left_path)
        right_path = os.path.join(base_dir, "images", "right", name)
        if os.path.exists(right_path):
            pairs.append((left_path, right_path))
    return pairs


def select_common_points(
    charuco_corners_l: np.ndarray,
    charuco_ids_l: np.ndarray,
    charuco_corners_r: np.ndarray,
    charuco_ids_r: np.ndarray,
    board_corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids_l = charuco_ids_l.flatten().astype(int)
    ids_r = charuco_ids_r.flatten().astype(int)
    common_ids = np.intersect1d(ids_l, ids_r)

    idx_l = {cid: i for i, cid in enumerate(ids_l)}
    idx_r = {cid: i for i, cid in enumerate(ids_r)}

    obj = np.array([board_corners[cid] for cid in common_ids], dtype=np.float32)
    img_l = np.array([charuco_corners_l[idx_l[cid], 0, :] for cid in common_ids], dtype=np.float32)
    img_r = np.array([charuco_corners_r[idx_r[cid], 0, :] for cid in common_ids], dtype=np.float32)
    return obj, img_l, img_r


def calibrate_stereo(
    base_dir: str,
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    min_corners: int,
) -> int:
    dictionary, board = create_charuco_board(squares_x, squares_y, square_length, marker_length)
    detector_params = cv2.aruco.DetectorParameters()
    board_corners = get_board_corners(board)

    pairs = collect_image_pairs(base_dir)
    if len(pairs) < 10:
        print(f"[ERR] 可用图像对过少: {len(pairs)}，建议至少 15-20 对。")
        return 1

    obj_points_stereo: List[np.ndarray] = []
    img_points_l_stereo: List[np.ndarray] = []
    img_points_r_stereo: List[np.ndarray] = []

    charuco_corners_l_all: List[np.ndarray] = []
    charuco_ids_l_all: List[np.ndarray] = []
    charuco_corners_r_all: List[np.ndarray] = []
    charuco_ids_r_all: List[np.ndarray] = []

    image_size: Optional[Tuple[int, int]] = None
    used_pairs = 0

    for left_path, right_path in pairs:
        img_l = cv2.imread(left_path)
        img_r = cv2.imread(right_path)
        if img_l is None or img_r is None:
            continue

        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

        found_l, corners_l, ids_l, _, _ = detect_charuco(gray_l, dictionary, board, detector_params, min_corners)
        found_r, corners_r, ids_r, _, _ = detect_charuco(gray_r, dictionary, board, detector_params, min_corners)
        if not (found_l and found_r):
            continue
        if corners_l is None or ids_l is None or corners_r is None or ids_r is None:
            continue

        if image_size is None:
            image_size = (gray_l.shape[1], gray_l.shape[0])

        obj_pts, img_pts_l, img_pts_r = select_common_points(corners_l, ids_l, corners_r, ids_r, board_corners)
        if len(obj_pts) < min_corners:
            continue

        obj_points_stereo.append(obj_pts)
        img_points_l_stereo.append(img_pts_l)
        img_points_r_stereo.append(img_pts_r)

        charuco_corners_l_all.append(corners_l)
        charuco_ids_l_all.append(ids_l)
        charuco_corners_r_all.append(corners_r)
        charuco_ids_r_all.append(ids_r)
        used_pairs += 1

    if used_pairs < 10 or image_size is None:
        print(f"[ERR] 有效 ChArUco 图像对不足: {used_pairs}")
        return 1

    print(f"[INFO] 进入标定，使用有效图像对: {used_pairs}")

    flags_mono = cv2.CALIB_RATIONAL_MODEL
    criteria_mono = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    ret_l, mtx_l, dist_l, _, _ = cv2.aruco.calibrateCameraCharuco(
        charuco_corners_l_all,
        charuco_ids_l_all,
        board,
        image_size,
        cast(Any, None),
        cast(Any, None),
        flags=flags_mono,
        criteria=criteria_mono,
    )
    ret_r, mtx_r, dist_r, _, _ = cv2.aruco.calibrateCameraCharuco(
        charuco_corners_r_all,
        charuco_ids_r_all,
        board,
        image_size,
        cast(Any, None),
        cast(Any, None),
        flags=flags_mono,
        criteria=criteria_mono,
    )

    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7)
    ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points_stereo,
        img_points_l_stereo,
        img_points_r_stereo,
        mtx_l,
        dist_l,
        mtx_r,
        dist_r,
        image_size,
        criteria=criteria_stereo,
        flags=flags_stereo,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, R, T, alpha=0
    )

    map1x, map1y = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, image_size, cv2.CV_32FC1)

    calib_file = os.path.join(base_dir, "stereo_calib.npz")
    np.savez(
        calib_file,
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix_left=mtx_l,
        dist_coeffs_left=dist_l,
        camera_matrix_right=mtx_r,
        dist_coeffs_right=dist_r,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        roi1=np.array(roi1),
        roi2=np.array(roi2),
        reproj_error_left=np.array([ret_l], dtype=np.float64),
        reproj_error_right=np.array([ret_r], dtype=np.float64),
        reproj_error_stereo=np.array([ret_s], dtype=np.float64),
    )

    summary = {
        "pairs_found": len(pairs),
        "pairs_used": used_pairs,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "board": {
            "type": "ChArUco",
            "dictionary": "DICT_4X4_50",
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_length": square_length,
            "marker_length": marker_length,
        },
        "reproj_error_left": float(ret_l),
        "reproj_error_right": float(ret_r),
        "reproj_error_stereo": float(ret_s),
        "output": calib_file,
    }
    summary_file = os.path.join(base_dir, "calibration_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 标定完成，参数保存到: {calib_file}")
    print(f"[INFO] 摘要保存到: {summary_file}")
    print(
        "[INFO] 重投影误差: "
        f"left={ret_l:.4f}, right={ret_r:.4f}, stereo={ret_s:.4f} (越小越好，通常 <1 较理想)"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MX219 双目标定工具（ChArUco 采集 + 求解）")
    parser.add_argument("--mode", choices=["capture", "calibrate", "all"], default="all")
    parser.add_argument("--output-dir", default="calib_data", help="标定数据输出目录")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode in ["capture", "all"]:
        cfg = CaptureConfig(
            cam0=CAM0,
            cam1=CAM1,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            flip_method=FLIP_METHOD,
            rotate0=ROTATE0,
            rotate1=ROTATE1,
            swap_lr=SWAP_LR,
        )
        count = capture_pairs(
            config=cfg,
            output_dir=args.output_dir,
            squares_x=SQUARES_X,
            squares_y=SQUARES_Y,
            square_length=SQUARE_LENGTH,
            marker_length=MARKER_LENGTH,
            min_corners=MIN_CORNERS,
            min_common_corners=MIN_COMMON_CORNERS,
            min_interval=MIN_SAVE_INTERVAL,
            required_pairs=PAIRS,
        )
        print(f"[INFO] 采集完成，共保存图像对: {count}")

    if args.mode in ["calibrate", "all"]:
        return calibrate_stereo(
            base_dir=args.output_dir,
            squares_x=SQUARES_X,
            squares_y=SQUARES_Y,
            square_length=SQUARE_LENGTH,
            marker_length=MARKER_LENGTH,
            min_corners=MIN_CORNERS,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

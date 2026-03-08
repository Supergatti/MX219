#!/usr/bin/env python3
"""
IMX219 CSI 摄像头 + YOLO 实时目标检测
- 优先使用 TensorRT engine（GPU 加速，自动导出）
- 摄像头先于模型初始化，避免 NVMAP 内存竞争
- 纯净检测框显示，无弹幕
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2


@dataclass
class AppConfig:
    sensor_id: int   = int(os.environ.get("CAM_ID",         "0"))
    cam_width: int   = int(os.environ.get("CAM_WIDTH",   "1280"))
    cam_height: int  = int(os.environ.get("CAM_HEIGHT",   "720"))
    cam_fps: int     = int(os.environ.get("CAM_FPS",       "60"))
    flip_method: int = int(os.environ.get("CAM_FLIP",       "2"))  # 2=旋转180°
    conf: float      = float(os.environ.get("YOLO_CONF",  "0.50"))
    iou: float       = float(os.environ.get("YOLO_IOU",   "0.45"))  # NMS 重叠阈值，越小框越少
    imgsz: int       = int(os.environ.get("YOLO_IMGSZ",   "640"))
    infer_every: int = max(1, int(os.environ.get("YOLO_INFER_EVERY", "1")))
    window_name: str = "YOLO Detection"


@dataclass
class Detection:
    cls_name: str
    conf: float
    xyxy: Tuple[int, int, int, int]


# ──────────────────────────────────────────────
#  摄像头
# ──────────────────────────────────────────────

def build_argus_pipeline(cfg: AppConfig) -> str:
    """
    IMX219 最佳 GStreamer pipeline。
    nvarguscamerasrc → NVMM 硬件解码 → BGR 输出给 OpenCV。
    """
    fm = cfg.flip_method if cfg.flip_method in range(8) else 2
    return (
        f"nvarguscamerasrc sensor-id={cfg.sensor_id} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){cfg.cam_width}, "
        f"height=(int){cfg.cam_height}, format=(string)NV12, "
        f"framerate=(fraction){cfg.cam_fps}/1 ! "
        f"nvvidconv flip-method={fm} ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoconvert ! video/x-raw, format=(string)BGR ! "
        "appsink drop=1 max-buffers=2 sync=false"
    )


def open_camera(cfg: AppConfig) -> cv2.VideoCapture:
    pipeline = build_argus_pipeline(cfg)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[摄像头] nvarguscamerasrc OK  sensor-id={cfg.sensor_id}  "
              f"{cfg.cam_width}x{cfg.cam_height}@{cfg.cam_fps}fps")
        return cap

    # 回退 V4L2（调试用）
    print(f"[摄像头] GStreamer 失败，回退 V4L2 /dev/video{cfg.sensor_id}")
    cap = cv2.VideoCapture(cfg.sensor_id)
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 sensor-id={cfg.sensor_id}\n"
            "可用 CAM_WIDTH / CAM_HEIGHT / CAM_FPS / CAM_FLIP 调整参数"
        )
    return cap


# ──────────────────────────────────────────────
#  YOLO（TensorRT 优先）
# ──────────────────────────────────────────────

def _find_pt_model() -> str:
    candidates = [
        os.environ.get("YOLO_MODEL", "").strip(),
        "/home/jetson/Desktop/MX219/yolo11n.pt",
        "/home/jetson/Desktop/MX219/yolo11n-seg.pt",
        "yolo11n.pt",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return "yolo11n.pt"


def _engine_path(pt_path: str, imgsz: int) -> str:
    base = os.path.splitext(pt_path)[0]
    return f"{base}_imgsz{imgsz}.engine"


def init_yolo(cfg: AppConfig):
    """
    加载 YOLO。优先顺序：
    1. TensorRT .engine（GPU，最快，内存效率最高）
    2. 自动从 .pt 导出 TensorRT .engine
    3. 直接 CUDA（PyTorch）
    4. CPU（兜底）

    必须在 open_camera() 之后调用——摄像头 DMA 缓冲先占 NVMAP，
    CUDA 再分配剩余内存，避免相互抢占导致 OOM。
    """
    try:
        import torch
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("缺少依赖: pip install ultralytics") from exc

    pt_path  = _find_pt_model()
    eng_path = _engine_path(pt_path, cfg.imgsz)
    use_cuda = torch.cuda.is_available()

    # 尝试加载已有 engine
    if os.path.exists(eng_path):
        try:
            model = YOLO(eng_path, task="segment")
            print(f"[YOLO] TensorRT engine 已加载: {eng_path}")
            return model, "tensorrt"
        except Exception as e:
            print(f"[YOLO] engine 加载失败 ({e})，重新导出")
            os.remove(eng_path)

    # 导出 TensorRT engine
    if use_cuda and os.path.exists(pt_path):
        try:
            print("[YOLO] 正在导出 TensorRT engine（首次约需 1-3 分钟）...")
            tmp = YOLO(pt_path)
            tmp.export(format="engine", imgsz=cfg.imgsz, device=0, half=True)
            # ultralytics 导出路径
            exported = os.path.splitext(pt_path)[0] + ".engine"
            if os.path.exists(exported):
                os.rename(exported, eng_path)
                model = YOLO(eng_path, task="segment")
                print(f"[YOLO] TensorRT engine 导出成功: {eng_path}")
                return model, "tensorrt"
        except Exception as e:
            print(f"[YOLO] TensorRT 导出失败 ({e})")

    # PyTorch CUDA
    if use_cuda:
        try:
            model = YOLO(pt_path)
            model.to("cuda:0")
            print("[YOLO] PyTorch CUDA 模式")
            return model, "cuda:0"
        except RuntimeError as e:
            print(f"[YOLO] CUDA 加载失败 ({e})，回退 CPU")

    # CPU 兜底
    model = YOLO(pt_path)
    model.to("cpu")
    print("[YOLO] CPU 模式（可设 YOLO_IMGSZ=320 YOLO_INFER_EVERY=2 提速）")
    return model, "cpu"


# ──────────────────────────────────────────────
#  推理 & 绘制
# ──────────────────────────────────────────────

def run_inference(model, frame, conf: float, iou: float, imgsz: int) -> List[Detection]:
    results = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    result  = results[0]
    boxes   = result.boxes
    names   = result.names
    dets: List[Detection] = []

    if boxes is None or len(boxes) == 0:
        return dets

    for b in boxes:
        cls_id = int(b.cls.item())
        score  = float(b.conf.item())
        # 置信度必须在 [0,1]，否则说明 task 解析仍有问题
        if not (0.0 < score <= 1.0):
            continue
        x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        dets.append(Detection(cls_name=cls_name, conf=score, xyxy=(x1, y1, x2, y2)))
    return dets


def draw_detections(frame, dets: Sequence[Detection]) -> None:
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        lx, ly = x1, max(20, y1 - 8)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 1, cv2.LINE_AA)


def draw_hud(frame, fps: float, infer_ms: float, device: str, n_det: int) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    text = (f"FPS {fps:5.1f}  Infer {infer_ms:5.1f}ms  "
            f"Det {n_det}  Dev {device}")
    cv2.putText(frame, text, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)


# ──────────────────────────────────────────────
#  主循环
# ──────────────────────────────────────────────

def main() -> int:
    cfg = AppConfig()

    # ① 先开摄像头 → DMA 缓冲先占 NVMAP
    cap = open_camera(cfg)

    # ② 再加载 YOLO → CUDA 使用剩余内存
    model, device = init_yolo(cfg)

    prev_time = time.time()
    infer_ms  = 0.0
    frame_idx = 0
    cached_det: List[Detection] = []
    fail_count = 0
    MAX_FAIL = 30  # 连续读帧失败超过此次数则报错退出

    cv2.namedWindow(cfg.window_name, cv2.WINDOW_AUTOSIZE)
    print("[主循环] 按 q 或 ESC 退出")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            if fail_count >= MAX_FAIL:
                print(f"[错误] 连续 {MAX_FAIL} 帧读取失败，摄像头流异常。")
                print("请运行: sudo systemctl restart nvargus-daemon  然后重新启动程序")
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        now = time.time()
        fps = 1.0 / max(1e-3, now - prev_time)
        prev_time = now
        frame_idx += 1

        if frame_idx % cfg.infer_every == 0:
            t0 = time.perf_counter()
            cached_det = run_inference(model, frame, cfg.conf, cfg.iou, cfg.imgsz)
            infer_ms = (time.perf_counter() - t0) * 1000.0

        draw_detections(frame, cached_det)
        draw_hud(frame, fps, infer_ms, device, len(cached_det))

        cv2.imshow(cfg.window_name, frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

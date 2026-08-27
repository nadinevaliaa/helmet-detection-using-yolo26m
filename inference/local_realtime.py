import cv2
import os
import time
import threading
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CAMERA_INDEX = 0
MODEL_PATH = "Weights/best.pt"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_DISPLAY_FPS = 30

# Karena sebelumnya tangan kanan tampil di kiri,
# maka frame perlu dibalik horizontal.
FIX_MIRROR = True

# Confidence default realtime.
CONF_THRESHOLD = 0.35
MIN_DRAW_CONF = 0.35

# Untuk realtime:
# 416 = lebih cepat
# 512 = lebih akurat
# 640 = lebih akurat lagi, tapi lebih berat
YOLO_IMGSZ = 512

# Kalau laptop terasa berat, ubah ke 416.
# Kalau ingin lebih akurat dan masih kuat, ubah ke 640.

YOLO_IOU = 0.45
CLASS_AWARE_NMS_IOU = 0.45
CROSS_CLASS_NMS_IOU = 0.60

MAX_DET = 300

# TTA jangan dipakai untuk realtime karena sangat lambat.
USE_TTA = False

WINDOW_NAME = "Helmet Detection - Realtime Optimized"


# ============================================================
# LABEL / FONT SETTING
# ============================================================

LABEL_FONT_SCALE = 0.35
LABEL_THICKNESS = 1
BOX_THICKNESS = 2

SUMMARY_FONT_SCALE = 0.50
SUMMARY_THICKNESS = 1


# ============================================================
# OPTIMASI CPU / GPU
# ============================================================

cv2.setUseOptimized(True)

CPU_COUNT = os.cpu_count() or 4
CPU_THREADS = max(2, min(6, CPU_COUNT - 2))

try:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
except Exception:
    pass

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("CUDA tersedia:", torch.cuda.is_available())
print("Device digunakan:", "GPU/CUDA" if DEVICE == 0 else "CPU")
print("CPU thread digunakan:", CPU_THREADS)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}\n"
        "Pastikan file best.pt berada di folder Weights/best.pt"
    )

model = YOLO(MODEL_PATH)

if DEVICE == 0:
    model.to("cuda")

try:
    model.fuse()
except Exception:
    pass

print("Class dari model:", model.names)


# ============================================================
# GLOBAL STATE
# ============================================================

latest_frame = None
latest_frame_id = 0

latest_detections = []
latest_processed_frame_id = -1

display_fps_value = 0.0
yolo_fps_value = 0.0
inference_ms_value = 0.0

frame_lock = threading.Lock()
detection_lock = threading.Lock()

stop_event = threading.Event()


# ============================================================
# HELPER CLASS
# ============================================================

def get_class_name(cls_id):
    try:
        names = model.names

        if isinstance(names, dict):
            return str(names.get(cls_id, f"Class {cls_id}"))

        if isinstance(names, list) and cls_id < len(names):
            return str(names[cls_id])

    except Exception:
        pass

    return f"Class {cls_id}"


def get_class_color(cls_id):
    name = get_class_name(cls_id).lower()

    if "without" in name or "no helmet" in name or "no_helmet" in name:
        return (60, 60, 230)  # merah BGR

    return (70, 200, 90)  # hijau BGR


def classify_count(cls_id):
    name = get_class_name(cls_id).lower()

    if "without" in name or "no helmet" in name or "no_helmet" in name:
        return "without"

    if "with" in name or "helmet" in name:
        return "with"

    return "other"


# ============================================================
# DETECTION STRUCTURE
# ============================================================

def detection_dict(x1, y1, x2, y2, conf, cls):
    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
        "conf": float(conf),
        "cls": int(cls)
    }


def clip_detection(det, width, height):
    det["x1"] = max(0, min(det["x1"], width - 1))
    det["x2"] = max(0, min(det["x2"], width - 1))
    det["y1"] = max(0, min(det["y1"], height - 1))
    det["y2"] = max(0, min(det["y2"], height - 1))
    return det


def box_area(det):
    return max(0, det["x2"] - det["x1"]) * max(0, det["y2"] - det["y1"])


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h

    area_a = box_area(a)
    area_b = box_area(b)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# ============================================================
# NMS FUNCTIONS
# ============================================================

def class_aware_nms(detections, iou_threshold=0.45):
    if not detections:
        return []

    final_detections = []
    class_ids = sorted(set(det["cls"] for det in detections))

    for cls_id in class_ids:
        cls_dets = [det for det in detections if det["cls"] == cls_id]
        cls_dets = sorted(cls_dets, key=lambda d: d["conf"], reverse=True)

        while cls_dets:
            best = cls_dets.pop(0)
            final_detections.append(best)

            remaining = []

            for det in cls_dets:
                overlap = box_iou(best, det)

                if overlap < iou_threshold:
                    remaining.append(det)

            cls_dets = remaining

    return sorted(final_detections, key=lambda d: d["conf"], reverse=True)


def cross_class_nms(detections, iou_threshold=0.60):
    """
    Menghapus deteksi ganda antar class pada objek yang sama.
    Confidence tertinggi dipertahankan.
    """
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["conf"], reverse=True)
    final_detections = []

    while detections:
        best = detections.pop(0)
        final_detections.append(best)

        remaining = []

        for det in detections:
            overlap = box_iou(best, det)

            if overlap < iou_threshold:
                remaining.append(det)

        detections = remaining

    return sorted(final_detections, key=lambda d: d["conf"], reverse=True)


def remove_invalid_boxes(detections, min_area=60):
    clean = []

    for det in detections:
        w = det["x2"] - det["x1"]
        h = det["y2"] - det["y1"]

        if w <= 2 or h <= 2:
            continue

        if w * h < min_area:
            continue

        if det["conf"] < MIN_DRAW_CONF:
            continue

        clean.append(det)

    return clean


# ============================================================
# YOLO PREDICTION
# ============================================================

def predict_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]

    results = model.predict(
        source=frame_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONF_THRESHOLD,
        iou=YOLO_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        half=True if DEVICE == 0 else False,
        augment=USE_TTA,
        verbose=False
    )

    detections = []

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < CONF_THRESHOLD:
                continue

            det = detection_dict(
                x1,
                y1,
                x2,
                y2,
                conf,
                cls
            )

            det = clip_detection(det, w, h)
            detections.append(det)

    detections = remove_invalid_boxes(detections)

    detections = class_aware_nms(
        detections,
        iou_threshold=CLASS_AWARE_NMS_IOU
    )

    detections = cross_class_nms(
        detections,
        iou_threshold=CROSS_CLASS_NMS_IOU
    )

    return detections


# ============================================================
# DRAW FUNCTIONS
# ============================================================

def draw_label(frame, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = LABEL_FONT_SCALE
    thickness = LABEL_THICKNESS

    (tw, th), _ = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    img_h, img_w = frame.shape[:2]

    x = max(0, min(x, img_w - tw - 10))
    y = max(th + 8, min(y, img_h - 6))

    cv2.rectangle(
        frame,
        (x, y - th - 5),
        (x + tw + 7, y + 4),
        color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA
    )


def draw_summary(frame, with_count, without_count):
    total_object = with_count + without_count

    compliance = (
        round((with_count / total_object) * 100, 1)
        if total_object > 0
        else 0
    )

    summary = (
        f"Total: {total_object} | "
        f"With Helmet: {with_count} | "
        f"Without Helmet: {without_count} | "
        f"Compliance: {compliance}% | "
        f"Conf: {CONF_THRESHOLD} | "
        f"FPS: {display_fps_value:.1f} | "
        f"YOLO: {yolo_fps_value:.1f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(
        summary,
        font,
        SUMMARY_FONT_SCALE,
        SUMMARY_THICKNESS
    )

    bar_w = min(tw + 24, frame.shape[1] - 20)
    bar_h = th + 18

    cv2.rectangle(
        frame,
        (10, 10),
        (10 + bar_w, 10 + bar_h),
        (30, 45, 65),
        -1
    )

    cv2.putText(
        frame,
        summary,
        (20, 10 + th + 6),
        font,
        SUMMARY_FONT_SCALE,
        (255, 255, 255),
        SUMMARY_THICKNESS,
        lineType=cv2.LINE_AA
    )


def draw_detections(frame_bgr, detections):
    output = frame_bgr.copy()

    with_helmet_count = 0
    without_helmet_count = 0

    detections_sorted = sorted(
        detections,
        key=lambda d: box_area(d),
        reverse=True
    )

    for idx, det in enumerate(detections_sorted, start=1):
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]
        conf = det["conf"]
        cls = det["cls"]

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            continue

        label_name = get_class_name(cls)
        color = get_class_color(cls)
        kind = classify_count(cls)

        if kind == "with":
            with_helmet_count += 1
        elif kind == "without":
            without_helmet_count += 1

        label = f"{idx}. {label_name} {conf:.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS
        )

        label_y = y1 - 5

        if label_y < 15:
            label_y = y1 + 14

        draw_label(
            output,
            label,
            x1,
            label_y,
            color
        )

    draw_summary(
        output,
        with_helmet_count,
        without_helmet_count
    )

    return output


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_DISPLAY_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


# ============================================================
# THREAD CAMERA
# ============================================================

def camera_worker(cap):
    global latest_frame
    global latest_frame_id

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            time.sleep(0.003)
            continue

        if FIX_MIRROR:
            frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame
            latest_frame_id += 1


# ============================================================
# THREAD DETECTION
# ============================================================

def detection_worker():
    global latest_detections
    global latest_processed_frame_id
    global yolo_fps_value
    global inference_ms_value

    while not stop_event.is_set():
        frame_for_detection = None
        frame_id_for_detection = None

        with frame_lock:
            if latest_frame is not None and latest_frame_id != latest_processed_frame_id:
                frame_for_detection = latest_frame.copy()
                frame_id_for_detection = latest_frame_id

        if frame_for_detection is None:
            time.sleep(0.002)
            continue

        start_time = time.perf_counter()

        try:
            detections = predict_frame(frame_for_detection)
        except Exception as e:
            print("Error saat YOLO inference:", e)
            time.sleep(0.05)
            continue

        inference_time = time.perf_counter() - start_time

        yolo_fps = 1 / inference_time if inference_time > 0 else 0
        inference_ms = inference_time * 1000

        with detection_lock:
            latest_detections = detections
            yolo_fps_value = yolo_fps
            inference_ms_value = inference_ms
            latest_processed_frame_id = frame_id_for_detection


# ============================================================
# MAIN
# ============================================================

def main():
    global display_fps_value

    cap = open_camera()

    if cap is None:
        print("Kamera tidak dapat dibuka.")
        return

    print("Camera FPS terbaca:", cap.get(cv2.CAP_PROP_FPS))
    print("Camera width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Camera height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Confidence Threshold:", CONF_THRESHOLD)
    print("YOLO imgsz:", YOLO_IMGSZ)
    print("Tekan Q untuk keluar.")

    camera_thread = threading.Thread(
        target=camera_worker,
        args=(cap,),
        daemon=True
    )

    detector_thread = threading.Thread(
        target=detection_worker,
        daemon=True
    )

    camera_thread.start()
    detector_thread.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    next_frame_time = time.perf_counter()

    try:
        while not stop_event.is_set():
            frame_to_show = None

            with frame_lock:
                if latest_frame is not None:
                    frame_to_show = latest_frame.copy()

            if frame_to_show is None:
                time.sleep(0.003)
                continue

            with detection_lock:
                detections_to_draw = latest_detections.copy()

            now = time.perf_counter()

            raw_fps = 1 / (now - prev_time) if now != prev_time else 0
            prev_time = now

            display_fps_value = (display_fps_value * 0.9) + (raw_fps * 0.1)

            output_frame = draw_detections(
                frame_to_show,
                detections_to_draw
            )

            cv2.imshow(
                WINDOW_NAME,
                output_frame
            )

            next_frame_time += 1 / TARGET_DISPLAY_FPS
            sleep_time = next_frame_time - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.perf_counter()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                stop_event.set()
                break

    finally:
        stop_event.set()
        time.sleep(0.2)

        cap.release()
        cv2.destroyAllWindows()

        print("Program realtime detection dihentikan.")


if __name__ == "__main__":
    main()
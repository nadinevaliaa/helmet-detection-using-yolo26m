import cv2
import time
import threading
from collections import deque

from detection_engine import (
    model,
    CLASS_NAMES,
    clamp_conf,
    calculate_rates,
    rgb_to_bgr,
    bgr_to_rgb
)


# =====================================================
# REALTIME CONFIG
# =====================================================
REALTIME_IMGSZ = 192
REALTIME_DETECT_WIDTH = 320
REALTIME_MAX_DET = 10
REALTIME_IOU = 0.45

# Untuk CPU: 0.35 - 0.50 biasanya paling stabil.
DETECTION_INTERVAL_SECONDS = 0.40
CACHE_TTL_SECONDS = 1.60
SMOOTHING_ALPHA = 0.65

WITH_HELMET_CLASS_ID = 0
WITHOUT_HELMET_CLASS_ID = 1


# =====================================================
# GLOBAL CACHE
# =====================================================
YOLO_LOCK = threading.Lock()

LAST_DETECTIONS = []
LAST_HELMET_COUNT = 0
LAST_NO_HELMET_COUNT = 0

LAST_MODEL_FPS = 0.0
LAST_PROCESS_FPS = 0.0
LAST_LATENCY_MS = 0.0
LAST_PROCESS_DURATION = 0.0

LAST_DETECTION_TIME = 0.0
LAST_RETURN_TIME = 0.0
LAST_ERROR_MESSAGE = ""

FPS_HISTORY = deque(maxlen=30)


try:
    cv2.setNumThreads(1)
except Exception:
    pass


# =====================================================
# HELPER
# =====================================================
def resize_for_realtime(frame_bgr, target_width=320):
    h, w = frame_bgr.shape[:2]

    if w <= target_width:
        return frame_bgr.copy(), 1.0

    scale = target_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def smooth_detections(old_detections, new_detections):
    if not old_detections:
        return new_detections

    if not new_detections:
        return []

    smoothed = []
    used_old_indexes = set()

    for new_det in new_detections:
        best_iou = 0.0
        best_old_index = None
        best_old_det = None

        for old_index, old_det in enumerate(old_detections):
            if old_index in used_old_indexes:
                continue

            if old_det["cls"] != new_det["cls"]:
                continue

            iou = calculate_iou(old_det["box"], new_det["box"])

            if iou > best_iou:
                best_iou = iou
                best_old_index = old_index
                best_old_det = old_det

        if best_old_det is not None and best_iou >= 0.20:
            used_old_indexes.add(best_old_index)

            old_box = best_old_det["box"]
            new_box = new_det["box"]

            smooth_box = [
                int(old_box[0] * (1 - SMOOTHING_ALPHA) + new_box[0] * SMOOTHING_ALPHA),
                int(old_box[1] * (1 - SMOOTHING_ALPHA) + new_box[1] * SMOOTHING_ALPHA),
                int(old_box[2] * (1 - SMOOTHING_ALPHA) + new_box[2] * SMOOTHING_ALPHA),
                int(old_box[3] * (1 - SMOOTHING_ALPHA) + new_box[3] * SMOOTHING_ALPHA),
            ]

            smoothed.append({
                "box": smooth_box,
                "cls": new_det["cls"],
                "score": new_det["score"]
            })
        else:
            smoothed.append(new_det)

    return smoothed


# =====================================================
# YOLO INFERENCE
# =====================================================
def run_realtime_yolo(frame_bgr, conf_value):
    conf = clamp_conf(conf_value)

    original_h, original_w = frame_bgr.shape[:2]

    resized_bgr, scale = resize_for_realtime(
        frame_bgr,
        target_width=REALTIME_DETECT_WIDTH
    )

    model_start = time.perf_counter()

    results = model.predict(
        source=resized_bgr,
        conf=conf,
        imgsz=REALTIME_IMGSZ,
        iou=REALTIME_IOU,
        max_det=REALTIME_MAX_DET,
        verbose=False,
        stream=False
    )

    model_duration = time.perf_counter() - model_start
    model_fps = round(1 / model_duration, 2) if model_duration > 0 else 0.0
    latency_ms = round(model_duration * 1000, 1)

    detections = []
    helmet_count = 0
    no_helmet_count = 0

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])
            score = float(box.conf[0])

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            x1 = max(0, min(x1, original_w - 1))
            y1 = max(0, min(y1, original_h - 1))
            x2 = max(0, min(x2, original_w - 1))
            y2 = max(0, min(y2, original_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            if cls_id == WITH_HELMET_CLASS_ID:
                helmet_count += 1
            elif cls_id == WITHOUT_HELMET_CLASS_ID:
                no_helmet_count += 1

            detections.append({
                "box": [x1, y1, x2, y2],
                "cls": cls_id,
                "score": score
            })

    detections = sorted(
        detections,
        key=lambda item: item["score"],
        reverse=True
    )

    return detections, helmet_count, no_helmet_count, model_fps, latency_ms


# =====================================================
# DRAW
# =====================================================
def draw_cached_detections(frame_bgr, detections):
    output = frame_bgr.copy()

    frame_h, frame_w = output.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["cls"]
        score = det["score"]

        if cls_id == WITH_HELMET_CLASS_ID:
            color = (0, 200, 0)
        elif cls_id == WITHOUT_HELMET_CLASS_ID:
            color = (0, 0, 255)
        else:
            color = (255, 180, 0)

        label_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        label = f"{label_name} {score:.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
            lineType=cv2.LINE_AA
        )

        font_scale = 0.46
        thickness = 2

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        y_text = y1 - 8

        if y_text - th - 8 < 0:
            y_text = y1 + th + 10

        bg_x1 = x1
        bg_y1 = max(0, y_text - th - 8)
        bg_x2 = min(frame_w - 1, x1 + tw + 10)
        bg_y2 = min(frame_h - 1, y_text + 5)

        cv2.rectangle(
            output,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 5, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA
        )

    return output


# =====================================================
# CACHE
# =====================================================
def should_run_detection():
    now = time.perf_counter()
    return (now - LAST_DETECTION_TIME) >= DETECTION_INTERVAL_SECONDS


def cache_is_stale():
    now = time.perf_counter()
    return (now - LAST_DETECTION_TIME) > CACHE_TTL_SECONDS


def update_cache(detections, helmet_count, no_helmet_count, model_fps, latency_ms):
    global LAST_DETECTIONS
    global LAST_HELMET_COUNT
    global LAST_NO_HELMET_COUNT
    global LAST_MODEL_FPS
    global LAST_LATENCY_MS
    global LAST_DETECTION_TIME

    LAST_DETECTIONS = smooth_detections(
        LAST_DETECTIONS,
        detections
    )

    LAST_HELMET_COUNT = helmet_count
    LAST_NO_HELMET_COUNT = no_helmet_count
    LAST_MODEL_FPS = model_fps
    LAST_LATENCY_MS = latency_ms
    LAST_DETECTION_TIME = time.perf_counter()


def clear_cache_if_stale():
    global LAST_DETECTIONS
    global LAST_HELMET_COUNT
    global LAST_NO_HELMET_COUNT

    if cache_is_stale():
        LAST_DETECTIONS = []
        LAST_HELMET_COUNT = 0
        LAST_NO_HELMET_COUNT = 0


def make_metrics(process_duration, process_fps):
    fps_average = round(sum(FPS_HISTORY) / len(FPS_HISTORY), 2) if FPS_HISTORY else process_fps

    return {
        "duration_process": round(process_duration, 2),
        "fps_process": round(process_fps, 2),
        "fps_model": round(LAST_MODEL_FPS, 2),
        "latency_ms": round(LAST_LATENCY_MS, 1),
        "fps_average": round(fps_average, 2),
        "progress_percent": 100,
        "processed_frame": 0,
        "total_frame": 0,
        "status": "Live Recording"
    }


# =====================================================
# MAIN REALTIME CORE
# =====================================================
def detect_realtime_core(frame_rgb, conf_value):
    global LAST_RETURN_TIME
    global LAST_PROCESS_FPS
    global LAST_PROCESS_DURATION
    global LAST_ERROR_MESSAGE

    conf = clamp_conf(conf_value)

    if frame_rgb is None:
        return {
            "success": False,
            "image": None,
            "with_helmet": 0,
            "without_helmet": 0,
            "confidence": conf,
            "message": "Menunggu input kamera...",
            "metrics": make_metrics(0, 0)
        }

    process_start = time.perf_counter()

    try:
        frame_bgr = rgb_to_bgr(frame_rgb)

        if frame_bgr is None:
            return {
                "success": False,
                "image": frame_rgb,
                "with_helmet": LAST_HELMET_COUNT,
                "without_helmet": LAST_NO_HELMET_COUNT,
                "confidence": conf,
                "message": "Frame kamera gagal dibaca.",
                "metrics": make_metrics(0, 0)
            }

        clear_cache_if_stale()

        if should_run_detection():
            acquired = YOLO_LOCK.acquire(blocking=False)

            if acquired:
                try:
                    detections, helmet_count, no_helmet_count, model_fps, latency_ms = run_realtime_yolo(
                        frame_bgr,
                        conf
                    )

                    update_cache(
                        detections=detections,
                        helmet_count=helmet_count,
                        no_helmet_count=no_helmet_count,
                        model_fps=model_fps,
                        latency_ms=latency_ms
                    )

                    LAST_ERROR_MESSAGE = ""

                except Exception as e:
                    LAST_ERROR_MESSAGE = str(e)

                finally:
                    YOLO_LOCK.release()

        detected_bgr = draw_cached_detections(
            frame_bgr,
            LAST_DETECTIONS
        )

        detected_rgb = bgr_to_rgb(detected_bgr)

        process_duration = time.perf_counter() - process_start
        process_fps = round(1 / process_duration, 2) if process_duration > 0 else 0

        FPS_HISTORY.append(process_fps)

        LAST_PROCESS_DURATION = process_duration
        LAST_PROCESS_FPS = process_fps

        compliance_rate, violation_rate = calculate_rates(
            LAST_HELMET_COUNT,
            LAST_NO_HELMET_COUNT
        )

        metrics = make_metrics(process_duration, process_fps)

        message = (
            f"Confidence {conf:.2f} | "
            f"With Helmet {LAST_HELMET_COUNT} | "
            f"Without Helmet {LAST_NO_HELMET_COUNT} | "
            f"Compliance {compliance_rate}% | "
            f"Violation {violation_rate}% | "
            f"Duration {metrics['duration_process']}s | "
            f"FPS Process {metrics['fps_process']} | "
            f"FPS Model {metrics['fps_model']} | "
            f"Latency {metrics['latency_ms']} ms | "
            f"FPS Average {metrics['fps_average']}"
        )

        if LAST_ERROR_MESSAGE:
            message += f" | Last Error: {LAST_ERROR_MESSAGE[:60]}"

        return {
            "success": True,
            "image": detected_rgb,
            "with_helmet": LAST_HELMET_COUNT,
            "without_helmet": LAST_NO_HELMET_COUNT,
            "confidence": conf,
            "message": message,
            "metrics": metrics
        }

    except Exception as e:
        try:
            fallback_bgr = rgb_to_bgr(frame_rgb)
            fallback_rgb = bgr_to_rgb(fallback_bgr) if fallback_bgr is not None else frame_rgb
        except Exception:
            fallback_rgb = frame_rgb

        metrics = make_metrics(0, 0)

        return {
            "success": False,
            "image": fallback_rgb,
            "with_helmet": LAST_HELMET_COUNT,
            "without_helmet": LAST_NO_HELMET_COUNT,
            "confidence": conf,
            "message": f"Realtime error ditahan agar web tidak crash: {str(e)}",
            "metrics": metrics
        }
import cv2
import time

from detection_engine import (
    model,
    CLASS_NAMES,
    clamp_conf,
    calculate_rates,
    rgb_to_bgr,
    bgr_to_rgb
)


# =====================================================
# IMAGE DETECTION CONFIG
# =====================================================
# Dibuat lebih akurat untuk objek kecil/jauh pada gambar jalan raya.
IMAGE_MAX_WIDTH = 1280
IMAGE_IMGSZ = 768
IMAGE_IOU = 0.50
IMAGE_MAX_DET = 160

# Confidence dibuat lebih rendah khusus image agar objek kecil ikut terbaca.
# Slider tetap dipakai, tapi jika terlalu tinggi akan diturunkan otomatis
# supaya deteksi gambar tidak kehilangan objek kecil.
IMAGE_CONF_CAP = 0.22

# Multi-crop membantu objek jauh/kecil lebih mudah terbaca.
USE_TILED_INFERENCE = True

# Crop ratio: semakin kecil, objek makin besar saat inference,
# tapi proses makin lama. Nilai ini masih aman untuk CPU.
TILE_W_RATIO = 0.58
TILE_H_RATIO = 0.72

# Filter box terlalu kecil/noise.
MIN_BOX_WIDTH = 4
MIN_BOX_HEIGHT = 4

# NMS tambahan agar hasil tidak dobel terlalu banyak.
NMS_IOU_THRESHOLD = 0.42

WITH_HELMET_CLASS_ID = 0
WITHOUT_HELMET_CLASS_ID = 1


# =====================================================
# BASIC HELPER
# =====================================================
def resize_image_for_detection(frame_bgr, max_width=IMAGE_MAX_WIDTH):
    if frame_bgr is None:
        return None, 1.0

    h, w = frame_bgr.shape[:2]

    if w <= max_width:
        return frame_bgr.copy(), 1.0

    scale = max_width / w
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        frame_bgr,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized, scale


def detection_iou(det_a, det_b):
    ax1, ay1, ax2, ay2 = det_a["box"]
    bx1, by1, bx2, by2 = det_b["box"]

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
        return 0

    return inter_area / union_area


def class_agnostic_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    NMS dibuat class-agnostic agar satu objek tidak dobel
    menjadi With Helmet dan Without Helmet sekaligus.
    """
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)

        remaining = []

        for det in detections:
            iou_value = detection_iou(best, det)

            if iou_value < iou_threshold:
                remaining.append(det)

        detections = remaining

    return kept


def build_image_crops(frame_bgr):
    """
    Membuat crop gambar agar objek kecil di kejauhan ikut terbaca.
    Area full frame tetap diproses, lalu ditambah crop kiri/tengah/kanan.
    """
    h, w = frame_bgr.shape[:2]

    crops = [
        {
            "crop": frame_bgr,
            "x_offset": 0,
            "y_offset": 0,
            "name": "full"
        }
    ]

    if not USE_TILED_INFERENCE:
        return crops

    if w < 640 or h < 420:
        return crops

    crop_w = int(w * TILE_W_RATIO)
    crop_h = int(h * TILE_H_RATIO)

    crop_w = max(360, min(crop_w, w))
    crop_h = max(320, min(crop_h, h))

    x_positions = [
        0,
        max(0, (w - crop_w) // 2),
        max(0, w - crop_w)
    ]

    y_positions = [
        0,
        max(0, h - crop_h)
    ]

    unique_positions = []

    for y in y_positions:
        for x in x_positions:
            position = (x, y)

            if position not in unique_positions:
                unique_positions.append(position)

    for x, y in unique_positions:
        crop = frame_bgr[y:y + crop_h, x:x + crop_w]

        if crop is None or crop.size == 0:
            continue

        crops.append({
            "crop": crop,
            "x_offset": x,
            "y_offset": y,
            "name": "tile"
        })

    return crops


def run_yolo_on_crop(crop_bgr, conf_value, x_offset, y_offset, frame_w, frame_h):
    detections = []

    results = model.predict(
        source=crop_bgr,
        conf=conf_value,
        imgsz=IMAGE_IMGSZ,
        iou=IMAGE_IOU,
        max_det=IMAGE_MAX_DET,
        augment=False,
        verbose=False
    )

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])
            score = float(box.conf[0])

            if cls_id not in [WITH_HELMET_CLASS_ID, WITHOUT_HELMET_CLASS_ID]:
                continue

            x1 = int(x1 + x_offset)
            y1 = int(y1 + y_offset)
            x2 = int(x2 + x_offset)
            y2 = int(y2 + y_offset)

            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < MIN_BOX_WIDTH or box_h < MIN_BOX_HEIGHT:
                continue

            detections.append({
                "box": [x1, y1, x2, y2],
                "cls": cls_id,
                "score": score
            })

    return detections


def run_image_inference(frame_bgr, conf_value):
    frame_h, frame_w = frame_bgr.shape[:2]

    all_detections = []
    crops = build_image_crops(frame_bgr)

    inference_start = time.perf_counter()

    for crop_item in crops:
        crop_detections = run_yolo_on_crop(
            crop_bgr=crop_item["crop"],
            conf_value=conf_value,
            x_offset=crop_item["x_offset"],
            y_offset=crop_item["y_offset"],
            frame_w=frame_w,
            frame_h=frame_h
        )

        all_detections.extend(crop_detections)

    inference_time = time.perf_counter() - inference_start

    final_detections = class_agnostic_nms(
        all_detections,
        iou_threshold=NMS_IOU_THRESHOLD
    )

    return final_detections, inference_time


def draw_image_detection(frame_bgr, detections):
    output = frame_bgr.copy()

    helmet_count = 0
    no_helmet_count = 0

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_id = det["cls"]
        score = det["score"]

        if cls_id == WITH_HELMET_CLASS_ID:
            color = (0, 190, 0)
            label_name = "With Helmet"
            helmet_count += 1
        elif cls_id == WITHOUT_HELMET_CLASS_ID:
            color = (0, 0, 255)
            label_name = "Without Helmet"
            no_helmet_count += 1
        else:
            color = (255, 180, 0)
            label_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"

        label = f"{label_name} {score:.2f}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2
        )

        y_text = max(y1 - 8, th + 10)

        cv2.rectangle(
            output,
            (x1, y_text - th - 8),
            (x1 + tw + 10, y_text + 6),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 5, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

    return output, helmet_count, no_helmet_count


# =====================================================
# MAIN IMAGE CORE
# =====================================================
def detect_image_core(image_rgb, conf_value):
    start_time = time.perf_counter()

    user_conf = clamp_conf(conf_value)

    # Untuk image, confidence terlalu tinggi bikin objek kecil tidak kebaca.
    effective_conf = min(user_conf, IMAGE_CONF_CAP)

    if image_rgb is None:
        return {
            "success": False,
            "image": None,
            "with_helmet": 0,
            "without_helmet": 0,
            "confidence": effective_conf,
            "message": "Silakan upload gambar terlebih dahulu.",
            "metrics": {
                "duration_process": 0,
                "fps_process": 0,
                "fps_model": 0,
                "latency_ms": 0,
                "fps_average": 0,
                "progress_percent": 0,
                "processed_frame": 0,
                "total_frame": 1,
                "status": "Waiting Image"
            }
        }

    frame_bgr = rgb_to_bgr(image_rgb)

    if frame_bgr is None:
        return {
            "success": False,
            "image": None,
            "with_helmet": 0,
            "without_helmet": 0,
            "confidence": effective_conf,
            "message": "Gambar gagal dibaca.",
            "metrics": {
                "duration_process": 0,
                "fps_process": 0,
                "fps_model": 0,
                "latency_ms": 0,
                "fps_average": 0,
                "progress_percent": 0,
                "processed_frame": 0,
                "total_frame": 1,
                "status": "Failed"
            }
        }

    resized_bgr, scale = resize_image_for_detection(
        frame_bgr,
        max_width=IMAGE_MAX_WIDTH
    )

    detections, inference_time = run_image_inference(
        resized_bgr,
        effective_conf
    )

    detected_bgr, helmet_count, no_helmet_count = draw_image_detection(
        resized_bgr,
        detections
    )

    detected_rgb = bgr_to_rgb(detected_bgr)

    total_time = time.perf_counter() - start_time

    fps_process = round(1 / total_time, 2) if total_time > 0 else 0
    fps_model = round(1 / inference_time, 2) if inference_time > 0 else 0
    latency_ms = round(inference_time * 1000, 1)

    compliance_rate, violation_rate = calculate_rates(
        helmet_count,
        no_helmet_count
    )

    message = (
        f"Confidence: {effective_conf:.2f} | "
        f"With Helmet: {helmet_count} | "
        f"Without Helmet: {no_helmet_count} | "
        f"Compliance: {compliance_rate}% | "
        f"Violation: {violation_rate}% | "
        f"Latency: {latency_ms} ms"
    )

    return {
        "success": True,
        "image": detected_rgb,
        "with_helmet": helmet_count,
        "without_helmet": no_helmet_count,
        "confidence": effective_conf,
        "message": message,
        "metrics": {
            "duration_process": round(total_time, 2),
            "fps_process": fps_process,
            "fps_model": fps_model,
            "latency_ms": latency_ms,
            "fps_average": fps_process,
            "progress_percent": 100,
            "processed_frame": 1,
            "total_frame": 1,
            "status": "Completed"
        }
    }
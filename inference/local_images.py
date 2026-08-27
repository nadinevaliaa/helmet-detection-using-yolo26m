import cv2
import os
import time
import torch
from ultralytics import YOLO

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "Weights/best.pt"
IMAGE_PATH = "Media_Images/p1.jpeg"

OUTPUT_DIR = "outputs"
OUTPUT_NAME = "image_detection_high_accuracy_result.jpg"

# Untuk image detection, kita fokus akurasi.
FULL_IMAGE_IMGSZ = 1280

# Deteksi full image dengan beberapa pembesaran.
USE_MULTI_SCALE_FULL_IMAGE = True
FULL_IMAGE_SCALES = [1.0, 1.25, 1.5, 2.0]

# Tiled detection untuk objek kecil.
USE_TILED_DETECTION = True

# Multi tile size:
# 384 lebih detail untuk objek kecil.
# 512 seimbang.
# 640 menangkap konteks lebih luas.
TILE_SIZES = [384, 512, 640]

# Overlap besar membantu objek yang berada di batas tile.
TILE_OVERLAP = 0.45

# Input size untuk tile.
TILE_IMGSZ = 640

# Tile juga diproses dengan skala tambahan.
USE_MULTI_SCALE_TILE = True
TILE_SCALES = [1.0, 1.35]

# Confidence dibuat sensitif agar objek kecil tidak hilang.
# Kalau terlalu banyak salah deteksi, naikkan ke 0.25 atau 0.30.
CONF_THRESHOLD = 0.20
MIN_DRAW_CONF = 0.20

# NMS setting.
YOLO_IOU = 0.45
CLASS_AWARE_NMS_IOU = 0.45
CROSS_CLASS_NMS_IOU = 0.60

MAX_DET = 1500

# TTA bisa lebih akurat, tetapi sangat lambat.
# Coba True kalau ingin mode super teliti.
USE_TTA = False

SHOW_RESULT = True
SAVE_RESULT = True


# =====================================================
# LABEL / FONT SETTING
# =====================================================

# Font kecil, tetapi teks tetap lengkap.
LABEL_FONT_SCALE = 0.30
LABEL_THICKNESS = 1
BOX_THICKNESS = 2

SUMMARY_FONT_SCALE = 0.50
SUMMARY_THICKNESS = 1

# Semua label tetap lengkap.
# Contoh: "1. With Helmet 0.87"
FORCE_FULL_LABEL = True


# =====================================================
# OPTIMASI CPU / GPU
# =====================================================

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


# =====================================================
# LOAD MODEL
# =====================================================

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


# =====================================================
# HELPER CLASS
# =====================================================

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

    # Without dicek dulu karena "without helmet" juga mengandung kata "helmet".
    if "without" in name or "no helmet" in name or "no_helmet" in name:
        return "without"

    if "with" in name or "helmet" in name:
        return "with"

    return "other"


# =====================================================
# DETECTION DATA STRUCTURE
# =====================================================

def detection_dict(x1, y1, x2, y2, conf, cls, source=""):
    return {
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
        "conf": float(conf),
        "cls": int(cls),
        "source": source
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


# =====================================================
# NMS FUNCTIONS
# =====================================================

def class_aware_nms(detections, iou_threshold=0.45):
    """
    NMS per class.
    With Helmet dan Without Helmet tidak langsung saling menghapus.
    """
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
    NMS tambahan antar class.
    Berguna jika satu kepala terdeteksi ganda sebagai With Helmet dan Without Helmet.
    Deteksi dengan confidence tertinggi dipertahankan.
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


# =====================================================
# YOLO PREDICTION
# =====================================================

def predict_on_image(frame_bgr, imgsz, source_name=""):
    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
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

            detections.append(
                detection_dict(
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    cls,
                    source=source_name
                )
            )

    return detections


def predict_scaled_full_image(image_bgr):
    """
    Deteksi full image dengan beberapa skala.
    Koordinat dikembalikan ke ukuran gambar asli.
    """
    h, w = image_bgr.shape[:2]

    scales = FULL_IMAGE_SCALES if USE_MULTI_SCALE_FULL_IMAGE else [1.0]
    all_detections = []

    for scale in scales:
        print(f"Full image scale: {scale}")

        if scale == 1.0:
            scaled = image_bgr
        else:
            new_w = int(w * scale)
            new_h = int(h * scale)

            scaled = cv2.resize(
                image_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_LINEAR
            )

        dets = predict_on_image(
            scaled,
            imgsz=FULL_IMAGE_IMGSZ,
            source_name=f"full_scale_{scale}"
        )

        for det in dets:
            if scale != 1.0:
                det["x1"] /= scale
                det["x2"] /= scale
                det["y1"] /= scale
                det["y2"] /= scale

                det["x1"] = int(round(det["x1"]))
                det["x2"] = int(round(det["x2"]))
                det["y1"] = int(round(det["y1"]))
                det["y2"] = int(round(det["y2"]))

            det = clip_detection(det, w, h)
            all_detections.append(det)

    return all_detections


def generate_tiles(image_bgr, tile_size=512, overlap=0.45):
    h, w = image_bgr.shape[:2]

    if w <= tile_size and h <= tile_size:
        yield 0, 0, image_bgr
        return

    stride = int(tile_size * (1 - overlap))
    stride = max(1, stride)

    x_positions = list(range(0, max(w - tile_size, 0) + 1, stride))
    y_positions = list(range(0, max(h - tile_size, 0) + 1, stride))

    if not x_positions or x_positions[-1] != max(w - tile_size, 0):
        x_positions.append(max(w - tile_size, 0))

    if not y_positions or y_positions[-1] != max(h - tile_size, 0):
        y_positions.append(max(h - tile_size, 0))

    for y in y_positions:
        for x in x_positions:
            tile = image_bgr[
                y:min(y + tile_size, h),
                x:min(x + tile_size, w)
            ]

            yield x, y, tile


def predict_tiles(image_bgr):
    h, w = image_bgr.shape[:2]

    all_detections = []
    total_tile_count = 0

    tile_scales = TILE_SCALES if USE_MULTI_SCALE_TILE else [1.0]

    for tile_size in TILE_SIZES:
        print(f"Memproses tile size: {tile_size}")

        tile_count = 0

        for x_offset, y_offset, tile in generate_tiles(
            image_bgr,
            tile_size=tile_size,
            overlap=TILE_OVERLAP
        ):
            tile_count += 1
            total_tile_count += 1

            tile_h, tile_w = tile.shape[:2]

            for scale in tile_scales:
                if scale == 1.0:
                    tile_for_pred = tile
                else:
                    new_w = int(tile_w * scale)
                    new_h = int(tile_h * scale)

                    tile_for_pred = cv2.resize(
                        tile,
                        (new_w, new_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                tile_dets = predict_on_image(
                    tile_for_pred,
                    imgsz=TILE_IMGSZ,
                    source_name=f"tile{tile_size}_{tile_count}_scale_{scale}"
                )

                for det in tile_dets:
                    if scale != 1.0:
                        det["x1"] /= scale
                        det["x2"] /= scale
                        det["y1"] /= scale
                        det["y2"] /= scale

                    det["x1"] = int(round(det["x1"] + x_offset))
                    det["x2"] = int(round(det["x2"] + x_offset))
                    det["y1"] = int(round(det["y1"] + y_offset))
                    det["y2"] = int(round(det["y2"] + y_offset))

                    det = clip_detection(det, w, h)
                    all_detections.append(det)

        print(f"Jumlah tile untuk size {tile_size}: {tile_count}")

    print("Total seluruh tile diproses:", total_tile_count)

    return all_detections


def predict_high_accuracy(image_bgr):
    all_detections = []

    print("Memproses full image multi-scale...")
    full_detections = predict_scaled_full_image(image_bgr)
    all_detections.extend(full_detections)

    if USE_TILED_DETECTION:
        print("Memproses tiled detection multi-size multi-scale...")
        tile_detections = predict_tiles(image_bgr)
        all_detections.extend(tile_detections)

    print("Total kandidat sebelum filter:", len(all_detections))

    all_detections = remove_invalid_boxes(all_detections)

    print("Total kandidat setelah filter:", len(all_detections))

    final_detections = class_aware_nms(
        all_detections,
        iou_threshold=CLASS_AWARE_NMS_IOU
    )

    print("Total setelah class-aware NMS:", len(final_detections))

    final_detections = cross_class_nms(
        final_detections,
        iou_threshold=CROSS_CLASS_NMS_IOU
    )

    print("Total setelah cross-class NMS:", len(final_detections))

    return final_detections


# =====================================================
# DRAW RESULT
# =====================================================

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


def build_label(idx, label_name, conf):
    # Label tetap lengkap, hanya font yang dikecilkan.
    return f"{idx}. {label_name} {conf:.2f}"


def draw_detections(frame_bgr, detections):
    output = frame_bgr.copy()

    with_helmet_count = 0
    without_helmet_count = 0
    other_count = 0

    # Box besar digambar dulu, box kecil belakangan supaya tetap kelihatan.
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
        else:
            other_count += 1

        label = build_label(idx, label_name, conf)

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

    total_object = with_helmet_count + without_helmet_count + other_count

    compliance = (
        round((with_helmet_count / total_object) * 100, 1)
        if total_object > 0
        else 0
    )

    summary = (
        f"Total: {total_object} | "
        f"With Helmet: {with_helmet_count} | "
        f"Without Helmet: {without_helmet_count} | "
        f"Compliance: {compliance}% | "
        f"Conf: {CONF_THRESHOLD}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), _ = cv2.getTextSize(
        summary,
        font,
        SUMMARY_FONT_SCALE,
        SUMMARY_THICKNESS
    )

    bar_w = min(tw + 24, output.shape[1] - 20)
    bar_h = th + 18

    cv2.rectangle(
        output,
        (10, 10),
        (10 + bar_w, 10 + bar_h),
        (30, 45, 65),
        -1
    )

    cv2.putText(
        output,
        summary,
        (20, 10 + th + 6),
        font,
        SUMMARY_FONT_SCALE,
        (255, 255, 255),
        SUMMARY_THICKNESS,
        lineType=cv2.LINE_AA
    )

    return output, with_helmet_count, without_helmet_count, other_count


def resize_for_display(image_bgr, max_width=1280, max_height=720):
    h, w = image_bgr.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(
            image_bgr,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    return image_bgr


# =====================================================
# MAIN
# =====================================================

def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {IMAGE_PATH}"
        )

    image_bgr = cv2.imread(IMAGE_PATH)

    if image_bgr is None:
        raise ValueError(
            f"Gambar gagal dibaca: {IMAGE_PATH}"
        )

    print("Image path:", IMAGE_PATH)
    print("Image size:", image_bgr.shape[1], "x", image_bgr.shape[0])
    print("Confidence Threshold:", CONF_THRESHOLD)
    print("Full Image imgsz:", FULL_IMAGE_IMGSZ)
    print("Full Image Scales:", FULL_IMAGE_SCALES)
    print("Use Tiled Detection:", USE_TILED_DETECTION)
    print("Tile Sizes:", TILE_SIZES)
    print("Tile Overlap:", TILE_OVERLAP)
    print("Tile Scales:", TILE_SCALES)

    start_time = time.perf_counter()

    detections = predict_high_accuracy(image_bgr)

    elapsed = time.perf_counter() - start_time

    output_bgr, with_count, without_count, other_count = draw_detections(
        image_bgr,
        detections
    )

    total = with_count + without_count + other_count

    print("Detection selesai.")
    print("Total Object:", total)
    print("With Helmet:", with_count)
    print("Without Helmet:", without_count)
    print("Other:", other_count)
    print("Waktu proses:", round(elapsed, 3), "detik")

    print("\nDetail detections:")

    for i, det in enumerate(detections, start=1):
        print(
            f"{i}. {get_class_name(det['cls'])} | "
            f"conf={det['conf']:.2f} | "
            f"box=({det['x1']},{det['y1']},{det['x2']},{det['y2']}) | "
            f"source={det.get('source', '')}"
        )

    if SAVE_RESULT:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(
            OUTPUT_DIR,
            OUTPUT_NAME
        )

        cv2.imwrite(output_path, output_bgr)

        print("Output disimpan:", output_path)

    if SHOW_RESULT:
        display_img = resize_for_display(output_bgr)

        cv2.imshow(
            "Image Detection - High Accuracy",
            display_img
        )

        print("Tekan Q untuk keluar.")

        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
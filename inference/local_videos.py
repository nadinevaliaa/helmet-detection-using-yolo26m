import cv2
import os
import time
import torch
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "Media_Video/v3.mp4"
MODEL_PATH = "Weights/best.pt"

OUTPUT_DIR = "outputs"
OUTPUT_NAME = "video_detection_normal_box_count.mp4"

CONF_THRESHOLD = 0.40
MIN_DRAW_CONF = 0.40

# Untuk video normal:
# 416 = ringan
# 512 = lebih akurat
# 640 = lebih berat
YOLO_IMGSZ = 416

# Deteksi setiap N frame.
# 1 = semua frame
# 2 = lebih ringan
# 3 = lebih ringan lagi
DETECT_EVERY_N_FRAMES = 2

YOLO_IOU = 0.45
MAX_DET = 300

SHOW_PREVIEW = True
SAVE_OUTPUT = True

WINDOW_NAME = "Helmet Detection - Normal Video"


# ============================================================
# LABEL SETTING
# ============================================================

LABEL_FONT_SCALE = 0.35
LABEL_THICKNESS = 1
BOX_THICKNESS = 2

SUMMARY_FONT_SCALE = 0.48
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
# DETECTION
# ============================================================

def detect_frame(frame_bgr):
    results = model.predict(
        source=frame_bgr,
        imgsz=YOLO_IMGSZ,
        conf=CONF_THRESHOLD,
        iou=YOLO_IOU,
        max_det=MAX_DET,
        device=DEVICE,
        half=True if DEVICE == 0 else False,
        verbose=False
    )

    detections = []

    h, w = frame_bgr.shape[:2]

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf < MIN_DRAW_CONF:
                continue

            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "conf": conf,
                "cls": cls
            })

    return detections


# ============================================================
# DRAWING
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


def count_current_boxes(detections):
    with_count = 0
    without_count = 0

    for det in detections:
        kind = classify_count(det["cls"])

        if kind == "with":
            with_count += 1
        elif kind == "without":
            without_count += 1

    total = with_count + without_count

    return total, with_count, without_count


def draw_summary(
    frame,
    frame_idx,
    total_frames,
    total_box,
    with_count,
    without_count,
    process_fps
):
    compliance = (
        round((with_count / total_box) * 100, 1)
        if total_box > 0
        else 0
    )

    summary_1 = (
        f"Frame: {frame_idx}/{total_frames} | "
        f"Total Bounding Box: {total_box} | "
        f"With Helmet: {with_count} | "
        f"Without Helmet: {without_count}"
    )

    summary_2 = (
        f"Compliance: {compliance}% | "
        f"Conf: {CONF_THRESHOLD} | "
        f"YOLO imgsz: {YOLO_IMGSZ} | "
        f"Detect every: {DETECT_EVERY_N_FRAMES} frame | "
        f"Process FPS: {process_fps:.1f}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw1, th1), _ = cv2.getTextSize(
        summary_1,
        font,
        SUMMARY_FONT_SCALE,
        SUMMARY_THICKNESS
    )

    (tw2, th2), _ = cv2.getTextSize(
        summary_2,
        font,
        SUMMARY_FONT_SCALE,
        SUMMARY_THICKNESS
    )

    bar_w = min(max(tw1, tw2) + 24, frame.shape[1] - 20)
    bar_h = th1 + th2 + 30

    cv2.rectangle(
        frame,
        (10, 10),
        (10 + bar_w, 10 + bar_h),
        (30, 45, 65),
        -1
    )

    cv2.putText(
        frame,
        summary_1,
        (20, 10 + th1 + 6),
        font,
        SUMMARY_FONT_SCALE,
        (255, 255, 255),
        SUMMARY_THICKNESS,
        lineType=cv2.LINE_AA
    )

    cv2.putText(
        frame,
        summary_2,
        (20, 10 + th1 + th2 + 18),
        font,
        SUMMARY_FONT_SCALE,
        (255, 255, 255),
        SUMMARY_THICKNESS,
        lineType=cv2.LINE_AA
    )


def draw_detections(
    frame_bgr,
    detections,
    frame_idx,
    total_frames,
    process_fps
):
    output = frame_bgr.copy()

    # Hitung sesuai bounding box yang sedang ditampilkan
    total_box, with_count, without_count = count_current_boxes(detections)

    for idx, det in enumerate(detections, start=1):
        x1 = det["x1"]
        y1 = det["y1"]
        x2 = det["x2"]
        y2 = det["y2"]
        conf = det["conf"]
        cls = det["cls"]

        label_name = get_class_name(cls)
        color = get_class_color(cls)

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
        frame_idx,
        total_frames,
        total_box,
        with_count,
        without_count,
        process_fps
    )

    return output, total_box, with_count, without_count


# ============================================================
# VIDEO HELPER
# ============================================================

def make_even(value):
    value = int(value)
    return value if value % 2 == 0 else value - 1


def resize_preview(frame, max_width=1280, max_height=720):
    h, w = frame.shape[:2]

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    if scale < 1.0:
        return cv2.resize(
            frame,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video tidak ditemukan: {VIDEO_PATH}"
        )

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise ValueError(
            f"Video gagal dibuka: {VIDEO_PATH}"
        )

    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps is None or source_fps <= 0:
        source_fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_width = make_even(width)
    output_height = make_even(height)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME
    )

    writer = None

    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            source_fps,
            (output_width, output_height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Gagal membuat output video.")

    print("Video path:", VIDEO_PATH)
    print("Output path:", output_path)
    print("Source FPS:", source_fps)
    print("Output FPS:", source_fps)
    print("Resolution:", output_width, "x", output_height)
    print("Total frames:", total_frames)
    print("YOLO imgsz:", YOLO_IMGSZ)
    print("Detect every N frames:", DETECT_EVERY_N_FRAMES)
    print("Confidence:", CONF_THRESHOLD)

    frame_idx = 0
    last_detections = []
    process_fps = 0.0

    start_total = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_idx += 1

        if frame_bgr.shape[1] != output_width or frame_bgr.shape[0] != output_height:
            frame_bgr = cv2.resize(
                frame_bgr,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA
            )

        should_detect = (
            frame_idx == 1
            or frame_idx % DETECT_EVERY_N_FRAMES == 0
        )

        if should_detect:
            start_infer = time.perf_counter()

            last_detections = detect_frame(frame_bgr)

            elapsed_infer = time.perf_counter() - start_infer
            process_fps = 1 / elapsed_infer if elapsed_infer > 0 else 0

        output_frame, total_box, with_count, without_count = draw_detections(
            frame_bgr,
            last_detections,
            frame_idx,
            total_frames,
            process_fps
        )

        if SAVE_OUTPUT and writer is not None:
            writer.write(output_frame)

        if SHOW_PREVIEW:
            preview = resize_preview(output_frame)

            cv2.imshow(
                WINDOW_NAME,
                preview
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Proses dihentikan oleh user.")
                break

        if frame_idx % 100 == 0:
            elapsed_total = time.perf_counter() - start_total
            avg_loop_fps = frame_idx / elapsed_total if elapsed_total > 0 else 0

            print(
                f"Frame {frame_idx}/{total_frames} | "
                f"Total Box: {total_box} | "
                f"With: {with_count} | "
                f"Without: {without_count} | "
                f"Avg loop FPS: {avg_loop_fps:.2f} | "
                f"YOLO FPS: {process_fps:.2f}"
            )

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_total
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\nVideo detection selesai.")
    print("Total frame:", frame_idx)
    print("Total waktu proses:", round(total_time, 2), "detik")
    print("Rata-rata proses:", round(avg_fps, 2), "FPS")

    if SAVE_OUTPUT:
        print("Output disimpan di:", output_path)


if __name__ == "__main__":
    main()
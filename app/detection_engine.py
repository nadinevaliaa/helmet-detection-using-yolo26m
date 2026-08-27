import cv2
import os
from ultralytics import YOLO


# =====================================================
# GLOBAL CONFIG
# =====================================================
MODEL_PATH = "Weights/best.pt"
CLASS_NAMES = ["With Helmet", "Without Helmet"]

CONF_MIN = 0.00
CONF_MAX = 1.00
CONF_DEFAULT = 0.30


# =====================================================
# LOAD MODEL SEKALI SAJA
# =====================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}\n"
        f"Pastikan file best.pt berada di folder Weights/best.pt"
    )

model = YOLO(MODEL_PATH)


# =====================================================
# HELPER
# =====================================================
def clamp_conf(conf):
    try:
        conf = float(conf)
    except Exception:
        conf = CONF_DEFAULT

    if conf < CONF_MIN:
        conf = CONF_MIN

    if conf > CONF_MAX:
        conf = CONF_MAX

    return conf


def rgb_to_bgr(image_rgb):
    if image_rgb is None:
        return None

    if len(image_rgb.shape) == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    if image_rgb.shape[2] == 4:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image_bgr):
    if image_bgr is None:
        return None

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def calculate_rates(helmet_count, no_helmet_count):
    total = helmet_count + no_helmet_count

    if total <= 0:
        return 0, 0

    compliance_rate = round((helmet_count / total) * 100, 1)
    violation_rate = round((no_helmet_count / total) * 100, 1)

    return compliance_rate, violation_rate


# =====================================================
# CORE DETECTION
# =====================================================
def draw_detection(frame_bgr, conf_thres):
    if frame_bgr is None:
        return None, 0, 0

    conf_thres = clamp_conf(conf_thres)

    results = model(
        frame_bgr,
        conf=conf_thres,
        imgsz=640,
        verbose=False
    )

    helmet_count = 0
    no_helmet_count = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            score = float(box.conf[0])

            if cls == 0:
                color = (0, 200, 0)      # With Helmet = hijau
                helmet_count += 1
            else:
                color = (0, 0, 255)      # Without Helmet = merah
                no_helmet_count += 1

            label_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"Class {cls}"
            label = f"{label_name} {score:.2f}"

            cv2.rectangle(
                frame_bgr,
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

            y_text = max(y1 - 10, th + 10)

            cv2.rectangle(
                frame_bgr,
                (x1, y_text - th - 8),
                (x1 + tw + 12, y_text + 6),
                color,
                -1
            )

            cv2.putText(
                frame_bgr,
                label,
                (x1 + 6, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    return frame_bgr, helmet_count, no_helmet_count


def detect_rgb_frame(image_rgb, conf_value):
    """
    Input  : RGB image dari Gradio
    Output : dict berisi image RGB hasil deteksi + jumlah object
    """
    conf = clamp_conf(conf_value)

    if image_rgb is None:
        return {
            "success": False,
            "image": None,
            "with_helmet": 0,
            "without_helmet": 0,
            "confidence": conf,
            "message": "Input gambar kosong."
        }

    frame_bgr = rgb_to_bgr(image_rgb)

    detected_bgr, helmet_count, no_helmet_count = draw_detection(
        frame_bgr,
        conf
    )

    detected_rgb = bgr_to_rgb(detected_bgr)

    return {
        "success": True,
        "image": detected_rgb,
        "with_helmet": helmet_count,
        "without_helmet": no_helmet_count,
        "confidence": conf,
        "message": "Deteksi berhasil."
    }
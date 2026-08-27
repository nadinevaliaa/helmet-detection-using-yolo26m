import cv2
import os
import sys
import time
import html
import json
import base64
import atexit
import shutil
import subprocess
from urllib.parse import quote

import gradio as gr

from detection_engine import (
    model,
    CLASS_NAMES,
    calculate_rates,
    clamp_conf,
    CONF_MIN,
    CONF_MAX,
    CONF_DEFAULT
)

from images_detection import detect_image_core
from realtime_detection import detect_realtime_core


# =====================================================
# CONFIG
# =====================================================
APP_TITLE = "Helmet Detection Dashboard"
REALTIME_MONITOR_UPDATE_INTERVAL = 0.75

OUTPUT_DIR = os.path.abspath("outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTORY_PATH = os.path.join(OUTPUT_DIR, "detection_history_session.json")

WITH_HELMET_CLASS_ID = 0
WITHOUT_HELMET_CLASS_ID = 1

# =====================================================
# VIDEO DETECTION BALANCED CONFIG
# =====================================================
# Catatan: jangan dibuat terlalu tinggi seperti 960 + tiled inference karena CPU akan sangat lambat.
# Setting ini menjaga alur tetap sama, hasil video tetap smooth, dan deteksi lebih stabil.
VIDEO_DETECT_WIDTH = 640
VIDEO_IMGSZ = 640
VIDEO_IOU = 0.50
VIDEO_MAX_DET = 70

# Final video logic:
# - Video result dibuat sedikit turun FPS supaya bounding box tidak terlihat terlambat.
# - Frame yang ditulis ke output selalu frame yang benar-benar sudah dibaca YOLO.
# - Jadi tidak ada lagi frame kosong/stale yang membuat box terlihat telat mengikuti object.
VIDEO_FRAME_STRIDE = 1
VIDEO_DETECT_EVERY_N_FRAMES = 1
VIDEO_PREVIEW_EVERY_N_FRAMES = 15
VIDEO_OUTPUT_MAX_WIDTH = 960
VIDEO_RESULT_MIN_FPS = 10
VIDEO_RESULT_MAX_FPS = 18

# Filter agar bounding box tidak terlalu ramai.
MIN_DET_BOX_WIDTH = 6
MIN_DET_BOX_HEIGHT = 6
MIN_DET_BOX_AREA_RATIO = 0.00006
MAX_DET_BOX_AREA_RATIO = 0.08000
VIDEO_NMS_IOU = 0.42
CROSS_CLASS_NMS_IOU = 0.62

# Unique object tracking.
TRACK_IOU_THRESHOLD = 0.16
TRACK_MAX_CENTER_DISTANCE = 105
TRACK_MAX_LOST_FRAMES = 42
TRACK_DRAW_MAX_LOST_FRAMES = 8
TRACK_MIN_HITS_TO_CONFIRM = 2
TRACK_MIN_SCORE_TO_CONFIRM = 0.28

VIDEO_EMBED_MAX_MB = 90

MODE_NAME = {
    "dashboard": "Dashboard",
    "realtime": "Realtime Detection",
    "image": "Image Detection",
    "video": "Video Detection",
    "history": "History Detection",
}


# =====================================================
# HISTORY STORAGE
# =====================================================
def load_history_from_file():
    try:
        if not os.path.exists(HISTORY_PATH):
            return []

        with open(HISTORY_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history_to_file(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as file:
            json.dump(history or [], file, ensure_ascii=False, indent=2)
    except Exception:
        pass


def clear_history_file():
    try:
        if os.path.exists(HISTORY_PATH):
            os.remove(HISTORY_PATH)
    except Exception:
        pass


atexit.register(clear_history_file)


# =====================================================
# ICON
# =====================================================
MOTORCYCLE_HELMET_SVG = """
<svg class="helmet-svg" viewBox="0 0 160 160" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
    <defs>
        <linearGradient id="helmetMain" x1="20" y1="20" x2="140" y2="140" gradientUnits="userSpaceOnUse">
            <stop offset="0" stop-color="#60a5fa"/>
            <stop offset="0.55" stop-color="#2563eb"/>
            <stop offset="1" stop-color="#1e3a8a"/>
        </linearGradient>
        <linearGradient id="helmetVisor" x1="54" y1="50" x2="137" y2="96" gradientUnits="userSpaceOnUse">
            <stop offset="0" stop-color="#111827"/>
            <stop offset="1" stop-color="#020617"/>
        </linearGradient>
    </defs>
    <path d="M24 85C24 47 51 22 88 22C121 22 145 45 145 79V93C145 107 134 118 120 118H83C63 118 47 102 47 82V79C47 60 62 45 81 45H113C104 36 92 32 78 34C55 37 40 55 40 82V107C40 114 35 119 28 119C21 119 16 114 16 107V91C16 88 19 85 24 85Z" fill="url(#helmetMain)"/>
    <path d="M53 79C53 62 66 51 83 51H122C131 51 138 58 138 67V75C138 78 136 81 133 82L96 94C89 96 82 92 80 85L78 77C77 73 74 70 70 70H59C56 70 53 73 53 76V79Z" fill="url(#helmetVisor)"/>
    <path d="M83 57H122C127 57 132 61 132 67V72L94 85C91 86 88 84 87 81L85 73C84 68 80 64 75 64H65C70 60 76 57 83 57Z" fill="#111827"/>
    <path d="M51 92C55 111 70 126 91 126H120C133 126 144 116 146 103C143 119 130 136 104 136H76C61 136 49 124 49 109V94C49 92 50 91 51 92Z" fill="#1d4ed8"/>
    <path d="M42 83H61C67 83 72 88 72 94V105C72 112 67 117 60 117H42V83Z" fill="#3b82f6"/>
    <path d="M91 102H124C128 102 131 105 131 109C131 113 128 116 124 116H91C87 116 84 113 84 109C84 105 87 102 91 102Z" fill="#0f172a" opacity="0.88"/>
    <path d="M91 110H121" stroke="#93c5fd" stroke-width="4" stroke-linecap="round" opacity="0.85"/>
    <path d="M59 40C72 29 91 27 107 33" stroke="#bfdbfe" stroke-width="7" stroke-linecap="round" fill="none" opacity="0.9"/>
</svg>
"""


# =====================================================
# BASIC HELPERS
# =====================================================
def bgr_to_rgb(frame_bgr):
    if frame_bgr is None:
        return None
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb):
    if image_rgb is None:
        return None

    if len(image_rgb.shape) == 2:
        return cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)

    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 4:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2BGR)

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def get_uploaded_video_path(video):
    if video is None:
        return None

    if isinstance(video, str):
        return video if os.path.exists(video) else None

    if isinstance(video, dict):
        for key in ["path", "name", "orig_name"]:
            value = video.get(key)
            if isinstance(value, str) and os.path.exists(value):
                return value

    if isinstance(video, (list, tuple)) and len(video) > 0:
        return get_uploaded_video_path(video[0])

    if hasattr(video, "name"):
        value = getattr(video, "name")
        if isinstance(value, str) and os.path.exists(value):
            return value

    return None


def resize_bgr_keep_aspect(frame_bgr, max_width):
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


def make_even(value):
    value = int(value)
    if value < 2:
        return 2
    return value if value % 2 == 0 else value - 1


def sanitize_fps(fps):
    try:
        fps = float(fps)
    except Exception:
        fps = 25.0

    if fps <= 0 or fps > 120:
        fps = 25.0

    return fps


def sanitize_result_fps(fps):
    fps = sanitize_fps(fps)

    if fps < VIDEO_RESULT_MIN_FPS:
        return float(VIDEO_RESULT_MIN_FPS)

    if fps > VIDEO_RESULT_MAX_FPS:
        return float(VIDEO_RESULT_MAX_FPS)

    return float(fps)


def extract_video_frame_rgb(video_path, frame_index=0, max_width=780):
    if video_path is None or not os.path.exists(video_path):
        return None

    cap = None

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_index is None:
            frame_index = max(0, total_frames // 3)

        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

        ret, frame_bgr = cap.read()

        if not ret or frame_bgr is None:
            return None

        frame_bgr, _ = resize_bgr_keep_aspect(frame_bgr, max_width)

        return bgr_to_rgb(frame_bgr)

    except Exception:
        return None

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def rgb_image_to_data_uri(image_rgb, max_width=240):
    if image_rgb is None:
        return ""

    try:
        image_rgb = image_rgb.copy()
        h, w = image_rgb.shape[:2]

        if w > max_width:
            scale = max_width / w
            image_rgb = cv2.resize(
                image_rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        image_bgr = rgb_to_bgr(image_rgb)

        ok, buffer = cv2.imencode(
            ".jpg",
            image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 78]
        )

        if not ok:
            return ""

        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    except Exception:
        return ""


def video_to_data_uri_thumbnail(video_path, max_width=240):
    frame_rgb = extract_video_frame_rgb(
        video_path,
        frame_index=None,
        max_width=max_width
    )

    if frame_rgb is None:
        return ""

    return rgb_image_to_data_uri(frame_rgb, max_width=max_width)


# =====================================================
# FFMPEG + VIDEO OUTPUT HELPERS
# =====================================================
def find_ffmpeg_exe():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path
    except Exception:
        pass

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "imageio-ffmpeg", "-q"]
        )

        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path

    except Exception:
        pass

    return None


def is_valid_video_file(video_path):
    if video_path is None or not os.path.exists(video_path):
        return False

    if os.path.getsize(video_path) <= 0:
        return False

    cap = None

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return False

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return False

        ret, frame = cap.read()

        if not ret or frame is None:
            return False

        return True

    except Exception:
        return False

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def convert_to_browser_mp4(input_path, output_fps):
    if input_path is None or not os.path.exists(input_path):
        return None

    ffmpeg_path = find_ffmpeg_exe()

    if ffmpeg_path is None:
        return None

    timestamp = int(time.time())
    output_path = os.path.abspath(
        os.path.join(OUTPUT_DIR, f"helmet_detection_result_{timestamp}.mp4")
    )

    output_fps = sanitize_fps(output_fps)

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        os.path.abspath(input_path),
        "-an",
        "-r",
        f"{output_fps}",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=900
        )

        if is_valid_video_file(output_path):
            return output_path

    except Exception:
        pass

    return None


def prepare_uploaded_video(video):
    video_path = get_uploaded_video_path(video)

    if video_path is None or not os.path.exists(video_path):
        return None, None

    first_preview = extract_video_frame_rgb(video_path, frame_index=0, max_width=780)

    return video_path, first_preview


def make_video_src(video_path):
    if video_path is None or not os.path.exists(video_path):
        return ""

    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    if size_mb <= VIDEO_EMBED_MAX_MB:
        try:
            with open(video_path, "rb") as file:
                encoded = base64.b64encode(file.read()).decode("utf-8")

            return f"data:video/mp4;base64,{encoded}"
        except Exception:
            pass

    path_for_browser = os.path.abspath(video_path).replace("\\", "/")
    encoded_path = quote(path_for_browser, safe="/:")

    return f"/file={encoded_path}?t={int(time.time())}"


def make_video_result_placeholder(title="Video Detection Result", text="Hasil video deteksi akan muncul di sini setelah proses selesai."):
    return f"""
    <div class="video-result-html fade-card">
        <div class="video-result-label">▣ {html.escape(title)}</div>
        <div class="video-result-empty">
            <h3>{html.escape(text)}</h3>
        </div>
    </div>
    """


def make_video_result_processing(progress_percent):
    return f"""
    <div class="video-result-html fade-card">
        <div class="video-result-label">▣ Video Detection Result</div>
        <div class="video-result-empty">
            <h3>Processing video detection...</h3>
            <p>{progress_percent}% completed</p>
        </div>
    </div>
    """


def make_video_result_player(video_path):
    if video_path is None or not os.path.exists(video_path):
        return make_video_result_placeholder(
            "Video Detection Result",
            "Video hasil deteksi belum tersedia."
        )

    video_src = make_video_src(video_path)

    if not video_src:
        return make_video_result_placeholder(
            "Video Detection Result",
            "Video hasil deteksi gagal dimuat."
        )

    file_name = html.escape(os.path.basename(video_path))

    return f"""
    <div class="video-result-html fade-card">
        <div class="video-result-label">▣ Video Detection Result</div>

        <video
            class="video-result-player"
            controls
            autoplay
            muted
            playsinline
            preload="auto">
            <source src="{video_src}" type="video/mp4">
            Browser tidak dapat memutar video ini.
        </video>

        <div class="video-result-caption">
            <span>Output: {file_name}</span>
        </div>
    </div>
    """


# =====================================================
# PERFORMANCE HELPERS
# =====================================================
def empty_metrics(status="Standby"):
    return {
        "duration_process": 0,
        "fps_process": 0,
        "fps_model": 0,
        "latency_ms": 0,
        "fps_average": 0,
        "progress_percent": 0,
        "processed_frame": 0,
        "total_frame": 0,
        "status": status
    }


def metric_value(metrics, key, default=0):
    try:
        return metrics.get(key, default)
    except Exception:
        return default


def build_single_process_metrics(process_start, status="Completed"):
    duration = time.perf_counter() - process_start
    fps = round(1 / duration, 2) if duration > 0 else 0

    return {
        "duration_process": round(duration, 2),
        "fps_process": fps,
        "fps_model": fps,
        "latency_ms": round(duration * 1000, 1),
        "fps_average": fps,
        "progress_percent": 100,
        "processed_frame": 1,
        "total_frame": 1,
        "status": status
    }


def make_performance_panel(title, metrics=None):
    if metrics is None:
        metrics = empty_metrics()

    duration_process = metric_value(metrics, "duration_process")
    fps_process = metric_value(metrics, "fps_process")
    fps_model = metric_value(metrics, "fps_model")
    latency_ms = metric_value(metrics, "latency_ms")
    fps_average = metric_value(metrics, "fps_average")
    progress_percent = metric_value(metrics, "progress_percent")
    processed_frame = metric_value(metrics, "processed_frame")
    total_frame = metric_value(metrics, "total_frame")
    status = metric_value(metrics, "status", "Standby")

    try:
        progress_percent = round(float(progress_percent), 1)
    except Exception:
        progress_percent = 0

    if total_frame and total_frame > 0:
        frame_text = f"{processed_frame} / {total_frame} frame"
    else:
        frame_text = "Live stream"

    return f"""
    <div class="performance-wrap fade-card">
        <div class="performance-head">
            <div>
                <h3>{title}</h3>
                <p>Status: <b>{status}</b> · {frame_text}</p>
            </div>
            <div class="performance-badge">{progress_percent}%</div>
        </div>

        <div class="performance-progress">
            <div class="performance-progress-fill" style="width:{progress_percent}%"></div>
        </div>

        <div class="performance-grid">
            <div class="performance-card">
                <span>Durasi Process</span>
                <h2>{duration_process}s</h2>
                <p>Total waktu proses berjalan</p>
            </div>

            <div class="performance-card">
                <span>FPS Process</span>
                <h2>{fps_process}</h2>
                <p>Kecepatan proses sistem</p>
            </div>

            <div class="performance-card">
                <span>FPS Model</span>
                <h2>{fps_model}</h2>
                <p>Kecepatan inference YOLO</p>
            </div>

            <div class="performance-card">
                <span>Latency</span>
                <h2>{latency_ms} ms</h2>
                <p>Waktu model membaca frame</p>
            </div>

            <div class="performance-card">
                <span>FPS Average</span>
                <h2>{fps_average}</h2>
                <p>Rata-rata performa proses</p>
            </div>
        </div>
    </div>
    """


# =====================================================
# VIDEO DETECTION + UNIQUE OBJECT TRACKING FINAL
# =====================================================
def bbox_iou(box_a, box_b):
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
        return 0

    return inter_area / union_area


def bbox_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_distance(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def detection_iou(det_a, det_b):
    return bbox_iou(det_a["box"], det_b["box"])


def nms_detections(detections, iou_threshold=VIDEO_NMS_IOU):
    """
    NMS tambahan agar bounding box tidak terlalu ramai.
    Dijalankan setelah YOLO, sebelum tracking.
    """
    if not detections:
        return []

    final_detections = []

    for cls_id in [WITH_HELMET_CLASS_ID, WITHOUT_HELMET_CLASS_ID]:
        class_detections = [det for det in detections if det["cls"] == cls_id]
        class_detections = sorted(class_detections, key=lambda x: x["score"], reverse=True)

        kept = []

        while class_detections:
            best = class_detections.pop(0)
            kept.append(best)

            class_detections = [
                det for det in class_detections
                if detection_iou(best, det) < iou_threshold
            ]

        final_detections.extend(kept)

    # Cross-class suppression: jika With Helmet dan Without Helmet menimpa area yang sama,
    # ambil yang confidence-nya lebih tinggi agar label tidak dobel.
    final_detections = sorted(final_detections, key=lambda x: x["score"], reverse=True)
    cleaned = []

    for det in final_detections:
        duplicate = False

        for kept in cleaned:
            if detection_iou(det, kept) >= CROSS_CLASS_NMS_IOU:
                duplicate = True
                break

        if not duplicate:
            cleaned.append(det)

    return cleaned


def filter_video_detections(detections, frame_w, frame_h):
    """
    Filter ringan untuk mengurangi false-positive kecil/aneh tanpa mengubah alur program.
    """
    filtered = []
    frame_area = max(1, frame_w * frame_h)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        area_ratio = box_area / frame_area

        if box_w < MIN_DET_BOX_WIDTH or box_h < MIN_DET_BOX_HEIGHT:
            continue

        if area_ratio < MIN_DET_BOX_AREA_RATIO:
            continue

        if area_ratio > MAX_DET_BOX_AREA_RATIO:
            continue

        filtered.append(det)

    return filtered


class SimpleObjectTracker:
    """
    Tracker final yang lebih stabil:
    - Objek baru belum langsung dihitung.
    - Objek dihitung sebagai unique object setelah muncul minimal TRACK_MIN_HITS_TO_CONFIRM.
    - Bounding box hanya digambar ketika track aktif dan sudah confirmed.
    - Count memakai confirmed object yang pernah muncul, bukan akumulasi per frame.
    """
    def __init__(
        self,
        iou_threshold=TRACK_IOU_THRESHOLD,
        max_center_distance=TRACK_MAX_CENTER_DISTANCE,
        max_lost_frames=TRACK_MAX_LOST_FRAMES,
        min_hits=TRACK_MIN_HITS_TO_CONFIRM
    ):
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_lost_frames = max_lost_frames
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks = {}
        self.confirmed_ids = set()
        self.counted_classes = {}

    def get_majority_class(self, track):
        votes = track.get("class_votes", {})
        with_votes = votes.get(WITH_HELMET_CLASS_ID, 0)
        without_votes = votes.get(WITHOUT_HELMET_CLASS_ID, 0)

        if with_votes >= without_votes:
            return WITH_HELMET_CLASS_ID

        return WITHOUT_HELMET_CLASS_ID

    def is_confirmed(self, track):
        return (
            track.get("hits", 0) >= self.min_hits
            and track.get("best_score", 0) >= TRACK_MIN_SCORE_TO_CONFIRM
        )

    def register_confirmed_count(self, track_id, track):
        # Count bersifat kumulatif selama satu video diproses.
        # Jadi ketika object sudah confirmed, jumlahnya tidak turun lagi
        # walaupun object keluar dari frame atau track lama dibersihkan.
        if self.is_confirmed(track):
            self.confirmed_ids.add(track_id)
            self.counted_classes[track_id] = self.get_majority_class(track)

    def update(self, detections, frame_index):
        matched_track_ids = set()
        matched_detection_indexes = set()

        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        for det_index, det in enumerate(detections):
            det_box = det["box"]
            det_cls = det["cls"]

            best_track_id = None
            best_score = -1

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                if frame_index - track["last_seen"] > self.max_lost_frames:
                    continue

                iou_value = bbox_iou(det_box, track["box"])
                dist_value = center_distance(det_box, track["box"])

                score = -1

                if iou_value >= self.iou_threshold:
                    score = 2.0 + iou_value
                elif dist_value <= self.max_center_distance:
                    score = 1.0 - (dist_value / self.max_center_distance)

                if det_cls == self.get_majority_class(track):
                    score += 0.18

                if score > best_score:
                    best_score = score
                    best_track_id = track_id

            if best_track_id is not None and best_score >= 0:
                track = self.tracks[best_track_id]

                old_box = track["box"]
                new_box = det_box

                # Smoothing box agar hasil video tidak terlalu bergetar.
                smoothed_box = [
                    int(old_box[0] * 0.45 + new_box[0] * 0.55),
                    int(old_box[1] * 0.45 + new_box[1] * 0.55),
                    int(old_box[2] * 0.45 + new_box[2] * 0.55),
                    int(old_box[3] * 0.45 + new_box[3] * 0.55),
                ]

                track["box"] = smoothed_box
                track["score"] = det["score"]
                track["best_score"] = max(track.get("best_score", 0), det["score"])
                track["last_seen"] = frame_index
                track["hits"] += 1
                track["class_votes"][det_cls] = track["class_votes"].get(det_cls, 0) + 1
                track["cls"] = self.get_majority_class(track)

                self.register_confirmed_count(best_track_id, track)

                matched_track_ids.add(best_track_id)
                matched_detection_indexes.add(det_index)

        for det_index, det in enumerate(detections):
            if det_index in matched_detection_indexes:
                continue

            new_track_id = self.next_id
            self.next_id += 1

            self.tracks[new_track_id] = {
                "id": new_track_id,
                "box": det["box"],
                "cls": det["cls"],
                "score": det["score"],
                "best_score": det["score"],
                "last_seen": frame_index,
                "hits": 1,
                "class_votes": {
                    det["cls"]: 1
                }
            }

        self.remove_old_tracks(frame_index)
        return self.get_active_tracks(frame_index)

    def remove_old_tracks(self, frame_index):
        remove_ids = []

        for track_id, track in self.tracks.items():
            if frame_index - track["last_seen"] > self.max_lost_frames:
                remove_ids.append(track_id)

        for track_id in remove_ids:
            self.tracks.pop(track_id, None)

    def get_active_tracks(self, frame_index):
        active_tracks = []

        for track_id, track in self.tracks.items():
            age = frame_index - track["last_seen"]

            if track_id not in self.confirmed_ids:
                continue

            if age > TRACK_DRAW_MAX_LOST_FRAMES:
                continue

            track_copy = track.copy()
            track_copy["cls"] = self.get_majority_class(track)
            active_tracks.append(track_copy)

        return active_tracks

    def get_unique_counts(self):
        with_count = 0
        without_count = 0

        # Pakai counted_classes agar jumlah tidak reset/turun ketika object
        # yang sudah confirmed keluar dari frame. Count hanya bertambah
        # saat ada object unik baru yang terkonfirmasi.
        for cls_id in self.counted_classes.values():
            if cls_id == WITH_HELMET_CLASS_ID:
                with_count += 1
            elif cls_id == WITHOUT_HELMET_CLASS_ID:
                without_count += 1

        return with_count, without_count


def run_yolo_on_video_frame(frame_bgr, conf_value):
    """
    Final balanced detection:
    - Resize inference ke 640, bukan 512 terlalu kecil dan bukan 960 terlalu berat.
    - YOLO tetap sekali per frame interval, tidak multi-crop, sehingga proses tidak melonjak lambat.
    - NMS + filter object mengurangi box terlalu ramai.
    """
    conf = clamp_conf(conf_value)

    original_h, original_w = frame_bgr.shape[:2]
    resized_bgr, scale = resize_bgr_keep_aspect(
        frame_bgr,
        VIDEO_DETECT_WIDTH
    )

    start_infer = time.perf_counter()

    results = model.predict(
        source=resized_bgr,
        conf=conf,
        imgsz=VIDEO_IMGSZ,
        iou=VIDEO_IOU,
        max_det=VIDEO_MAX_DET,
        verbose=False,
        augment=False
    )

    inference_time = time.perf_counter() - start_infer
    latency_ms = round(inference_time * 1000, 1)
    fps_model = round(1 / inference_time, 2) if inference_time > 0 else 0

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cls_id = int(box.cls[0])
            score = float(box.conf[0])

            if cls_id not in [WITH_HELMET_CLASS_ID, WITHOUT_HELMET_CLASS_ID]:
                continue

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

            detections.append({
                "box": [x1, y1, x2, y2],
                "cls": cls_id,
                "score": score
            })

    detections = filter_video_detections(detections, original_w, original_h)
    detections = nms_detections(detections, VIDEO_NMS_IOU)

    return detections, fps_model, latency_ms


def draw_video_tracks(frame_bgr, tracks):
    output = frame_bgr.copy()

    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        cls_id = int(track["cls"])
        score = float(track.get("score", 0))
        track_id = int(track["id"])

        if cls_id == WITH_HELMET_CLASS_ID:
            color = (0, 190, 0)
            label_name = "With Helmet"
        else:
            color = (0, 0, 255)
            label_name = "Without Helmet"

        # Label dibuat lebih pendek supaya video tidak terlalu ramai.
        label = f"{label_name} #{track_id}"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        (text_w, text_h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            2
        )

        y_text = max(y1 - 8, text_h + 10)

        cv2.rectangle(
            output,
            (x1, y_text - text_h - 8),
            (x1 + text_w + 10, y_text + 6),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 5, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            2
        )

    return output


def create_avi_writer(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if writer.isOpened():
        return writer

    try:
        writer.release()
    except Exception:
        pass

    return None


def process_video_direct(video_path, conf_value):
    cap = None
    writer = None

    try:
        conf = clamp_conf(conf_value)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            yield {
                "success": False,
                "final": True,
                "message": "Video gagal dibuka.",
                "metrics": empty_metrics("Failed")
            }
            return

        input_fps = sanitize_fps(cap.get(cv2.CAP_PROP_FPS))
        result_fps = sanitize_result_fps(input_fps / VIDEO_FRAME_STRIDE)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if total_frames <= 0:
            total_frames = 1

        if original_width <= 0 or original_height <= 0:
            yield {
                "success": False,
                "final": True,
                "message": "Resolusi video tidak terbaca.",
                "metrics": empty_metrics("Failed")
            }
            return

        if original_width > VIDEO_OUTPUT_MAX_WIDTH:
            output_scale = VIDEO_OUTPUT_MAX_WIDTH / original_width
            output_width = make_even(original_width * output_scale)
            output_height = make_even(original_height * output_scale)
        else:
            output_width = make_even(original_width)
            output_height = make_even(original_height)

        timestamp = int(time.time())

        raw_avi_path = os.path.abspath(
            os.path.join(OUTPUT_DIR, f"helmet_detection_raw_{timestamp}.avi")
        )

        writer = create_avi_writer(
            raw_avi_path,
            result_fps,
            output_width,
            output_height
        )

        if writer is None:
            yield {
                "success": False,
                "final": True,
                "message": "Gagal membuat writer video.",
                "metrics": empty_metrics("Failed")
            }
            return

        tracker = SimpleObjectTracker()
        process_start = time.perf_counter()

        source_frame_index = 0
        processed_frame_index = 0
        written_frames = 0
        estimated_output_frames = max(1, (total_frames + VIDEO_FRAME_STRIDE - 1) // VIDEO_FRAME_STRIDE)

        last_fps_model = 0
        last_latency_ms = 0
        active_tracks = []
        fps_process_values = []

        while True:
            ret, frame_bgr = cap.read()

            if not ret or frame_bgr is None:
                break

            source_frame_index += 1

            # Output video sengaja dibuat sedikit lebih rendah FPS-nya.
            # Yang ditulis hanya frame yang benar-benar diproses YOLO,
            # sehingga bounding box tidak terlihat telat karena frame kosong/stale.
            if (source_frame_index - 1) % VIDEO_FRAME_STRIDE != 0:
                continue

            processed_frame_index += 1

            frame_bgr = cv2.resize(
                frame_bgr,
                (output_width, output_height),
                interpolation=cv2.INTER_AREA
            )

            should_detect = (
                processed_frame_index == 1
                or processed_frame_index % VIDEO_DETECT_EVERY_N_FRAMES == 0
            )

            if should_detect:
                detections, last_fps_model, last_latency_ms = run_yolo_on_video_frame(
                    frame_bgr,
                    conf
                )

                active_tracks = tracker.update(
                    detections=detections,
                    frame_index=processed_frame_index
                )
            else:
                active_tracks = tracker.get_active_tracks(processed_frame_index)

            unique_with, unique_without = tracker.get_unique_counts()

            detected_bgr = draw_video_tracks(
                frame_bgr,
                active_tracks
            )

            writer.write(detected_bgr)
            written_frames += 1

            elapsed = time.perf_counter() - process_start
            fps_process = round(written_frames / elapsed, 2) if elapsed > 0 else 0
            fps_process_values.append(fps_process)

            fps_average = round(
                sum(fps_process_values) / len(fps_process_values),
                2
            ) if fps_process_values else 0

            progress_percent = min(
                100,
                round((source_frame_index / total_frames) * 100, 1)
            )

            metrics = {
                "duration_process": round(elapsed, 2),
                "fps_process": fps_process,
                "fps_model": last_fps_model,
                "latency_ms": last_latency_ms,
                "fps_average": fps_average,
                "progress_percent": progress_percent,
                "processed_frame": processed_frame_index,
                "total_frame": estimated_output_frames,
                "status": "Processing Video"
            }

            if processed_frame_index == 1 or processed_frame_index % VIDEO_PREVIEW_EVERY_N_FRAMES == 0:
                yield {
                    "success": True,
                    "final": False,
                    "with_helmet": unique_with,
                    "without_helmet": unique_without,
                    "confidence": conf,
                    "metrics": metrics,
                    "video_path": None
                }

        try:
            cap.release()
            cap = None
        except Exception:
            pass

        try:
            writer.release()
            writer = None
        except Exception:
            pass

        if not os.path.exists(raw_avi_path) or os.path.getsize(raw_avi_path) <= 0:
            unique_with, unique_without = tracker.get_unique_counts()

            yield {
                "success": False,
                "final": True,
                "with_helmet": unique_with,
                "without_helmet": unique_without,
                "confidence": conf,
                "metrics": empty_metrics("Output Failed"),
                "video_path": None,
                "message": "File video hasil deteksi gagal dibuat."
            }
            return

        final_mp4_path = convert_to_browser_mp4(raw_avi_path, result_fps)

        unique_with, unique_without = tracker.get_unique_counts()

        elapsed = time.perf_counter() - process_start
        final_fps_process = round(written_frames / elapsed, 2) if elapsed > 0 else 0

        final_metrics = {
            "duration_process": round(elapsed, 2),
            "fps_process": final_fps_process,
            "fps_model": last_fps_model,
            "latency_ms": last_latency_ms,
            "fps_average": final_fps_process,
            "progress_percent": 100,
            "processed_frame": written_frames,
            "total_frame": estimated_output_frames,
            "status": "Completed"
        }

        yield {
            "success": final_mp4_path is not None,
            "final": True,
            "with_helmet": unique_with,
            "without_helmet": unique_without,
            "confidence": conf,
            "metrics": final_metrics,
            "video_path": final_mp4_path,
            "message": "Video detection completed."
        }

    except Exception as error:
        yield {
            "success": False,
            "final": True,
            "message": f"Error video detection: {str(error)}",
            "metrics": empty_metrics("Error")
        }

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass


# =====================================================
# HISTORY DISPLAY
# =====================================================
def add_history(
    history,
    mode,
    conf_value,
    helmet_count,
    no_helmet_count,
    media_type,
    media_preview
):
    if history is None:
        history = []

    total = helmet_count + no_helmet_count
    compliance_rate, violation_rate = calculate_rates(
        helmet_count,
        no_helmet_count
    )

    if total == 0:
        status = "No Object"
    elif no_helmet_count == 0:
        status = "Compliant"
    else:
        status = "Violation"

    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "confidence": f"{float(conf_value):.2f}",
        "total": total,
        "with_helmet": helmet_count,
        "without_helmet": no_helmet_count,
        "compliance_rate": f"{compliance_rate}%",
        "violation_rate": f"{violation_rate}%",
        "status": status,
        "media_type": media_type,
        "media_preview": media_preview,
    }

    new_history = [record] + list(history)
    new_history = new_history[:50]

    save_history_to_file(new_history)

    return new_history


def make_history_table(history):
    if not history:
        return """
        <div class="history-wrap fade-card">
            <div class="history-header">
                <div>
                    <h2>Detection History</h2>
                    <p>Riwayat hasil deteksi dari Image Detection dan Video Detection akan tampil di sini.</p>
                </div>
                <div class="history-count">0 Records</div>
            </div>

            <div class="history-empty">
                <h2>Belum Ada Riwayat Deteksi</h2>
                <p>
                    Jalankan deteksi gambar atau video terlebih dahulu. Hasilnya akan tersimpan sebagai riwayat
                    dengan preview, waktu, mode deteksi, jumlah objek, compliance, violation, dan status.
                </p>
            </div>
        </div>
        """

    rows = ""

    for index, item in enumerate(history, start=1):
        status = str(item.get("status", "-"))

        if status == "Compliant":
            status_class = "history-safe"
        elif status == "No Object":
            status_class = "history-neutral"
        else:
            status_class = "history-danger"

        media_type = html.escape(str(item.get("media_type", "-")))
        media_preview = item.get("media_preview", "")

        if media_preview:
            media_html = f"""
            <div class="history-media-box">
                <img class="history-thumb" src="{media_preview}" alt="Detection Preview">
                <span class="media-type-badge">{media_type}</span>
            </div>
            """
        else:
            media_html = f"""
            <div class="history-media-empty">
                <span>{media_type}</span>
            </div>
            """

        rows += f"""
        <tr>
            <td>{index}</td>
            <td>{media_html}</td>
            <td>{html.escape(str(item.get("time", "-")))}</td>
            <td>{html.escape(str(item.get("mode", "-")))}</td>
            <td>{html.escape(str(item.get("confidence", "-")))}</td>
            <td>{html.escape(str(item.get("total", "-")))}</td>
            <td>{html.escape(str(item.get("with_helmet", "-")))}</td>
            <td>{html.escape(str(item.get("without_helmet", "-")))}</td>
            <td>{html.escape(str(item.get("compliance_rate", "-")))}</td>
            <td>{html.escape(str(item.get("violation_rate", "-")))}</td>
            <td><span class="{status_class}">{html.escape(status)}</span></td>
        </tr>
        """

    return f"""
    <div class="history-wrap fade-card">
        <div class="history-header">
            <div>
                <h2>Detection History</h2>
                <p>Riwayat hasil deteksi tersimpan dari Image Detection dan Video Detection.</p>
            </div>
            <div class="history-count">{len(history)} Records</div>
        </div>

        <div class="history-table-scroll">
            <table class="history-table">
                <thead>
                    <tr>
                        <th>No</th>
                        <th>Preview</th>
                        <th>Time</th>
                        <th>Mode</th>
                        <th>Confidence</th>
                        <th>Total Object</th>
                        <th>With Helmet</th>
                        <th>Without Helmet</th>
                        <th>Compliance</th>
                        <th>Violation</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>
    """


def clear_history():
    empty_history = []
    clear_history_file()

    return (
        empty_history,
        make_history_table(empty_history),
        make_history_side_panel(empty_history)
    )


def load_history_on_page_open():
    history = load_history_from_file()

    return (
        history,
        make_history_table(history)
    )


# =====================================================
# UI HTML
# =====================================================
def make_metric_cards(helmet_count, no_helmet_count):
    total = helmet_count + no_helmet_count

    return f"""
    <div class="metric-grid fade-card">
        <div class="metric-card">
            <div class="metric-icon blue">🛵</div>
            <div>
                <p>Total Object</p>
                <h2>{total}</h2>
                <span>Unique detected rider objects</span>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-icon sky">🪖</div>
            <div>
                <p>With Helmet</p>
                <h2>{helmet_count}</h2>
                <span>Unique safe objects</span>
            </div>
        </div>

        <div class="metric-card">
            <div class="metric-icon soft">⚠️</div>
            <div>
                <p>Without Helmet</p>
                <h2>{no_helmet_count}</h2>
                <span>Unique violation objects</span>
            </div>
        </div>
    </div>
    """


def make_detection_summary_panel(helmet_count, no_helmet_count, mode="Detection"):
    compliance_rate, violation_rate = calculate_rates(
        helmet_count,
        no_helmet_count
    )

    return f"""
    <div class="right-title">Detection Summary</div>

    <div class="summary-card main-summary">
        <div>
            <p>Current Mode</p>
            <h3>{mode}</h3>
        </div>
        <span class="online-badge">Online</span>
    </div>

    <div class="summary-card">
        <p>With Helmet</p>
        <h2 class="green-summary-text">{helmet_count}</h2>
        <span>Unique compliant riders</span>
    </div>

    <div class="summary-card">
        <p>Without Helmet</p>
        <h2 class="red-text">{no_helmet_count}</h2>
        <span>Unique potential violations</span>
    </div>

    <div class="summary-card">
        <p>Helmet Compliance Rate</p>
        <div class="progress-wrap">
            <div class="progress-bar-compliance" style="width:{compliance_rate}%"></div>
        </div>
        <h2 class="green-summary-text">{compliance_rate}%</h2>
        <span>Based on unique rider objects</span>
    </div>

    <div class="summary-card">
        <p>Helmet Violation Rate</p>
        <div class="progress-wrap">
            <div class="progress-bar-violation" style="width:{violation_rate}%"></div>
        </div>
        <h2 class="red-text">{violation_rate}%</h2>
        <span>Based on unique rider objects</span>
    </div>
    """


def make_dashboard_side_panel():
    return """
    <div class="right-title">System Overview</div>

    <div class="summary-card main-summary">
        <div>
            <p>Application</p>
            <h3>HelmetVision</h3>
        </div>
        <span class="online-badge">Online</span>
    </div>

    <div class="summary-card">
        <p>Detection Modes</p>
        <h2>3</h2>
        <span>Realtime, Image, and Video</span>
    </div>

    <div class="summary-card">
        <p>Main Focus</p>
        <h3>Helmet Compliance</h3>
        <span>Monitoring helmet usage and violations</span>
    </div>

    <div class="summary-card">
        <p>Tracking Logic</p>
        <h3>Unique Object</h3>
        <span>Video count uses simple object tracking</span>
    </div>
    """


def make_history_side_panel(history=None):
    if history is None:
        history = []

    total_records = len(history)
    image_count = sum(1 for item in history if item.get("media_type") == "Image")
    video_count = sum(1 for item in history if item.get("media_type") == "Video")

    return f"""
    <div class="right-title">History Overview</div>

    <div class="summary-card main-summary">
        <div>
            <p>Current Mode</p>
            <h3>History Detection</h3>
        </div>
        <span class="online-badge">Online</span>
    </div>

    <div class="summary-card">
        <p>Total Records</p>
        <h2>{total_records}</h2>
        <span>Saved detection history</span>
    </div>

    <div class="summary-card">
        <p>Image History</p>
        <h2>{image_count}</h2>
        <span>Records from image detection</span>
    </div>

    <div class="summary-card">
        <p>Video History</p>
        <h2>{video_count}</h2>
        <span>Records from video detection</span>
    </div>
    """


def make_dashboard_intro():
    return """
    <div class="dashboard-page fade-card">
        <div class="dashboard-hero-clean">
            <div class="dashboard-hero-content">
                <span class="hero-badge">Helmet Detection System</span>
                <h1>Hallo, User!</h1>
                <p class="hero-subtitle">Selamat datang di dashboard Helmet Detection.</p>

                <h2>HelmetVision Dashboard</h2>
                <p>
                    Platform visual untuk mendeteksi penggunaan helm pada pengendara melalui kamera realtime,
                    gambar, dan video. Sistem ini membantu memantau kepatuhan penggunaan helm secara lebih cepat,
                    terukur, dan mudah dipahami.
                </p>
            </div>

            <div class="hero-mini-grid">
                <div class="hero-mini-card">
                    <h3>Realtime Detection</h3>
                    <p>Memproses input kamera secara langsung untuk memantau objek pengendara.</p>
                </div>

                <div class="hero-mini-card">
                    <h3>Image Detection</h3>
                    <p>Mendeteksi penggunaan helm dari file gambar yang diunggah pengguna.</p>
                </div>

                <div class="hero-mini-card">
                    <h3>Video Detection</h3>
                    <p>Menganalisis video, memberi ID objek, dan menghitung unique object.</p>
                </div>

                <div class="hero-mini-card">
                    <h3>Detection History</h3>
                    <p>Menyimpan riwayat hasil deteksi gambar dan video dalam tabel yang mudah dibaca.</p>
                </div>
            </div>
        </div>

        <div class="workflow-section fade-card">
            <h2>How the System Works</h2>

            <div class="workflow-grid">
                <div class="workflow-card">
                    <div class="workflow-number">01</div>
                    <h3>Input Media</h3>
                    <p>Pilih kamera, gambar, atau video sesuai kebutuhan pengujian.</p>
                </div>

                <div class="workflow-card">
                    <div class="workflow-number">02</div>
                    <h3>YOLO Inference</h3>
                    <p>Model membaca objek dan mengklasifikasikan With Helmet atau Without Helmet.</p>
                </div>

                <div class="workflow-card">
                    <div class="workflow-number">03</div>
                    <h3>Object Tracking</h3>
                    <p>Objek yang sama pada frame berbeda tetap dihitung sebagai satu objek unik.</p>
                </div>

                <div class="workflow-card">
                    <div class="workflow-number">04</div>
                    <h3>Video Result</h3>
                    <p>Video hasil deteksi ditampilkan ulang dengan bounding box dan ID objek.</p>
                </div>
            </div>
        </div>
    </div>
    """


def make_page_intro(page):
    data = {
        "realtime": {
            "title": "Realtime Detection",
            "subtitle": "Gunakan kamera untuk mendeteksi penggunaan helm secara langsung.",
            "badge": "Live Camera Mode",
        },
        "image": {
            "title": "Image Detection",
            "subtitle": "Upload gambar untuk melihat hasil deteksi With Helmet dan Without Helmet.",
            "badge": "Image Analysis Mode",
        },
        "video": {
            "title": "Video Detection",
            "subtitle": "Upload video, proses video, lalu hasil akhir tampil di panel kanan sebagai video dengan bounding box.",
            "badge": "Video Processing Mode",
        },
        "history": {
            "title": "Detection History",
            "subtitle": "Lihat riwayat deteksi gambar dan video yang telah diproses.",
            "badge": "Saved Records",
        },
    }

    item = data.get(page, data["realtime"])

    return f"""
    <div class="mode-intro fade-card">
        <span>{item["badge"]}</span>
        <h2>{item["title"]}</h2>
        <p>{item["subtitle"]}</p>
    </div>
    """


def make_sidebar_context_card(page):
    context = {
        "dashboard": {
            "title": "Dashboard Guide",
            "body": "Halaman ini memperkenalkan fungsi utama HelmetVision, alur kerja sistem, dan mode deteksi yang tersedia.",
        },
        "realtime": {
            "title": "Realtime Guide",
            "body": "Gunakan kamera lokal untuk monitoring langsung. Processed output menampilkan hasil deteksi secara live.",
        },
        "image": {
            "title": "Image Guide",
            "body": "Unggah satu gambar, lalu tekan Detect Image. Hasil deteksi otomatis tersimpan ke history.",
        },
        "video": {
            "title": "Video Guide",
            "body": "Panel kanan menampilkan video hasil akhir dengan bounding box. Total object dihitung berdasarkan unique object tracking, bukan jumlah per frame.",
        },
        "history": {
            "title": "History Guide",
            "body": "Riwayat hanya akan hilang jika tombol Clear History ditekan atau program Gradio ditutup.",
        },
    }

    item = context.get(page, context["dashboard"])

    return f"""
    <div class="info-box fade-card">
        <h4>{item["title"]}</h4>
        <p>{item["body"]}</p>
    </div>
    """


# =====================================================
# NAVIGATION
# =====================================================
def nav_button_updates(active_page):
    pages = ["dashboard", "realtime", "image", "video", "history"]

    updates = []

    for page in pages:
        if page == active_page:
            updates.append(gr.update(variant="primary"))
        else:
            updates.append(gr.update(variant="secondary"))

    return updates


def switch_page(page, history=None):
    if history is None:
        history = load_history_from_file()

    show_confidence = page in ["realtime", "image", "video"]
    show_metric_cards = page in ["realtime", "image", "video"]

    if page == "dashboard":
        side_panel = make_dashboard_side_panel()
    elif page == "history":
        side_panel = make_history_side_panel(history)
    else:
        side_panel = make_detection_summary_panel(
            0,
            0,
            MODE_NAME.get(page, "Detection")
        )

    metric_update = gr.update(
        visible=show_metric_cards,
        value=make_metric_cards(0, 0) if show_metric_cards else ""
    )

    page_updates = [
        gr.update(visible=page == "dashboard"),
        gr.update(visible=page == "realtime"),
        gr.update(visible=page == "image"),
        gr.update(visible=page == "video"),
        gr.update(visible=page == "history"),
        side_panel,
        metric_update,
        gr.update(visible=show_confidence),
        make_sidebar_context_card(page),
    ]

    return tuple(page_updates + nav_button_updates(page))


# =====================================================
# CALLBACKS
# =====================================================
def detect_frame_light(frame_rgb, conf_value, last_monitor_update):
    result = detect_realtime_core(frame_rgb, conf_value)

    detected_image = result.get("image", None)
    metrics = result.get("metrics", empty_metrics("Live Recording"))

    if detected_image is not None:
        h, w = detected_image.shape[:2]
        if w > 460:
            scale = 460 / w
            detected_image = cv2.resize(
                detected_image,
                (460, int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

    now = time.time()

    if last_monitor_update is None:
        last_monitor_update = 0

    should_update_monitor = (
        now - float(last_monitor_update)
    ) >= REALTIME_MONITOR_UPDATE_INTERVAL

    if should_update_monitor:
        helmet_count = int(result.get("with_helmet", 0))
        no_helmet_count = int(result.get("without_helmet", 0))

        monitor_html = make_performance_panel(
            "Realtime Performance Monitor",
            metrics
        )

        summary_html = make_detection_summary_panel(
            helmet_count,
            no_helmet_count,
            "Realtime Detection"
        )

        metric_html = gr.update(
            visible=True,
            value=make_metric_cards(helmet_count, no_helmet_count)
        )

        last_monitor_update = now
    else:
        monitor_html = gr.update()
        summary_html = gr.update()
        metric_html = gr.update()

    return (
        detected_image,
        summary_html,
        metric_html,
        monitor_html,
        last_monitor_update
    )


def detect_image(image_rgb, conf_value, history):
    if history is None:
        history = load_history_from_file()

    if image_rgb is None:
        metrics = empty_metrics("Waiting Image")

        return (
            None,
            make_detection_summary_panel(0, 0, "Image Detection"),
            gr.update(visible=True, value=make_metric_cards(0, 0)),
            make_performance_panel("Image Detection Performance Monitor", metrics),
            make_history_table(history),
            history
        )

    process_start = time.perf_counter()
    result = detect_image_core(image_rgb, conf_value)

    metrics = build_single_process_metrics(
        process_start=process_start,
        status="Completed"
    )

    helmet_count = int(result.get("with_helmet", 0))
    no_helmet_count = int(result.get("without_helmet", 0))
    detected_image = result.get("image", None)
    confidence = float(result.get("confidence", conf_value))
    success = bool(result.get("success", False))

    if success and detected_image is not None:
        media_preview = rgb_image_to_data_uri(detected_image)

        history = add_history(
            history=history,
            mode="Image Detection",
            conf_value=confidence,
            helmet_count=helmet_count,
            no_helmet_count=no_helmet_count,
            media_type="Image",
            media_preview=media_preview
        )

    performance_html = make_performance_panel(
        "Image Detection Performance Monitor",
        metrics
    )

    return (
        detected_image,
        make_detection_summary_panel(
            helmet_count,
            no_helmet_count,
            "Image Detection"
        ),
        gr.update(
            visible=True,
            value=make_metric_cards(helmet_count, no_helmet_count)
        ),
        performance_html,
        make_history_table(history),
        history
    )


def handle_video_upload(video):
    uploaded_video_path, first_preview = prepare_uploaded_video(video)

    return (
        uploaded_video_path,
        make_video_result_placeholder(
            "Video Detection Result",
            "Klik Process Video untuk menampilkan hasil video deteksi di panel ini."
        ),
        make_performance_panel(
            "Video Detection Performance Monitor",
            empty_metrics("Ready to Process")
        )
    )


def detect_video(video, conf_value, history):
    if history is None:
        history = load_history_from_file()

    final_history = history
    video_path = get_uploaded_video_path(video)

    if video_path is None or not os.path.exists(video_path):
        yield (
            make_video_result_placeholder(
                "Video Detection Result",
                "Silakan upload video terlebih dahulu."
            ),
            make_detection_summary_panel(0, 0, "Video Detection"),
            gr.update(visible=True, value=make_metric_cards(0, 0)),
            make_history_table(final_history),
            final_history,
            make_performance_panel(
                "Video Detection Performance Monitor",
                empty_metrics("Waiting Video")
            )
        )
        return

    for result in process_video_direct(video_path, conf_value):
        success = bool(result.get("success", False))
        final_result = bool(result.get("final", False))
        output_video_path = result.get("video_path", None)

        helmet_count = int(result.get("with_helmet", 0))
        no_helmet_count = int(result.get("without_helmet", 0))
        confidence = float(result.get("confidence", conf_value))
        metrics = result.get("metrics", empty_metrics("Processing Video"))

        performance_html = make_performance_panel(
            "Video Detection Performance Monitor",
            metrics
        )

        progress_percent = metric_value(metrics, "progress_percent", 0)

        if final_result and success and output_video_path is not None:
            video_html = make_video_result_player(output_video_path)

            media_preview = video_to_data_uri_thumbnail(output_video_path)

            final_history = add_history(
                history=final_history,
                mode="Video Detection",
                conf_value=confidence,
                helmet_count=helmet_count,
                no_helmet_count=no_helmet_count,
                media_type="Video",
                media_preview=media_preview
            )

            yield (
                video_html,
                make_detection_summary_panel(
                    helmet_count,
                    no_helmet_count,
                    "Video Detection"
                ),
                gr.update(
                    visible=True,
                    value=make_metric_cards(helmet_count, no_helmet_count)
                ),
                make_history_table(final_history),
                final_history,
                performance_html
            )

        elif final_result and not success:
            yield (
                make_video_result_placeholder(
                    "Video Detection Result",
                    "Video hasil deteksi gagal dibuat. Pastikan imageio-ffmpeg tersedia."
                ),
                make_detection_summary_panel(
                    helmet_count,
                    no_helmet_count,
                    "Video Detection"
                ),
                gr.update(
                    visible=True,
                    value=make_metric_cards(helmet_count, no_helmet_count)
                ),
                make_history_table(final_history),
                final_history,
                performance_html
            )

        else:
            yield (
                make_video_result_processing(progress_percent),
                make_detection_summary_panel(
                    helmet_count,
                    no_helmet_count,
                    "Video Detection"
                ),
                gr.update(
                    visible=True,
                    value=make_metric_cards(helmet_count, no_helmet_count)
                ),
                make_history_table(final_history),
                final_history,
                performance_html
            )


# =====================================================
# CSS
# =====================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Inter:wght@400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; }

body {
    background:
        radial-gradient(circle at top left, rgba(96, 165, 250, 0.20), transparent 34%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.14), transparent 30%),
        linear-gradient(135deg, #edf5ff 0%, #f8fbff 42%, #eaf3ff 100%) !important;
}

.gradio-container,
.gradio-container * {
    font-family: 'Plus Jakarta Sans', 'Inter', Helvetica, Arial, sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
    padding: 22px !important;
    background: transparent !important;
}

footer { display: none !important; }

.gradio-container .gr-form,
.gradio-container .gr-group,
.gradio-container .form,
.gradio-container fieldset {
    border: none !important;
    box-shadow: none !important;
}

.fade-card {
    animation: fadeInSoft 0.95s ease both;
}

.page-animate {
    animation: pageFadeIn 1.05s ease both;
}

@keyframes fadeInSoft {
    from {
        opacity: 0;
        transform: translateY(16px) scale(0.985);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}

@keyframes pageFadeIn {
    from {
        opacity: 0;
        transform: translateY(22px);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
        filter: blur(0);
    }
}

.top-nav {
    width: 100% !important;
    background: linear-gradient(145deg, rgba(255,255,255,0.94), rgba(242,247,255,0.94)) !important;
    border-radius: 28px !important;
    padding: 14px 18px !important;
    margin-bottom: 22px !important;
    align-items: center !important;
    box-shadow:
        0 18px 44px rgba(31, 72, 135, 0.12),
        inset 0 1px 0 rgba(255,255,255,0.90) !important;
    border: 1px solid rgba(198, 218, 244, 0.95) !important;
}

.logo-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    white-space: nowrap;
}

.logo-icon {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #4f8dff, #1f64ff);
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.logo-icon .helmet-svg {
    width: 34px;
    height: 34px;
}

.logo-text {
    font-size: 19px;
    font-weight: 900;
    color: #102d55;
}

.nav-pills {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 10px !important;
    flex-wrap: wrap !important;
}

.nav-btn {
    min-width: 126px !important;
}

.nav-btn button {
    width: 100% !important;
    border-radius: 16px !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 900 !important;
    font-size: 14px !important;
    padding: 12px 18px !important;
}

.nav-btn button.secondary {
    background: linear-gradient(145deg, #4a5a70, #334157) !important;
}

.nav-btn button.primary {
    background: linear-gradient(135deg, #1d4ed8 0%, #2f6bff 50%, #38bdf8 100%) !important;
}

.layout-row { gap: 20px !important; }

.left-panel,
.main-panel,
.right-panel {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.97), rgba(245,249,255,0.94)) !important;
    border-radius: 36px !important;
    padding: 24px !important;
    box-shadow:
        0 20px 50px rgba(31, 72, 135, 0.12),
        inset 0 1px 0 rgba(255,255,255,0.88) !important;
    border: 1px solid rgba(203, 221, 244, 0.95) !important;
    min-height: 82vh !important;
    overflow: hidden !important;
}

.profile-card {
    background:
        radial-gradient(circle at 30% 0%, rgba(96, 165, 250, 0.25), transparent 42%),
        linear-gradient(180deg, #eaf3ff, #ffffff);
    border-radius: 28px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    border: 1px solid rgba(214, 230, 249, 0.92);
}

.avatar-box {
    width: 126px;
    height: 126px;
    margin: 0 auto 16px auto;
    border-radius: 32px;
    background: linear-gradient(135deg, #dbeafe, #9ec5ff);
    display: flex;
    align-items: center;
    justify-content: center;
}

.avatar-box .helmet-svg {
    width: 98px;
    height: 98px;
}

.profile-card h3 {
    margin: 0;
    font-size: 20px;
    font-weight: 900;
    color: #102d55;
}

.profile-card p {
    margin: 7px 0 0 0;
    color: #6680a5;
    font-size: 13px;
    line-height: 1.6;
}

.status-pill {
    margin-top: 15px;
    display: inline-block;
    padding: 8px 15px;
    border-radius: 999px;
    background: linear-gradient(135deg, #102d55, #1f4f8d);
    color: white;
    font-size: 12px;
    font-weight: 800;
}

.confidence-card {
    background: linear-gradient(145deg, #1f3554, #27476f) !important;
    border-radius: 22px !important;
    padding: 16px !important;
    margin-bottom: 20px !important;
    border: 1px solid #365b8c !important;
}

.confidence-card label {
    color: white !important;
    font-weight: 900 !important;
}

.info-box {
    background:
        linear-gradient(145deg, #f7fbff, #edf5ff);
    border-radius: 24px;
    padding: 18px;
    margin-top: 18px;
    border: 1px solid #d8e7f8;
}

.info-box h4 {
    margin: 0 0 10px 0;
    color: #102d55;
    font-size: 15px;
    font-weight: 900;
}

.info-box p {
    font-size: 13px;
    line-height: 1.75;
    color: #5e789b;
    margin: 0;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(160px, 1fr));
    gap: 14px;
    margin-bottom: 20px;
}

.metric-card {
    background:
        linear-gradient(145deg, #ffffff, #f3f8ff);
    border-radius: 24px;
    padding: 17px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    border: 1px solid #dbe8f7;
}

.metric-card p {
    margin: 0;
    color: #637da3;
    font-size: 13px;
    font-weight: 800;
}

.metric-card h2 {
    margin: 4px 0;
    color: #102d55;
    font-size: 29px;
    font-weight: 900;
}

.metric-card span {
    color: #8aa0c0;
    font-size: 12px;
}

.metric-icon {
    min-width: 38px;
    height: 38px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.metric-icon.blue { background: #dce9ff; }
.metric-icon.sky { background: #dff2ff; }
.metric-icon.soft { background: #eef3ff; }

.dashboard-hero-clean,
.mode-intro {
    background:
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.20), transparent 30%),
        linear-gradient(145deg, #102d55, #1d4ed8);
    border-radius: 26px;
    padding: 22px;
    margin-bottom: 18px;
    color: white;
    overflow: hidden;
}

.dashboard-hero-clean {
    border-radius: 34px;
    padding: 38px;
    background:
        radial-gradient(circle at top left, rgba(147, 197, 253, 0.35), transparent 34%),
        radial-gradient(circle at top right, rgba(255,255,255,0.17), transparent 27%),
        linear-gradient(135deg, #183a66 0%, #2457d6 52%, #3ba4f4 100%);
}

.hero-badge,
.mode-intro span {
    display: inline-block;
    background: rgba(255, 255, 255, 0.16);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: white;
    padding: 8px 13px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 900;
    margin-bottom: 14px;
}

.dashboard-hero-clean h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 900;
}

.hero-subtitle {
    margin: 7px 0 20px 0 !important;
    color: #dbeafe !important;
}

.dashboard-hero-clean h2,
.mode-intro h2 {
    margin: 0;
    color: white;
    font-size: 34px;
    font-weight: 900;
}

.dashboard-hero-clean p,
.mode-intro p {
    max-width: 880px;
    margin: 12px 0 0 0;
    color: #dbeafe;
    font-size: 15px;
    line-height: 1.85;
}

.hero-mini-grid {
    margin-top: 25px;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
}

.hero-mini-card {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 22px;
    padding: 18px;
}

.hero-mini-card h3 {
    margin: 0 0 8px 0;
    font-size: 18px;
    font-weight: 900;
    color: white;
}

.hero-mini-card p {
    margin: 0;
    color: #dbeafe;
    font-size: 13px;
    line-height: 1.7;
}

.workflow-section {
    margin-top: 20px;
    background:
        linear-gradient(145deg, rgba(255,255,255,0.94), rgba(241,247,255,0.96));
    border: 1px solid #dbe7f6;
    border-radius: 30px;
    padding: 26px;
}

.workflow-section h2 {
    margin: 0 0 16px 0;
    color: #102d55;
    font-size: 24px;
    font-weight: 900;
}

.workflow-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.workflow-card {
    background: #ffffff;
    border: 1px solid #dbe7f6;
    border-radius: 22px;
    padding: 18px;
}

.workflow-number {
    width: 38px;
    height: 38px;
    border-radius: 14px;
    background: linear-gradient(135deg, #dbeafe, #bfdbfe);
    color: #245cf0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    margin-bottom: 12px;
}

.workflow-card h3 {
    margin: 0 0 8px 0;
    color: #102d55;
    font-size: 16px;
    font-weight: 900;
}

.workflow-card p {
    margin: 0;
    color: #607a9f;
    font-size: 13px;
    line-height: 1.7;
}

.media-box,
.video-file-upload {
    border-radius: 28px !important;
    overflow: hidden !important;
    background: linear-gradient(145deg, #f8fbff, #edf5ff) !important;
    border: 1px solid #dbe7f6 !important;
}

.video-file-upload {
    min-height: 95px !important;
    padding: 10px !important;
    margin-bottom: 14px !important;
}

.media-box img,
.media-box video {
    border-radius: 22px !important;
    object-fit: contain !important;
}

.video-two-panel {
    gap: 14px !important;
    align-items: stretch !important;
}

.video-result-html {
    width: 100%;
    min-height: 430px;
    border-radius: 28px;
    overflow: hidden;
    border: 1px solid #dbe7f6;
    background: #0f172a;
    position: relative;
}

.video-result-label {
    position: absolute;
    z-index: 5;
    top: 10px;
    left: 10px;
    padding: 8px 12px;
    border-radius: 7px;
    background: #2563eb;
    color: white;
    font-weight: 900;
    font-size: 13px;
}

.video-result-player {
    width: 100%;
    height: 430px;
    display: block;
    object-fit: contain;
    background: #0f172a;
}

.video-result-caption {
    position: absolute;
    left: 12px;
    bottom: 12px;
    z-index: 5;
    background: rgba(15, 23, 42, 0.78);
    color: #dbeafe;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 800;
}

.video-result-empty {
    width: 100%;
    min-height: 430px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #dbeafe;
    text-align: center;
    padding: 20px;
}

.video-result-empty h3 {
    color: #ffffff;
    font-size: 18px;
    font-weight: 900;
    margin: 0;
}

.video-result-empty p {
    color: #93c5fd;
    margin-top: 8px;
}

.action-row {
    margin-top: 14px !important;
    align-items: stretch !important;
    gap: 14px !important;
}

.button-wrap {
    background: linear-gradient(145deg, #1f3554, #27476f) !important;
    border-radius: 16px !important;
    padding: 12px !important;
    min-height: 104px !important;
    display: flex !important;
    align-items: stretch !important;
}

.big-action-btn button {
    width: 100% !important;
    min-height: 78px !important;
    border-radius: 22px !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #2f6bff, #38bdf8) !important;
    border: none !important;
    color: white !important;
}

.performance-wrap {
    background:
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.18), transparent 32%),
        linear-gradient(145deg, #1b3456, #102d55);
    border-radius: 26px;
    padding: 19px;
    margin-top: 17px;
    border: 1px solid #315681;
}

.performance-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}

.performance-head h3 {
    margin: 0;
    color: #ffffff;
    font-size: 18px;
    font-weight: 900;
}

.performance-head p {
    margin: 5px 0 0 0;
    color: #b8ccea;
    font-size: 13px;
}

.performance-badge {
    background: linear-gradient(135deg, #2f6bff, #38bdf8);
    color: #ffffff;
    border-radius: 999px;
    padding: 9px 14px;
    font-size: 13px;
    font-weight: 900;
    white-space: nowrap;
}

.performance-progress {
    width: 100%;
    height: 10px;
    background: #344d73;
    border-radius: 999px;
    overflow: hidden;
    margin-bottom: 14px;
}

.performance-progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #22c55e, #60a5fa);
    border-radius: 999px;
}

.performance-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(110px, 1fr));
    gap: 10px;
}

.performance-card {
    background: linear-gradient(145deg, #ffffff, #f3f8ff);
    border-radius: 18px;
    padding: 14px;
    border: 1px solid #dbe7f6;
}

.performance-card span {
    color: #637da3;
    font-size: 12px;
    font-weight: 900;
}

.performance-card h2 {
    margin: 5px 0;
    color: #102d55;
    font-size: 22px;
    font-weight: 900;
}

.performance-card p {
    margin: 0;
    color: #7d94b5;
    font-size: 11px;
    line-height: 1.4;
}

.right-title {
    font-size: 22px;
    font-weight: 900;
    color: #102d55;
    margin-bottom: 16px;
}

.summary-card {
    background:
        linear-gradient(145deg, #ffffff, #f4f9ff);
    border-radius: 24px;
    padding: 17px;
    margin-bottom: 14px;
    border: 1px solid #dfe8f5;
}

.summary-card p {
    margin: 0;
    color: #637da3;
    font-size: 13px;
    font-weight: 800;
}

.summary-card h2,
.summary-card h3 {
    margin: 6px 0;
    color: #102d55;
    font-weight: 900;
}

.summary-card span {
    color: #8aa0c0;
    font-size: 12px;
}

.main-summary {
    background: linear-gradient(135deg, #e9f1ff, #ffffff);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.online-badge {
    background: #e0ebff;
    color: #245cf0 !important;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: 900;
}

.green-summary-text { color: #16a34a !important; }
.red-text { color: #d94a4a !important; }

.progress-wrap {
    width: 100%;
    height: 12px;
    background: #dfe7f3;
    border-radius: 999px;
    overflow: hidden;
    margin: 10px 0;
}

.progress-bar-compliance {
    height: 100%;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 999px;
}

.progress-bar-violation {
    height: 100%;
    background: linear-gradient(135deg, #ff6b6b, #d94a4a);
    border-radius: 999px;
}

.history-wrap {
    background:
        linear-gradient(145deg, #ffffff, #f3f8ff);
    border: 1px solid #dbe7f6;
    border-radius: 32px;
    padding: 28px;
    overflow: hidden;
}

.history-header {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 18px;
}

.history-header h2 {
    margin: 0;
    color: #102d55;
    font-size: 28px;
    font-weight: 900;
}

.history-header p {
    margin: 8px 0 0 0;
    color: #607a9f;
    font-size: 14px;
}

.history-count {
    background: #e0ebff;
    color: #245cf0;
    border-radius: 999px;
    padding: 12px 18px;
    font-weight: 900;
    white-space: nowrap;
    height: fit-content;
}

.history-table-scroll {
    overflow-x: auto;
    border-radius: 18px;
    border: 1px solid #dbe7f6;
}

.history-table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1180px;
    background: white;
}

.history-table th {
    background: #24446f;
    color: white;
    padding: 14px;
    font-size: 13px;
    text-align: left;
}

.history-table td {
    padding: 13px 14px;
    border-bottom: 1px solid #e4edf8;
    color: #24446f;
    font-size: 13px;
    font-weight: 600;
}

.history-media-box {
    position: relative;
    width: 145px;
}

.history-thumb {
    width: 145px;
    height: 90px;
    object-fit: cover;
    border-radius: 14px;
    border: 2px solid #dbe7f6;
    display: block;
}

.media-type-badge {
    position: absolute;
    left: 8px;
    bottom: 8px;
    background: rgba(22, 52, 95, 0.92);
    color: white;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 900;
}

.history-safe,
.history-danger,
.history-neutral {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 900;
}

.history-safe { background: #e7f8ee; color: #16833a; }
.history-danger { background: #ffecec; color: #d94a4a; }
.history-neutral { background: #eef3ff; color: #245cf0; }

.history-empty {
    background: #ffffff;
    border: 1px dashed #bdd2ef;
    border-radius: 22px;
    padding: 26px;
}

.history-empty h2 {
    margin: 0 0 8px 0;
    color: #102d55;
    font-size: 22px;
    font-weight: 900;
}

.history-empty p {
    margin: 0;
    color: #607a9f;
    font-size: 14px;
    line-height: 1.7;
}

.clear-history-btn button {
    background: linear-gradient(135deg, #d94a4a, #b91c1c) !important;
    color: white !important;
    border: none !important;
    font-weight: 900 !important;
    border-radius: 16px !important;
}

@media (max-width: 1200px) {
    .performance-grid,
    .workflow-grid {
        grid-template-columns: repeat(2, minmax(110px, 1fr));
    }

    .hero-mini-grid {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .metric-grid,
    .workflow-grid,
    .performance-grid {
        grid-template-columns: 1fr;
    }
}
"""


# =====================================================
# UI
# =====================================================
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=custom_css,
    title=APP_TITLE
) as demo:

    initial_history = load_history_from_file()

    history_state = gr.State(value=initial_history)
    realtime_monitor_update_state = gr.State(value=0.0)

    with gr.Row(elem_classes="top-nav"):
        with gr.Column(scale=1, min_width=220):
            gr.HTML(f"""
            <div class="logo-wrap">
                <div class="logo-icon">{MOTORCYCLE_HELMET_SVG}</div>
                <div class="logo-text">HelmetVision</div>
            </div>
            """)

        with gr.Column(scale=4, min_width=650):
            with gr.Row(elem_classes="nav-pills"):
                btn_dashboard = gr.Button("Dashboard", variant="primary", elem_classes="nav-btn")
                btn_realtime = gr.Button("Realtime", variant="secondary", elem_classes="nav-btn")
                btn_image = gr.Button("Image", variant="secondary", elem_classes="nav-btn")
                btn_video = gr.Button("Video", variant="secondary", elem_classes="nav-btn")
                btn_history = gr.Button("History", variant="secondary", elem_classes="nav-btn")

    with gr.Row(elem_classes="layout-row"):

        with gr.Column(scale=1, min_width=250, elem_classes="left-panel"):
            gr.HTML(f"""
            <div class="profile-card">
                <div class="avatar-box">{MOTORCYCLE_HELMET_SVG}</div>
                <h3>Kelompok YOLO</h3>
                <p>Helmet Detection<br>System</p>
                <div class="status-pill">● System Active</div>
            </div>
            """)

            with gr.Group(visible=False) as confidence_group:
                conf_slider = gr.Slider(
                    minimum=CONF_MIN,
                    maximum=CONF_MAX,
                    value=CONF_DEFAULT,
                    step=0.01,
                    label="Confidence Threshold",
                    interactive=True,
                    elem_classes="confidence-card"
                )

            sidebar_context_panel = gr.HTML(make_sidebar_context_card("dashboard"))

        with gr.Column(scale=4, min_width=680, elem_classes="main-panel"):
            dashboard_metrics = gr.HTML(
                value="",
                visible=False
            )

            with gr.Group(visible=True, elem_classes="page-animate") as dashboard_page:
                gr.HTML(make_dashboard_intro())

            with gr.Group(visible=False, elem_classes="page-animate") as realtime_page:
                gr.HTML(make_page_intro("realtime"))

                with gr.Row():
                    webcam = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label="Input Camera",
                        height=430,
                        elem_classes="media-box"
                    )

                    webcam_output = gr.Image(
                        label="Processed Output",
                        height=430,
                        elem_classes="media-box"
                    )

                realtime_performance_panel = gr.HTML(
                    make_performance_panel(
                        "Realtime Performance Monitor",
                        empty_metrics("Waiting Camera")
                    )
                )

            with gr.Group(visible=False, elem_classes="page-animate") as image_page:
                gr.HTML(make_page_intro("image"))

                with gr.Row():
                    image_input = gr.Image(
                        type="numpy",
                        label="Upload Image",
                        height=430,
                        elem_classes="media-box"
                    )

                    image_output = gr.Image(
                        label="Processed Image",
                        height=430,
                        elem_classes="media-box"
                    )

                with gr.Row(elem_classes="action-row"):
                    with gr.Column(scale=1, elem_classes="button-wrap"):
                        btn_img = gr.Button(
                            "Detect Image",
                            variant="primary",
                            elem_classes="big-action-btn"
                        )

                image_performance_panel = gr.HTML(
                    make_performance_panel(
                        "Image Detection Performance Monitor",
                        empty_metrics("Waiting Image")
                    )
                )

            with gr.Group(visible=False, elem_classes="page-animate") as video_page:
                gr.HTML(make_page_intro("video"))

                video_input = gr.File(
                    label="Upload Video File",
                    file_types=[".mp4", ".avi", ".mov", ".mkv"],
                    type="filepath",
                    elem_classes="video-file-upload"
                )

                with gr.Row(elem_classes="video-two-panel"):
                    uploaded_video_player = gr.Video(
                        label="Uploaded Video",
                        height=430,
                        elem_classes="media-box"
                    )

                    video_result_panel = gr.HTML(
                        make_video_result_placeholder()
                    )

                with gr.Row(elem_classes="action-row"):
                    with gr.Column(scale=1, elem_classes="button-wrap"):
                        btn_vid = gr.Button(
                            "Process Video",
                            variant="primary",
                            elem_classes="big-action-btn"
                        )

                video_performance_panel = gr.HTML(
                    make_performance_panel(
                        "Video Detection Performance Monitor",
                        empty_metrics("Waiting Video")
                    )
                )

            with gr.Group(visible=False, elem_classes="page-animate") as history_page:
                gr.HTML(make_page_intro("history"))

                history_panel = gr.HTML(make_history_table(initial_history))

                clear_history_btn = gr.Button(
                    "Clear History",
                    variant="secondary",
                    elem_classes="clear-history-btn"
                )

        with gr.Column(scale=1, min_width=280, elem_classes="right-panel"):
            summary_panel = gr.HTML(make_dashboard_side_panel())

    page_outputs = [
        dashboard_page,
        realtime_page,
        image_page,
        video_page,
        history_page,
        summary_panel,
        dashboard_metrics,
        confidence_group,
        sidebar_context_panel,
        btn_dashboard,
        btn_realtime,
        btn_image,
        btn_video,
        btn_history,
    ]

    btn_dashboard.click(
        fn=lambda history: switch_page("dashboard", history),
        inputs=[history_state],
        outputs=page_outputs
    )

    btn_realtime.click(
        fn=lambda history: switch_page("realtime", history),
        inputs=[history_state],
        outputs=page_outputs
    )

    btn_image.click(
        fn=lambda history: switch_page("image", history),
        inputs=[history_state],
        outputs=page_outputs
    )

    btn_video.click(
        fn=lambda history: switch_page("video", history),
        inputs=[history_state],
        outputs=page_outputs
    )

    btn_history.click(
        fn=lambda history: switch_page("history", history),
        inputs=[history_state],
        outputs=page_outputs
    )

    demo.load(
        fn=load_history_on_page_open,
        inputs=[],
        outputs=[
            history_state,
            history_panel
        ]
    )

    video_input.change(
        fn=handle_video_upload,
        inputs=[video_input],
        outputs=[
            uploaded_video_player,
            video_result_panel,
            video_performance_panel
        ]
    )

    webcam.stream(
        fn=detect_frame_light,
        inputs=[
            webcam,
            conf_slider,
            realtime_monitor_update_state
        ],
        outputs=[
            webcam_output,
            summary_panel,
            dashboard_metrics,
            realtime_performance_panel,
            realtime_monitor_update_state
        ],
        queue=False,
        show_progress="hidden",
        stream_every=0.08
    )

    btn_img.click(
        fn=detect_image,
        inputs=[image_input, conf_slider, history_state],
        outputs=[
            image_output,
            summary_panel,
            dashboard_metrics,
            image_performance_panel,
            history_panel,
            history_state
        ]
    )

    btn_vid.click(
        fn=detect_video,
        inputs=[video_input, conf_slider, history_state],
        outputs=[
            video_result_panel,
            summary_panel,
            dashboard_metrics,
            history_panel,
            history_state,
            video_performance_panel
        ]
    )

    clear_history_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[
            history_state,
            history_panel,
            summary_panel
        ]
    )


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    demo.queue(
        default_concurrency_limit=1,
        max_size=10
    )

    demo.launch(
        share=True,
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        show_error=True,
        allowed_paths=[OUTPUT_DIR]
    )
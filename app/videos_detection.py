import cv2
import os
import time
import shutil
import subprocess
from collections import deque

from detection_engine import draw_detection, clamp_conf, calculate_rates


# =====================================================
# VIDEO CONFIG
# =====================================================
VIDEO_MAX_WIDTH = 640
FRAME_SKIP = 4
VIDEO_MIN_FPS = 5
VIDEO_OUTPUT_DIR = "outputs"
YIELD_EVERY_N_FRAMES = 5


# =====================================================
# HELPER
# =====================================================
def get_video_path(video):
    if video is None:
        return None

    if isinstance(video, str):
        return video

    if isinstance(video, dict):
        return video.get("path") or video.get("name")

    if isinstance(video, (list, tuple)) and len(video) > 0:
        return video[0]

    if hasattr(video, "name"):
        return video.name

    return None


def get_ffmpeg_path():
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def convert_to_browser_mp4(input_path, output_path):
    ffmpeg_path = get_ffmpeg_path()

    if ffmpeg_path is None:
        return input_path

    command = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-movflags", "+faststart",
        "-an",
        output_path
    ]

    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path

        return input_path

    except Exception:
        return input_path


def get_output_size(original_width, original_height, max_width=VIDEO_MAX_WIDTH):
    if original_width > max_width:
        scale = max_width / original_width
        output_width = int(original_width * scale)
        output_height = int(original_height * scale)
    else:
        output_width = original_width
        output_height = original_height

    output_width = max(2, output_width - (output_width % 2))
    output_height = max(2, output_height - (output_height % 2))

    return output_width, output_height


def make_video_metrics(
    process_start,
    processed_frame_count,
    total_frames_estimate,
    model_fps,
    latency_ms,
    fps_history
):
    duration = time.perf_counter() - process_start
    fps_process = processed_frame_count / duration if duration > 0 else 0
    fps_average = sum(fps_history) / len(fps_history) if fps_history else fps_process

    progress_percent = 0
    if total_frames_estimate and total_frames_estimate > 0:
        estimated_processed_total = max(1, total_frames_estimate // FRAME_SKIP)
        progress_percent = min(100, (processed_frame_count / estimated_processed_total) * 100)

    return {
        "duration_process": round(duration, 2),
        "fps_process": round(fps_process, 2),
        "fps_model": round(model_fps, 2),
        "latency_ms": round(latency_ms, 1),
        "fps_average": round(fps_average, 2),
        "progress_percent": round(progress_percent, 1),
        "processed_frame": processed_frame_count,
        "total_frame": total_frames_estimate,
        "status": "Processing Video"
    }


def make_base_result(
    success=False,
    final=False,
    video_path=None,
    with_helmet=0,
    without_helmet=0,
    confidence=0.3,
    message="",
    metrics=None
):
    if metrics is None:
        metrics = {
            "duration_process": 0,
            "fps_process": 0,
            "fps_model": 0,
            "latency_ms": 0,
            "fps_average": 0,
            "progress_percent": 0,
            "processed_frame": 0,
            "total_frame": 0,
            "status": "Standby"
        }

    return {
        "success": success,
        "final": final,
        "video_path": video_path,
        "with_helmet": with_helmet,
        "without_helmet": without_helmet,
        "confidence": confidence,
        "message": message,
        "metrics": metrics
    }


# =====================================================
# PROGRESS GENERATOR
# =====================================================
def detect_video_core_progress(video, conf_value):
    cap = None
    writer = None

    conf = clamp_conf(conf_value)
    video_path = get_video_path(video)

    if video_path is None:
        yield make_base_result(
            success=False,
            final=True,
            confidence=conf,
            message="Silakan upload video terlebih dahulu."
        )
        return

    if not os.path.exists(video_path):
        yield make_base_result(
            success=False,
            final=True,
            confidence=conf,
            message=f"File video tidak ditemukan: {video_path}"
        )
        return

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            yield make_base_result(
                success=False,
                final=True,
                confidence=conf,
                message="Video gagal dibuka. Gunakan format MP4 standar."
            )
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if original_fps is None or original_fps <= 0:
            original_fps = 25

        if original_width <= 0 or original_height <= 0:
            yield make_base_result(
                success=False,
                final=True,
                confidence=conf,
                message="Resolusi video tidak terbaca."
            )
            return

        output_width, output_height = get_output_size(
            original_width,
            original_height,
            VIDEO_MAX_WIDTH
        )

        output_fps = original_fps / FRAME_SKIP

        if output_fps < VIDEO_MIN_FPS:
            output_fps = VIDEO_MIN_FPS

        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)

        timestamp = int(time.time())

        raw_output_path = os.path.abspath(
            os.path.join(
                VIDEO_OUTPUT_DIR,
                f"helmet_detection_raw_{timestamp}.mp4"
            )
        )

        final_output_path = os.path.abspath(
            os.path.join(
                VIDEO_OUTPUT_DIR,
                f"helmet_detection_output_{timestamp}.mp4"
            )
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            raw_output_path,
            fourcc,
            output_fps,
            (output_width, output_height)
        )

        if writer is None or not writer.isOpened():
            yield make_base_result(
                success=False,
                final=True,
                confidence=conf,
                message="Gagal membuat video output. Codec video OpenCV tidak tersedia."
            )
            return

        total_helmet = 0
        total_no_helmet = 0
        frame_count = 0
        processed_frame_count = 0

        last_model_fps = 0
        last_latency_ms = 0
        fps_history = deque(maxlen=30)

        process_start = time.perf_counter()

        yield make_base_result(
            success=True,
            final=False,
            confidence=conf,
            message="Video mulai diproses...",
            metrics=make_video_metrics(
                process_start,
                processed_frame_count,
                total_frames_estimate,
                last_model_fps,
                last_latency_ms,
                fps_history
            )
        )

        while True:
            ret, frame_bgr = cap.read()

            if not ret:
                break

            frame_count += 1

            if (frame_count - 1) % FRAME_SKIP != 0:
                continue

            frame_bgr = cv2.resize(
                frame_bgr,
                (output_width, output_height)
            )

            model_start = time.perf_counter()

            detected_bgr, helmet_count, no_helmet_count = draw_detection(
                frame_bgr,
                conf
            )

            model_duration = time.perf_counter() - model_start
            last_latency_ms = round(model_duration * 1000, 1)
            last_model_fps = round(1 / model_duration, 2) if model_duration > 0 else 0

            total_helmet += helmet_count
            total_no_helmet += no_helmet_count
            processed_frame_count += 1

            fps_history.append(last_model_fps)

            writer.write(detected_bgr)

            if processed_frame_count % YIELD_EVERY_N_FRAMES == 0:
                compliance_rate, violation_rate = calculate_rates(
                    total_helmet,
                    total_no_helmet
                )

                metrics = make_video_metrics(
                    process_start,
                    processed_frame_count,
                    total_frames_estimate,
                    last_model_fps,
                    last_latency_ms,
                    fps_history
                )

                message = (
                    f"Sedang memproses video... "
                    f"Progress: {metrics['progress_percent']}% | "
                    f"Frame diproses: {processed_frame_count} | "
                    f"🟢 With Helmet: {total_helmet} | "
                    f"🔴 Without Helmet: {total_no_helmet} | "
                    f"✅ Compliance: {compliance_rate}% | "
                    f"⚠️ Violation: {violation_rate}%"
                )

                yield make_base_result(
                    success=True,
                    final=False,
                    confidence=conf,
                    with_helmet=total_helmet,
                    without_helmet=total_no_helmet,
                    message=message,
                    metrics=metrics
                )

        cap.release()
        writer.release()

        cap = None
        writer = None

        if processed_frame_count <= 0:
            yield make_base_result(
                success=False,
                final=True,
                confidence=conf,
                message="Tidak ada frame yang berhasil diproses."
            )
            return

        if not os.path.exists(raw_output_path) or os.path.getsize(raw_output_path) <= 0:
            yield make_base_result(
                success=False,
                final=True,
                confidence=conf,
                message="Video output gagal dibuat atau kosong."
            )
            return

        metrics_before_convert = make_video_metrics(
            process_start,
            processed_frame_count,
            total_frames_estimate,
            last_model_fps,
            last_latency_ms,
            fps_history
        )

        metrics_before_convert["status"] = "Converting Video"

        yield make_base_result(
            success=True,
            final=False,
            confidence=conf,
            with_helmet=total_helmet,
            without_helmet=total_no_helmet,
            message="Deteksi selesai. Video sedang dikonversi agar bisa dipreview di browser...",
            metrics=metrics_before_convert
        )

        browser_video_path = convert_to_browser_mp4(
            raw_output_path,
            final_output_path
        )

        if not os.path.exists(browser_video_path):
            browser_video_path = raw_output_path

        compliance_rate, violation_rate = calculate_rates(
            total_helmet,
            total_no_helmet
        )

        final_metrics = make_video_metrics(
            process_start,
            processed_frame_count,
            total_frames_estimate,
            last_model_fps,
            last_latency_ms,
            fps_history
        )

        final_metrics["progress_percent"] = 100
        final_metrics["status"] = "Completed"

        message = (
            f"Video selesai diproses. "
            f"Confidence: {conf:.2f} | "
            f"Total frame asli: {frame_count} dari estimasi {total_frames_estimate} | "
            f"Frame diproses: {processed_frame_count} | "
            f"Frame skip: {FRAME_SKIP} | "
            f"🟢 With Helmet: {total_helmet} | "
            f"🔴 Without Helmet: {total_no_helmet} | "
            f"✅ Compliance: {compliance_rate}% | "
            f"⚠️ Violation: {violation_rate}%"
        )

        yield make_base_result(
            success=True,
            final=True,
            video_path=browser_video_path,
            confidence=conf,
            with_helmet=total_helmet,
            without_helmet=total_no_helmet,
            message=message,
            metrics=final_metrics
        )

    except Exception as e:
        yield make_base_result(
            success=False,
            final=True,
            confidence=conf,
            message=f"Terjadi error saat memproses video: {str(e)}"
        )

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
# BACKWARD COMPATIBILITY
# =====================================================
def detect_video_core(video, conf_value):
    final_result = None

    for result in detect_video_core_progress(video, conf_value):
        final_result = result

    if final_result is None:
        final_result = make_base_result(
            success=False,
            final=True,
            confidence=conf_value,
            message="Video gagal diproses."
        )

    return final_result
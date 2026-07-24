import cv2
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Set, Tuple

import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# AYARLAR
# ============================================================

RTSP_URL = "<insert_rtsp_url>"
PERSON_MODEL_PATH = Path("models/yolo26m.pt")
ROI_FILE = Path("config/roi_coordinates.json")

# BoT-SORT takip algoritması
TRACKER = "botsort.yaml"

PERSON_CONFIDENCE = 0.80
PERSON_IOU_THRESHOLD = 0.70

# Aynı kişi sayısı bu kadar kare boyunca değişmezse sayaç onaylanır.
STABLE_COUNT_REQUIRED_FRAMES = 30

# Bir takip ID'si bu kadar kare görülmezse bellekten silinir.
TRACK_MAX_MISSING_FRAMES = 150

MAX_CONSECUTIVE_READ_FAILURES = 30
RECONNECT_DELAY_SECONDS = 5
SHOW_OUTSIDE_ROI = False
WINDOW_NAME = "Take-away Kisi Sayma ve Sira Takibi"

# Sıra numarası görüntüde yukarıdan aşağıya mı, aşağıdan yukarıya mı verilsin?
# Kamerada banko görüntünün üst tarafındaysa "top_to_bottom" uygundur.
QUEUE_DIRECTION = "top_to_bottom"  # veya "bottom_to_top"

# Yoğunluk eşikleri
LOW_DENSITY_MAX = 2
MEDIUM_DENSITY_MAX = 5


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_roi(
    roi_file: Path,
    current_width: int,
    current_height: int
) -> Optional[np.ndarray]:
    """ROI koordinatlarını yükler ve çözünürlük değişmişse ölçekler."""

    if not roi_file.exists():
        logger.error("ROI dosyası bulunamadı: %s", roi_file.resolve())
        return None

    try:
        with roi_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        saved_width = int(data["frame_width"])
        saved_height = int(data["frame_height"])
        points = data["points"]

        if saved_width <= 0 or saved_height <= 0 or len(points) != 4:
            logger.error("ROI dosyasındaki bilgiler geçersiz.")
            return None

        scale_x = current_width / saved_width
        scale_y = current_height / saved_height

        scaled_points = []
        for x, y in points:
            scaled_x = int(round(x * scale_x))
            scaled_y = int(round(y * scale_y))
            scaled_x = max(0, min(scaled_x, current_width - 1))
            scaled_y = max(0, min(scaled_y, current_height - 1))
            scaled_points.append([scaled_x, scaled_y])

        roi = np.asarray(scaled_points, dtype=np.int32)
        logger.info("ROI yüklendi: %s", roi.tolist())
        return roi

    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        logger.exception("ROI yüklenemedi: %s", error)
        return None


def is_inside_roi(point: Tuple[int, int], roi_polygon: np.ndarray) -> bool:
    """Alt-orta noktanın ROI içinde olup olmadığını kontrol eder."""
    return cv2.pointPolygonTest(
        roi_polygon,
        (float(point[0]), float(point[1])),
        False
    ) >= 0


class StableCountManager:
    """Kişi sayısını belirli kare boyunca sabit kaldıktan sonra onaylar."""

    def __init__(self, required_frames: int) -> None:
        self.required_frames = required_frames
        self.candidate_count: Optional[int] = None
        self.candidate_frames = 0
        self.confirmed_count = 0

    def update(self, current_count: int) -> Tuple[int, int]:
        """Onaylı sayıyı ve doğrulama ilerlemesini döndürür."""

        if self.candidate_count == current_count:
            self.candidate_frames += 1
        else:
            self.candidate_count = current_count
            self.candidate_frames = 1

        if self.candidate_frames >= self.required_frames:
            self.confirmed_count = current_count

        return self.confirmed_count, min(
            self.candidate_frames,
            self.required_frames
        )


class TrackMemory:
    """Takip ID'lerinin son görüldüğü kareyi ve toplam ziyaretçileri saklar."""

    def __init__(self, max_missing_frames: int) -> None:
        self.max_missing_frames = max_missing_frames
        self.last_seen: Dict[int, int] = {}
        self.unique_inside_ids: Set[int] = set()

    def mark_inside(self, track_id: int, frame_number: int) -> None:
        self.last_seen[track_id] = frame_number
        self.unique_inside_ids.add(track_id)

    def cleanup(self, frame_number: int) -> None:
        expired = [
            track_id
            for track_id, last_frame in self.last_seen.items()
            if frame_number - last_frame > self.max_missing_frames
        ]
        for track_id in expired:
            self.last_seen.pop(track_id, None)


def connect_rtsp(rtsp_url: str) -> Optional[cv2.VideoCapture]:
    """RTSP akışına bağlanır."""

    logger.info("RTSP bağlantısı kuruluyor...")
    capture = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not capture.isOpened():
        logger.error("RTSP bağlantısı açılamadı.")
        capture.release()
        return None

    logger.info("RTSP bağlantısı başarılı.")
    return capture


def get_density_label(count: int) -> Tuple[str, Tuple[int, int, int]]:
    """Kişi sayısına göre yoğunluk etiketi ve rengini döndürür."""

    if count <= LOW_DENSITY_MAX:
        return "Dusuk", (0, 255, 0)
    if count <= MEDIUM_DENSITY_MAX:
        return "Orta", (0, 255, 255)
    return "Yuksek", (0, 0, 255)


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    background_color: Tuple[int, int, int],
    font_scale: float = 0.6
) -> None:
    """Okunabilir arka planlı metin çizer."""

    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    y = max(y, text_height + baseline + 5)

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 6),
        (x + text_width + 8, y + 4),
        background_color,
        -1
    )
    cv2.putText(
        frame,
        text,
        (x + 4, y - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def main() -> None:
    """Take-away bankosu için canlı kişi sayma ve sıra takibini başlatır."""

    if not PERSON_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"İnsan tespit modeli bulunamadı: {PERSON_MODEL_PATH.resolve()}"
        )

    if not ROI_FILE.exists():
        raise FileNotFoundError(
            f"ROI dosyası bulunamadı: {ROI_FILE.resolve()}\n"
            "Önce ROI seçim kodunu çalıştır."
        )

    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info("Kullanılan cihaz: %s", "CUDA GPU" if device == "0" else "CPU")

    model = YOLO(str(PERSON_MODEL_PATH))
    logger.info("Model görevi: %s", model.task)

    stable_counter = StableCountManager(STABLE_COUNT_REQUIRED_FRAMES)
    track_memory = TrackMemory(TRACK_MAX_MISSING_FRAMES)

    capture: Optional[cv2.VideoCapture] = None
    roi_polygon: Optional[np.ndarray] = None

    frame_number = 0
    failed_reads = 0
    fps_history: Deque[float] = deque(maxlen=20)
    previous_time = time.perf_counter()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if capture is None or not capture.isOpened():
                capture = connect_rtsp(RTSP_URL)
                if capture is None:
                    time.sleep(RECONNECT_DELAY_SECONDS)
                    continue

                failed_reads = 0
                roi_polygon = None

            success, frame = capture.read()

            if not success or frame is None:
                failed_reads += 1
                logger.warning(
                    "Kare okunamadı: %d/%d",
                    failed_reads,
                    MAX_CONSECUTIVE_READ_FAILURES
                )

                if failed_reads >= MAX_CONSECUTIVE_READ_FAILURES:
                    capture.release()
                    capture = None
                    time.sleep(RECONNECT_DELAY_SECONDS)
                continue

            failed_reads = 0
            frame_number += 1
            frame_height, frame_width = frame.shape[:2]

            if roi_polygon is None:
                roi_polygon = load_roi(ROI_FILE, frame_width, frame_height)
                if roi_polygon is None:
                    break

            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=TRACKER,
                    classes=[0],
                    conf=PERSON_CONFIDENCE,
                    iou=PERSON_IOU_THRESHOLD,
                    device=device,
                    verbose=False
                )
            except Exception as error:
                logger.exception("Tespit/takip hatası: %s", error)
                continue

            cv2.polylines(
                frame,
                [roi_polygon],
                isClosed=True,
                color=(255, 255, 0),
                thickness=3
            )

            inside_people = []

            if results:
                boxes = results[0].boxes

                if boxes is not None and len(boxes) > 0:
                    xyxy_values = boxes.xyxy.cpu().numpy()
                    confidences = boxes.conf.cpu().numpy()

                    if boxes.id is not None:
                        track_ids = boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [-1] * len(xyxy_values)

                    for xyxy, confidence, track_id in zip(
                        xyxy_values,
                        confidences,
                        track_ids
                    ):
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1 = max(0, min(x1, frame_width - 1))
                        y1 = max(0, min(y1, frame_height - 1))
                        x2 = max(0, min(x2, frame_width))
                        y2 = max(0, min(y2, frame_height))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        foot_x = (x1 + x2) // 2
                        foot_y = y2 - 1
                        inside = is_inside_roi((foot_x, foot_y), roi_polygon)

                        if not inside:
                            if SHOW_OUTSIDE_ROI:
                                cv2.rectangle(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (128, 128, 128),
                                    2
                                )
                            continue

                        if track_id >= 0:
                            track_memory.mark_inside(track_id, frame_number)

                        inside_people.append({
                            "track_id": track_id,
                            "bbox": (x1, y1, x2, y2),
                            "foot": (foot_x, foot_y),
                            "confidence": float(confidence)
                        })

            # Bankoya en yakın kişiden başlayarak sıra numarası ver.
            reverse_sort = QUEUE_DIRECTION == "bottom_to_top"
            inside_people.sort(
                key=lambda person: person["foot"][1],
                reverse=reverse_sort
            )

            for queue_index, person in enumerate(inside_people, start=1):
                x1, y1, x2, y2 = person["bbox"]
                foot_x, foot_y = person["foot"]
                track_id = person["track_id"]
                confidence = person["confidence"]

                color = (0, 180, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (foot_x, foot_y), 5, color, -1)

                id_text = str(track_id) if track_id >= 0 else "Bekleniyor"
                label = (
                    f"Sira: {queue_index} | ID: {id_text} | "
                    f"Kisi {confidence:.2f}"
                )
                draw_text_with_background(frame, label, (x1, y1), color)

            current_count = len(inside_people)
            confirmed_count, stable_progress = stable_counter.update(current_count)
            track_memory.cleanup(frame_number)

            density_label, density_color = get_density_label(confirmed_count)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            if elapsed > 0:
                fps_history.append(1.0 / elapsed)
            average_fps = (
                sum(fps_history) / len(fps_history)
                if fps_history else 0.0
            )

            info_lines = [
                f"Anlik kisi: {current_count}",
                f"Onayli kisi: {confirmed_count}",
                f"Dogrulama: {stable_progress}/{STABLE_COUNT_REQUIRED_FRAMES}",
                f"Yogunluk: {density_label}",
                f"Toplam farkli ID: {len(track_memory.unique_inside_ids)}",
                f"FPS: {average_fps:.1f}",
                f"Tracker: BoT-SORT"
            ]

            for index, text in enumerate(info_lines):
                color = density_color if text.startswith("Yogunluk") else (40, 40, 40)
                draw_text_with_background(
                    frame,
                    text,
                    (15, 30 + index * 32),
                    color,
                    font_scale=0.65
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                logger.info("Program kullanıcı tarafından kapatıldı.")
                break

    except KeyboardInterrupt:
        logger.info("Program Ctrl+C ile durduruldu.")

    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()
        logger.info("RTSP bağlantısı ve pencereler kapatıldı.")


if __name__ == "__main__":
    main()

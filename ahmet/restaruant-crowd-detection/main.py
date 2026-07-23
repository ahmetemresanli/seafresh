import json
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO


# =========================================================
# SETTINGS
# =========================================================

RTSP_URL = "<insert_rtsp_url>" #insert rtsp url

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolo26m.pt"
)

ROI_FILE = os.path.join(
    BASE_DIR,
    "roi.json"
)

WINDOW_NAME = "Person Counting System"

PERSON_CLASS_ID = 0

CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50

DEVICE = 0

# CPU kullanmak için:
# DEVICE = "cpu"

DISPLAY_WIDTH = 1280
RECONNECT_DELAY = 3

# Tespit kutusunun en az yüzde kaçı ROI içerisindeyse
# o tespit tamamen yok sayılacak.
ROI_OVERLAP_THRESHOLD = 0.20


# =========================================================
# FRAME FUNCTIONS
# =========================================================

def resize_frame(frame, target_width):
    if target_width is None:
        return frame

    height, width = frame.shape[:2]

    if width == target_width:
        return frame

    scale = target_width / width
    target_height = int(height * scale)

    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_LINEAR
    )


# =========================================================
# ROI FUNCTIONS
# =========================================================

def load_rois():
    if not os.path.exists(ROI_FILE):
        raise FileNotFoundError(
            "roi.json was not found. Run roi_select.py first."
        )

    with open(ROI_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    roi_data = data.get("rois", [])

    if len(roi_data) < 1:
        raise ValueError(
            "At least one ROI is required."
        )

    roi_polygons = []

    for index, roi_item in enumerate(roi_data):
        points = roi_item.get("points", [])

        if len(points) < 3:
            raise ValueError(
                f"ROI {index + 1} must contain at least 3 points."
            )

        polygon = np.array(
            points,
            dtype=np.int32
        )

        roi_polygons.append(polygon)

    return roi_polygons


def mask_excluded_rois(frame, roi_polygons):
    """
    Bütün excluded ROI alanlarını siyaha kapatır.
    YOLO bu alanları görmez.
    """
    masked_frame = frame.copy()

    cv2.fillPoly(
        masked_frame,
        roi_polygons,
        color=(0, 0, 0)
    )

    return masked_frame


def is_point_inside_any_roi(
    point,
    roi_polygons
):
    for polygon in roi_polygons:
        result = cv2.pointPolygonTest(
            polygon,
            point,
            False
        )

        if result >= 0:
            return True

    return False


def detection_overlaps_single_roi(
    box,
    roi_polygon,
    frame_shape
):
    frame_height, frame_width = frame_shape[:2]

    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    if x2 <= x1 or y2 <= y1:
        return True

    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height

    if box_area <= 0:
        return True

    local_mask = np.zeros(
        (box_height, box_width),
        dtype=np.uint8
    )

    shifted_polygon = roi_polygon.copy()

    shifted_polygon[:, 0] -= x1
    shifted_polygon[:, 1] -= y1

    cv2.fillPoly(
        local_mask,
        [shifted_polygon],
        color=255
    )

    overlap_pixels = cv2.countNonZero(
        local_mask
    )

    overlap_ratio = overlap_pixels / box_area

    return overlap_ratio >= ROI_OVERLAP_THRESHOLD


def detection_overlaps_any_roi(
    box,
    roi_polygons,
    frame_shape
):
    for polygon in roi_polygons:
        if detection_overlaps_single_roi(
            box,
            polygon,
            frame_shape
        ):
            return True

    return False


def draw_excluded_rois(
    frame,
    roi_polygons
):
    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        roi_polygons,
        color=(0, 0, 0)
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame
    )

    for index, polygon in enumerate(roi_polygons):
        cv2.polylines(
            frame,
            [polygon],
            isClosed=True,
            color=(0, 0, 255),
            thickness=3
        )

        first_point = polygon[0]

        label_x = max(
            10,
            int(first_point[0])
        )

        label_y = max(
            30,
            int(first_point[1]) - 10
        )

        draw_text_with_background(
            frame=frame,
            text=f"Excluded Area {index + 1}",
            position=(label_x, label_y),
            background_color=(0, 0, 180),
            font_scale=0.55
        )


# =========================================================
# RTSP FUNCTIONS
# =========================================================

def open_rtsp_stream():
    print("Connecting to RTSP stream...")

    capture = cv2.VideoCapture(
        RTSP_URL,
        cv2.CAP_FFMPEG
    )

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not capture.isOpened():
        print("RTSP connection failed.")
        return None

    print("RTSP connection established.")

    return capture


def reconnect_rtsp(capture):
    if capture is not None:
        capture.release()

    print("Reconnecting to RTSP stream...")

    while True:
        time.sleep(RECONNECT_DELAY)

        new_capture = open_rtsp_stream()

        if new_capture is not None:
            return new_capture

        print(
            f"Retrying in {RECONNECT_DELAY} seconds..."
        )


# =========================================================
# DISPLAY FUNCTIONS
# =========================================================

def draw_text_with_background(
    frame,
    text,
    position,
    text_color=(255, 255, 255),
    background_color=(0, 0, 0),
    font_scale=0.70,
    thickness=2,
    padding=6
):
    font = cv2.FONT_HERSHEY_SIMPLEX

    x, y = position

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (
            max(0, x - padding),
            max(0, y - text_height - padding)
        ),
        (
            min(frame.shape[1] - 1, x + text_width + padding),
            min(frame.shape[0] - 1, y + baseline + padding)
        ),
        background_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# TRACKING FUNCTIONS
# =========================================================

def process_tracking_result(
    frame,
    result,
    roi_polygons
):
    current_person_ids = set()

    if result.boxes is None:
        return current_person_ids

    if result.boxes.id is None:
        return current_person_ids

    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()

    for box, track_id, class_id, confidence in zip(
        boxes,
        track_ids,
        class_ids,
        confidences
    ):
        if class_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box)

        center_point = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        foot_point = (
            int((x1 + x2) / 2),
            int(y2)
        )

        # Merkez herhangi bir ROI içindeyse yok say.
        if is_point_inside_any_roi(
            center_point,
            roi_polygons
        ):
            continue

        # Ayak noktası herhangi bir ROI içindeyse yok say.
        if is_point_inside_any_roi(
            foot_point,
            roi_polygons
        ):
            continue

        # Kutu herhangi bir ROI ile belirgin şekilde çakışıyorsa yok say.
        if detection_overlaps_any_roi(
            box,
            roi_polygons,
            frame.shape
        ):
            continue

        current_person_ids.add(
            track_id
        )

        box_color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        cv2.circle(
            frame,
            foot_point,
            5,
            box_color,
            -1
        )

        label = (
            f"ID: {track_id} | "
            f"Person | "
            f"{confidence:.2f}"
        )

        label_y = y1 - 10

        if label_y < 25:
            label_y = y1 + 25

        draw_text_with_background(
            frame=frame,
            text=label,
            position=(x1, label_y),
            background_color=box_color,
            font_scale=0.52,
            thickness=2
        )

    return current_person_ids


# =========================================================
# MAIN
# =========================================================

def main():
    maximum_person_count = 0

    try:
        roi_polygons = load_rois()

    except FileNotFoundError as error:
        print(f"ROI error: {error}")
        return

    except json.JSONDecodeError:
        print("ROI error: roi.json is not valid.")
        return

    except ValueError as error:
        print(f"ROI error: {error}")
        return

    print(
        f"{len(roi_polygons)} excluded ROI areas loaded."
    )

    if not os.path.exists(MODEL_PATH):
        print("Model file was not found:")
        print(MODEL_PATH)
        return

    print("Loading YOLO26m model...")

    try:
        model = YOLO(
            MODEL_PATH
        )

    except Exception as error:
        print(f"Model loading error: {error}")
        return

    print("Model loaded successfully.")

    capture = open_rtsp_stream()

    if capture is None:
        capture = reconnect_rtsp(
            capture
        )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    print("")
    print("Person counting started.")
    print("Q or ESC : Quit")
    print("C        : Reset maximum count")
    print("")

    while True:
        success, frame = capture.read()

        if not success or frame is None:
            print("Camera frame could not be read.")

            capture = reconnect_rtsp(
                capture
            )

            continue

        frame = resize_frame(
            frame,
            DISPLAY_WIDTH
        )

        # İki ROI alanını da YOLO görüntüsünde kapat.
        inference_frame = mask_excluded_rois(
            frame,
            roi_polygons
        )

        try:
            results = model.track(
                source=inference_frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[PERSON_CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False
            )

        except Exception as error:
            print(f"Tracking error: {error}")
            continue

        current_person_ids = set()

        if results:
            current_person_ids = process_tracking_result(
                frame=frame,
                result=results[0],
                roi_polygons=roi_polygons
            )

        current_person_count = len(
            current_person_ids
        )

        if current_person_count > maximum_person_count:
            maximum_person_count = current_person_count

        draw_excluded_rois(
            frame,
            roi_polygons
        )

        draw_text_with_background(
            frame=frame,
            text=(
                f"Current Person Count: "
                f"{current_person_count}"
            ),
            position=(20, 40),
            background_color=(0, 130, 0),
            font_scale=0.75
        )

        draw_text_with_background(
            frame=frame,
            text=(
                f"Maximum Person Count: "
                f"{maximum_person_count}"
            ),
            position=(20, 82),
            background_color=(180, 100, 0),
            font_scale=0.75
        )

        draw_text_with_background(
            frame=frame,
            text="Q: Quit | C: Reset Maximum Count",
            position=(20, frame.shape[0] - 20),
            background_color=(0, 0, 0),
            font_scale=0.55
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q"),
            27
        ):
            break

        elif key in (
            ord("c"),
            ord("C")
        ):
            maximum_person_count = current_person_count
            print("Maximum person count reset.")

    capture.release()
    cv2.destroyAllWindows()

    print("Program stopped.")


if __name__ == "__main__":
    main()

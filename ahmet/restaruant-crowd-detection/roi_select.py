import json
import os

import cv2
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

RTSP_URL = "<insert_rtsp_url>"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROI_FILE = os.path.join(
    BASE_DIR,
    "roi.json"
)

WINDOW_NAME = "ROI Selection"

DISPLAY_WIDTH = 1280

TOTAL_ROI_COUNT = 2
MINIMUM_POINTS = 3


# =========================================================
# GLOBAL VARIABLES
# =========================================================

completed_rois = []
current_roi_points = []


# =========================================================
# HELPER FUNCTIONS
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


def draw_text_with_background(
    frame,
    text,
    position,
    background_color=(0, 0, 0),
    text_color=(255, 255, 255),
    font_scale=0.65,
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


def save_rois(rois):
    data = {
        "rois": [
            {
                "points": [
                    [int(x), int(y)]
                    for x, y in roi
                ]
            }
            for roi in rois
        ]
    }

    with open(ROI_FILE, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4
        )

    print(f"ROI file saved: {ROI_FILE}")


def mouse_callback(event, x, y, flags, param):
    global current_roi_points

    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi_points.append((x, y))
        print(f"Point added: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_roi_points:
            removed_point = current_roi_points.pop()
            print(f"Point removed: {removed_point}")


def draw_polygon(
    frame,
    points,
    color,
    fill_alpha=0.20
):
    if len(points) < 3:
        return

    polygon = np.array(
        points,
        dtype=np.int32
    )

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        color
    )

    cv2.addWeighted(
        overlay,
        fill_alpha,
        frame,
        1.0 - fill_alpha,
        0,
        frame
    )

    cv2.polylines(
        frame,
        [polygon],
        isClosed=True,
        color=color,
        thickness=3
    )


# =========================================================
# MAIN
# =========================================================

def main():
    global completed_rois
    global current_roi_points

    completed_rois = []
    current_roi_points = []

    print("Connecting to RTSP camera...")

    capture = cv2.VideoCapture(
        RTSP_URL,
        cv2.CAP_FFMPEG
    )

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not capture.isOpened():
        print("Could not connect to RTSP camera.")
        return

    success, frame = capture.read()

    capture.release()

    if not success or frame is None:
        print("Could not read camera frame.")
        return

    frame = resize_frame(
        frame,
        DISPLAY_WIDTH
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    print("")
    print("ROI selection started.")
    print("Left Click  : Add point")
    print("Right Click : Remove last point")
    print("ENTER       : Complete current ROI")
    print("R           : Reset current ROI")
    print("B           : Go back to previous ROI")
    print("Q or ESC    : Quit")
    print("")

    while True:
        display_frame = frame.copy()

        # Previously completed ROIs
        for index, roi in enumerate(completed_rois):
            draw_polygon(
                display_frame,
                roi,
                color=(0, 0, 255)
            )

            first_point = roi[0]

            draw_text_with_background(
                display_frame,
                f"Excluded ROI {index + 1}",
                (
                    max(10, first_point[0]),
                    max(30, first_point[1] - 10)
                ),
                background_color=(0, 0, 180),
                font_scale=0.55
            )

        # Current ROI
        draw_polygon(
            display_frame,
            current_roi_points,
            color=(0, 255, 255)
        )

        for index, point in enumerate(current_roi_points):
            cv2.circle(
                display_frame,
                point,
                6,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                display_frame,
                str(index + 1),
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        current_roi_number = len(completed_rois) + 1

        draw_text_with_background(
            display_frame,
            (
                f"Select Excluded ROI "
                f"{current_roi_number}/{TOTAL_ROI_COUNT}"
            ),
            (20, 35)
        )

        draw_text_with_background(
            display_frame,
            "Left Click: Add | Right Click: Undo",
            (20, 70)
        )

        draw_text_with_background(
            display_frame,
            "ENTER: Complete ROI | R: Reset",
            (20, 105)
        )

        draw_text_with_background(
            display_frame,
            "B: Previous ROI | Q: Quit",
            (20, 140)
        )

        cv2.imshow(
            WINDOW_NAME,
            display_frame
        )

        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            if len(current_roi_points) < MINIMUM_POINTS:
                print(
                    f"At least {MINIMUM_POINTS} points are required."
                )
                continue

            completed_rois.append(
                current_roi_points.copy()
            )

            print(
                f"ROI {len(completed_rois)} completed."
            )

            current_roi_points = []

            if len(completed_rois) == TOTAL_ROI_COUNT:
                save_rois(completed_rois)
                print("All ROIs completed.")
                break

        elif key in (ord("r"), ord("R")):
            current_roi_points = []
            print("Current ROI reset.")

        elif key in (ord("b"), ord("B")):
            if completed_rois:
                current_roi_points = completed_rois.pop()
                print("Returned to previous ROI.")

        elif key in (ord("q"), ord("Q"), 27):
            print("ROI selection cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

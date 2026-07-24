import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple

# ==================================================
# AYARLAR
# ==================================================

RTSP_URL = "rtsp://your5eyt:rfg34hg-6he@77.44.64.69:554/cam/realmonitor?channel=7&subtype=0"

ROI_FILE = Path("config/roi_coordinates.json")

WINDOW_NAME = "ROI Selection"

selected_points: List[Tuple[int, int]] = []


# ==================================================
# MOUSE CALLBACK
# ==================================================

def mouse_callback(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        selected_points.append((x, y))

        print(f"{len(selected_points)}. nokta: ({x}, {y})")


# ==================================================
# ROI ÇİZ
# ==================================================

def draw_roi_selection(frame):

    display = frame.copy()

    # Noktalar
    for index, point in enumerate(selected_points):

        cv2.circle(
            display,
            point,
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display,
            str(index + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    # Polygon çiz
    if len(selected_points) >= 2:

        pts = np.array(selected_points, dtype=np.int32)

        cv2.polylines(
            display,
            [pts],
            False,
            (0, 255, 255),
            2
        )

    # Polygonu kapat ve içini boya
    if len(selected_points) >= 3:

        pts = np.array(selected_points, dtype=np.int32)

        overlay = display.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 255, 0)
        )

        display = cv2.addWeighted(
            overlay,
            0.25,
            display,
            0.75,
            0
        )

        cv2.polylines(
            display,
            [pts],
            True,
            (0, 255, 0),
            2
        )

    cv2.putText(
        display,
        "Left Click: Add Point",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        "Z: Undo   R: Reset   S: Save   Q: Quit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Selected Points : {len(selected_points)}",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return display


# ==================================================
# ROI KAYDET
# ==================================================

def save_roi(frame_width, frame_height):

    ROI_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    roi_data = {

        "frame_width": frame_width,

        "frame_height": frame_height,

        "points": [
            [x, y]
            for x, y in selected_points
        ]

    }

    with ROI_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            roi_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()

    print("ROI başarıyla kaydedildi.")

    print(ROI_FILE.resolve())


# ==================================================
# İLK FRAME
# ==================================================

def get_first_frame():

    print("RTSP bağlantısı kuruluyor...")

    cap = cv2.VideoCapture(RTSP_URL)

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    if not cap.isOpened():

        print("RTSP açılamadı.")

        return None

    frame = None

    for _ in range(100):

        success, current_frame = cap.read()

        if success:

            frame = current_frame

            break

    cap.release()

    if frame is None:

        print("Frame alınamadı.")

        return None

    print("İlk frame alındı.")

    return frame


# ==================================================
# MAIN
# ==================================================

def main():

    frame = get_first_frame()

    if frame is None:

        return

    frame_height, frame_width = frame.shape[:2]

    print(f"Resolution : {frame_width} x {frame_height}")

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    while True:

        display = draw_roi_selection(frame)

        cv2.imshow(
            WINDOW_NAME,
            display
        )

        key = cv2.waitKey(20) & 0xFF

        # Undo
        if key == ord("z"):

            if selected_points:

                removed = selected_points.pop()

                print(f"Silindi : {removed}")

        # Reset
        elif key == ord("r"):

            selected_points.clear()

            print("ROI sıfırlandı.")

        # Save
        elif key == ord("s"):

            if len(selected_points) < 3:

                print("En az 3 nokta seçmelisin.")

                continue

            save_roi(
                frame_width,
                frame_height
            )

            break

        # Quit
        elif key == ord("q") or key == 27:

            print("ROI kaydedilmedi.")

            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
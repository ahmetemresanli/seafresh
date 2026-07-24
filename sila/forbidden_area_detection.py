import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


class ForbiddenAreaDetector:
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        config_path: str = "config/roi_coordinates.json",
        rtsp_url: str = "rtsp://your5eyt:rfg34hg-6he@77.44.64.69:554/cam/playback?channel=7&subtype=0&starttime=2026_07_22_18_30_00&endtime=2026_07_22_21_00_00"
    ):
        base_dir = Path(__file__).resolve().parent

        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            candidate = (base_dir / self.model_path).resolve()
            if candidate.exists():
                self.model_path = candidate

        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            candidate = (base_dir / self.config_path).resolve()
            if candidate.exists():
                self.config_path = candidate

        self.model = YOLO(str(self.model_path))
        self.rtsp_url = rtsp_url
        self.roi_polygon = self._load_roi()

    def _load_roi(self) -> np.ndarray:
        """JSON dosyasından ROI koordinatlarını okur ve NumPy formatına çevirir."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = data.get("points", [])
        if len(points) < 3:
            raise ValueError("Geçerli bir ROI için en az 3 nokta gereklidir.")

        return np.array(points, dtype=np.int32)

    def is_inside_roi(self, point: tuple) -> bool:
        """Verilen (x, y) noktasının ROI alanının içinde olup olmadığını kontrol eder."""
        # > 0: İçinde, 0: Kenarında, < 0: Dışında
        result = cv2.pointPolygonTest(self.roi_polygon, (float(point[0]), float(point[1])), False)
        return result >= 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Görüntü üzerinde YOLO tespiti ve ROI kontrolü yapar."""
        results = self.model(frame, verbose=False, classes=[0])  # Sadece person (class 0) tespiti

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Kişinin ayak noktası (alt tabanın ortası)
                foot_point = ((x1 + x2) // 2, y2)

                # ROI Dışındaki kişileri tamamen görmezden gel
                if not self.is_inside_roi(foot_point):
                    continue

                # ROI içindeyse kırmızı kutu çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # Etiket ve metin ekle
                label = "PERSON IN FORBIDDEN AREA"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        # Görselleştirmek için ROI alanını da hafif çizebiliriz (isteğe bağlı)
        cv2.polylines(frame, [self.roi_polygon], True, (255, 255, 0), 2)
        return frame

    def run(self):
        """Kamera akışını başlatır ve ana döngüyü çalıştırır."""
        cap = cv2.VideoCapture(self.rtsp_url)

        if not cap.isOpened():
            print("RTSP bağlantısı kurulamadı.")
            return

        print("RTSP bağlantısı başarılı. Tespit başlatılıyor...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame okunamadı veya akış sona erdi.")
                break

            processed_frame = self.process_frame(frame)

            cv2.imshow("Forbidden Area Detection", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ForbiddenAreaDetector()
    detector.run()
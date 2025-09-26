# ===============================
# YOLOv5 Inference - Resize to Half + FPS
# ===============================

import cv2
import time
from ultralytics import YOLO

# 1. Load model hasil training
model_path = "yolov5_results/yolov5_helmet_train42.pt"
model = YOLO(model_path)

# 2. Path ke file video
video_path = "trafic.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Gagal membuka video: {video_path}")
    exit()

# 3. Ambil ukuran asli video
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 4. Hitung ukuran setengah
resize_width = original_width // 2
resize_height = original_height // 2

# 5. Loop frame demi frame
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (resize_width, resize_height))

    results = model.predict(source=resized_frame, conf=0.4, stream=True, device='cpu', verbose=False)

    for result in results:
        annotated_frame = result.plot()

        # Hitung FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)

        # Tambahkan teks FPS ke frame
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("YOLOv5 Helmet Detection - Half Size", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
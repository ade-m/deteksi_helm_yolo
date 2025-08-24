# ===============================
# YOLOv5 Inference - Webcam
# ===============================

import cv2
from ultralytics import YOLO

# 1. Load model hasil training
model_path = "yolov5_results/yolov5_helmet_train42.pt"   # pastikan sudah ada model hasil training
model = YOLO(model_path)

# 2. Buka webcam (0 = default webcam, kalau pakai kamera eksternal bisa coba 1 atau 2)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Tidak bisa membuka webcam")
    exit()

# 3. Loop untuk baca frame webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame dari webcam")
        break

    # 4. Inference dengan YOLO
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # 5. Ambil frame hasil dengan bounding box
    annotated_frame = results[0].plot()

    # 6. Tampilkan ke layar
    cv2.imshow("YOLOv5 Helmet Detection - Webcam", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Bersihkan resource
cap.release()
cv2.destroyAllWindows()

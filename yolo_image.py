# ===============================
# YOLOv5 Inference - Image
# ===============================

import cv2
from ultralytics import YOLO

# 1. Load model hasil training
model_path = "yolov5_results/yolov5_helmet_train42.pt"   # pastikan sudah ada model hasil training
model = YOLO(model_path)

# 2. Path ke file gambar
image_path = "sample.jpg"  # ganti sesuai file yang mau diuji

# 3. Load gambar
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Gagal membuka gambar: {image_path}")
    exit()

# 4. Inference dengan YOLO
results = model.predict(source=image, conf=0.5, verbose=False)

# 5. Ambil gambar hasil dengan bounding box
annotated_image = results[0].plot()

# 6. Tampilkan ke layar
cv2.imshow("YOLOv5 Helmet Detection - Image", annotated_image)
cv2.waitKey(0)  # tunggu sampai tombol ditekan
cv2.destroyAllWindows()

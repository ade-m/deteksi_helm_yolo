# ===============================
# YOLOv5 Training - Lokal Komputer
# ===============================

# 1. Install YOLOv5 (ultralytics) jika belum
# Buka terminal/cmd, jalankan:
# pip install ultralytics --upgrade

# 2. Import library
from ultralytics import YOLO
import os

# 3. Tentukan path dataset dan yaml
# Contoh struktur folder lokal:
# dataset/
# ├── images/
# │   ├── train/
# │   └── val/
# └── labels/
# yaml_path = "dataset/rider_helmet.yaml"
yaml_path = "dataset/rider_helmet.yaml"

# 4. Load pretrained YOLOv5 model
model = YOLO("yolov5s.pt")  # pastikan file yolov5s.pt ada di folder kerja atau download dari https://github.com/ultralytics/yolov5/releases

# 5. Train model
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="yolov5_helmet_train",
    project="yolov5_results"  # akan membuat folder di direktori kerja lokal
)

# 6. Evaluasi hasil training
metrics = model.val()
print(metrics)

# 7. Export model
model.export(format="pt")
print("Model tersimpan di folder yolov5_results/yolov5_helmet_train/weights")

# 8. Inference contoh
val_image = "dataset/images/val/sample.jpg"  # ganti sesuai file gambar lokal
results = model.predict(val_image)
results.show()  # menampilkan bounding box di gambar

# ===============================
# YOLOv8s Training - Lokal Komputer (CPU)
# ===============================

import os
from collections import Counter
from ultralytics import YOLO

# 1. Cek distribusi data per class
label_dir = "dataset/train/labels"
class_names = ["with helmet", "without helmet", "rider", "number plate"]

if not os.path.exists(label_dir):
    print(f"❌ Folder label tidak ditemukan: {label_dir}")
    exit()

class_counts = Counter()
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(label_dir, filename), "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                class_counts[int(class_id)] += 1

total_annotations = sum(class_counts.values())
print("\n📊 Distribusi Anotasi per Class:")
for class_id, count in sorted(class_counts.items()):
    percentage = (count / total_annotations) * 100
    print(f"- {class_names[class_id]} (class {class_id}): {count} anotasi ({percentage:.2f}%)")
print(f"\n🔢 Total semua anotasi: {total_annotations}")

# 2. Path ke YAML dataset
yaml_path = "dataset/rider_helmet.yaml"

# 3. Load model YOLOv8s
model = YOLO("yolov8s.pt")  # pastikan file ini ada di folder kerja

# 4. Training di CPU
results = model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=8,               # lebih ringan untuk Yoga 7i
    device='cpu',          # pakai CPU karena tidak ada CUDA
    name="yolov8s_helmet_train",
    project="yolov8_results"
)

# 5. Evaluasi hasil training
metrics = model.val()
print("\n📈 Hasil Evaluasi:")
print(metrics)

# 6. Simpan model hasil training
model.save("yolov8_results/yolov8s_helmet_final.pt")

# 7. Inference contoh gambar
val_image = "dataset/images/val/sample.jpg"
results = model.predict(val_image, conf=0.4, device='cpu')
results.show()
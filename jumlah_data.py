import os
from collections import Counter

# Path ke folder label training
label_dir = "dataset/train/labels"

# Cek apakah folder ada
if not os.path.exists(label_dir):
    print(f"❌ Folder tidak ditemukan: {label_dir}")
    exit()

# Inisialisasi counter
class_counts = Counter()

# Loop semua file label
for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(label_dir, filename), "r") as f:
            for line in f:
                class_id = line.strip().split()[0]
                class_counts[int(class_id)] += 1

# Daftar nama class dari YAML
class_names = ["with helmet", "without helmet", "rider", "number plate"]

# Total anotasi
total_annotations = sum(class_counts.values())

# Tampilkan hasil
print(f"\n📊 Distribusi Anotasi per Class:")
for class_id, count in sorted(class_counts.items()):
    percentage = (count / total_annotations) * 100
    print(f"- {class_names[class_id]} (class {class_id}): {count} anotasi ({percentage:.2f}%)")

print(f"\n🔢 Total semua anotasi: {total_annotations}")
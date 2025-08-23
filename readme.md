# 🛵 YOLOv5 Deteksi Helm

Proyek ini menggunakan model **YOLOv5** dari Ultralytics untuk mendeteksi penggunaan helm pada gambar atau video. Proyek ini dioptimalkan untuk dijalankan di lingkungan **macOS** dengan isolasi dependensi menggunakan virtual environment.



---

## 🚀 Instalasi dan Setup

Ikuti langkah-langkah berikut untuk menyiapkan dan menjalankan proyek di mesin Mac Anda:

1.  **Kloning Repositori:**
    ```bash
    git clone https://github.com/ade-m/deteksi_helm_yolo.git
    cd deteksi_helm_yolo
    ```

2.  **Jalankan Skrip Setup:**
    Skrip `setup.sh` akan secara otomatis membuat virtual environment dan menginstal semua dependensi yang diperlukan.
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

3.  **Aktifkan Virtual Environment:**
    ```bash
    source yolov5_venv/bin/activate
    ```

---

## 📂 Struktur Proyek
<pre>
yolov5_project/
├── dataset/             # Tempat menyimpan dataset (gambar dan label)
│ ├── images/
│ └── labels/
├── yolov5_venv/         # Virtual environment
├── train_yolo.py        # Skrip untuk melatih model
├── requirements.txt     # Daftar dependensi Python
└── setup.sh             # Skrip instalasi otomatis
</pre>
Pastikan struktur folder dataset Anda sesuai dengan format YOLOv5. Contohnya seperti di bawah ini:
<pre>
dataset/
├── images/
│ ├── train/       # Gambar untuk training
│ └── val/         # Gambar untuk validasi
└── labels/
├── train/       # File .txt untuk label training
└── val/         # File .txt untuk label validasi
</pre>
---

## 🧠 Training dan Inferensi

### **Melatih Model**

Setelah dataset disiapkan, jalankan skrip `train_yolo.py` untuk memulai pelatihan.
<pre>
train_yolo.py

Hasil pelatihan akan disimpan di folder yolov5_results/.

Melakukan Inferensi

Berikut adalah contoh kode untuk melakukan prediksi pada gambar setelah model dilatih.

from ultralytics import YOLO

# Muat model yang telah dilatih
model = YOLO("yolov5s.pt")  

# Prediksi pada gambar tunggal
results = model.predict("dataset/images/val/contoh.jpg")

# Tampilkan hasil prediksi
results.show()
</pre>

📋 Dependensi
<pre>
Semua dependensi yang diperlukan tercantum di requirements.txt.

ultralytics>=8.0.0

torch>=2.1.0

torchvision>=0.16.0

matplotlib>=3.7.0

numpy>=1.25.0

opencv-python>=4.8.0
</pre>
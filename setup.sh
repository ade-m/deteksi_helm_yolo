#!/bin/bash

# ==========================
# Setup YOLOv5 Environment - Mac
# ==========================

# 1. Buat virtual environment
python3 -m venv yolov5_venv
echo "Virtual environment 'yolov5_venv' dibuat."

# 2. Aktifkan venv
source yolov5_venv/bin/activate
echo "Virtual environment aktif."

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
echo "Semua dependencies terinstall."

# 5. Instruksi setelah setup
echo "Setup selesai. Untuk menjalankan training:"
echo "1. Aktifkan venv: source yolov5_venv/bin/activate"
echo "2. Jalankan script training: python train_yolo.py"

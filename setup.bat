@echo off
REM ==========================
REM Setup YOLOv5 Environment - Windows
REM ==========================

REM 1. Buat virtual environment
python -m venv yolov5_venv
echo Virtual environment 'yolov5_venv' dibuat.

REM 2. Aktifkan venv
call yolov5_venv\Scripts\activate.bat
echo Virtual environment aktif.

REM 3. Upgrade pip
pip install --upgrade pip

REM 4. Install dependencies
pip install -r requirements.txt
echo Semua dependencies terinstall.

REM 5. Instruksi setelah setup
echo Setup selesai. Untuk menjalankan training:
echo 1. Aktifkan venv: call yolov5_venv\Scripts\activate.bat
echo 2. Jalankan script training: python train_yolo.py
pause
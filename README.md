# Objectdetection-Model-1st
📦 Webcam Object Detection using MobileNetSSD (OpenCV)

This project uses MobileNetSSD + OpenCV to perform real-time object detection from your webcam.

🚀 Features

Detects 20 common objects

Runs on CPU

Uses OpenCV DNN (no deep learning libraries needed)

Works on Windows / Linux / macOS

Lightweight — perfect for low-end laptops

📂 Folder Structure
object_detection/
│── MobileNetSSD_deploy.prototxt
│── MobileNetSSD_deploy.caffemodel
│── object_detection.py
│── README.md

🔧 Installation
1️⃣ Install dependencies
pip install opencv-python numpy

2️⃣ Run the script
python object_detection.py

🎥 Usage

The script opens your webcam

Objects are detected in real time

Press Q to quit

📁 Model Files

The model files used:

MobileNetSSD_deploy.prototxt

MobileNetSSD_deploy.caffemodel

Both must be placed in the same folder as the Python script.

📝 Code Automatically Handles File Paths

 need to edit paths.

from pathlib import Path

# ----------------------------
# PATH SETUP (GitHub-Friendly)
# ----------------------------
# Resolve the directory where this script is located
base_dir = Path(__file__).resolve().parent

# Model files (must be in the same folder)
prototxt_path = base_dir / "MobileNetSSD_deploy.prototxt"
model_path = base_dir / "MobileNetSSD_deploy.caffemodel"

🧠 Classes Detected
aeroplane, bicycle, bird, boat, bottle,
bus, car, cat, chair, cow, diningtable,
dog, horse, motorbike, person, pottedplant,
sheep, sofa, train, tvmonitor

# Multi-Camera Person Tracking and Recognition System

This is a real-time **CCTV surveillance system** that detects, tracks, and recognizes people across multiple camera feeds using **YOLOv8**, **DeepSORT**, **FaceNet**, and **OpenCV**.

---

## Features
- **Person Detection**: Uses YOLOv8 to detect people in CCTV footage.
- **Face Recognition**: Identifies known individuals using FaceNet.
- **Person Tracking**: Uses DeepSORT for multi-camera tracking.
- **Clothing Recognition**: Tracks people based on clothing color when the face is unrecognized.
- **Web Dashboard**: Displays real-time feeds with detected faces and tracking IDs.
- **SQLite Database**: Stores face embeddings and tracking data.

---

## Installation

### **1. Clone the Repository**
```sh
https://github.com/breaktheloop-sns/CCTV_ID_tracking_using_ai.git
cd CCTV_ID_tracking_using_ai
```

### **2. Setup Virtual Environment (Recommended)**
#### **For Python Virtualenv**
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
```

#### **For Conda Users**
```sh
conda env create -f environment.yml
conda activate cctv-env
```

---

## Usage

### **1. Start the Surveillance System**
```sh
python main.py
```

## Folder Structure
```
📁 CCTV_ID_tracking_using_ai
├── main.py             # Entry point
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Contributing
Feel free to contribute by submitting pull requests or opening issues!

---

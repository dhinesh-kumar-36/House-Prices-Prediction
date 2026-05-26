# Hand Gesture Recognition System using Deep Learning
**CNN + OpenCV + TensorFlow + Python**

---

## Project Overview

Real-time hand gesture recognition system:
- 10 gesture classes (palm, fist, thumbs-up, OK, ...)
- ~20,000 training images (LeapGestRecog dataset)
- CNN model achieving ~99% test accuracy
- Live webcam detection with confidence score display

---

## Quick Start (VS Code PowerShell)

```powershell
# 1. Project folder-க்கு போங்கள்
cd "C:\path\to\hand_gesture_recognition"

# 2. Virtual environment (recommended)
python -m venv venv
venv\Scripts\Activate.ps1

# 3. Dependencies install
pip install -r requirements.txt

# 4. Dataset download (kaggle.json வேணும்)
python download_dataset.py

# 5. Model train பண்ணுங்கள்
python train.py

# 6. Live webcam detection
python predict.py
```

> Full detailed guide: **VSCODE_SETUP_GUIDE.md**

---

## Gesture Classes

| # | Folder Name | Display Name |
|---|-------------|--------------|
| 0 | 01_palm | Palm (Open Hand) |
| 1 | 02_l | L Shape |
| 2 | 03_fist | Fist |
| 3 | 04_fist_moved | Fist (Moving) |
| 4 | 05_thumb | Thumbs Up |
| 5 | 06_index | Index Finger |
| 6 | 07_ok | OK Sign |
| 7 | 08_palm_moved | Palm (Moving) |
| 8 | 09_c | C Shape |
| 9 | 10_down | Pointing Down |

---

## CNN Architecture

```
Input: 64x64x1 (grayscale)
  → Conv2D(32) x2 + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(64) x2 + BatchNorm + MaxPool + Dropout(0.25)
  → Conv2D(128) x2 + BatchNorm + MaxPool + Dropout(0.40)
  → Flatten → Dense(256) + BatchNorm + Dropout(0.50)
  → Dense(10, softmax)
```

---

## Dataset

**LeapGestRecog** by GTI-UPM
- Link: https://www.kaggle.com/gti-upm/leapgestrecog
- Images: ~20,000 PNG files
- Subjects: 10 people
- Gestures: 10 classes
- Resolution: various (resized to 64x64 during training)

---

## Project Files

```
hand_gesture_recognition/
├── train.py               ← CNN training pipeline
├── predict.py             ← Live webcam detection
├── download_dataset.py    ← Auto dataset downloader
├── requirements.txt       ← Python packages
├── README.md              ← This file
├── VSCODE_SETUP_GUIDE.md  ← Detailed VS Code setup
├── dataset/leapGestRecog/ ← Dataset here
├── model/                 ← Saved model (auto-created)
└── plots/                 ← Graphs (auto-created)
```

---

## Webcam Controls

| Key | Action |
|-----|--------|
| Q | Quit |
| S | Pause / Resume detection |
| + | Raise confidence threshold |
| - | Lower confidence threshold |

---

## Hardware Requirements

- Python 3.10 or 3.11
- RAM: 8GB minimum
- Webcam: Required for predict.py
- GPU: Optional (NVIDIA with CUDA = faster training)


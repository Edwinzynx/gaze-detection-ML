# Eye-Controlled Mouse Cursor using Computer Vision

## Description
This project implements a **hands-free mouse control system** using eye gaze and blink detection.
It uses a webcam to track the user’s iris position and maps eye movements to screen coordinates,
allowing cursor movement and mouse clicks without physical input devices.
This system is designed for people who have movement disabilities and the future scope is to integrate this into an app of some sort which creates accessibilty for all people.
---

## ✨ Features
- Real-time **eye-gaze based cursor movement**
- **Blink-based mouse click**
- **Triple-blink gesture** to toggle freeze mode
- Cursor **smoothing** for stable movement
- **Automatic baseline reset** to handle head drift
- Sensitivity control for different users

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV** – video capture & visualization
- **MediaPipe Face Mesh** – facial & iris landmark detection
- **PyAutoGUI** – mouse control
- **NumPy** – numerical operations

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/eye-control.git
cd eye-control
```

### 2. Install dependencies
```bash
pip install opencv-python mediapipe pyautogui numpy
```

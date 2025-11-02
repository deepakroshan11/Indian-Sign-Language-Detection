# 🤟 Indian Sign Language Detection with Emotion Recognition

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

**Real-time ISL recognition system with AI-powered emotion detection and professional web dashboard**

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#️-architecture) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## ✨ Features

### 🎯 Core Functionality
- **Real-time Hand Tracking** - MediaPipe-based hand landmark detection
- **ISL Letter Recognition** - TensorFlow CNN model for A-Z alphabet recognition
- **Letter Confidence Overlay** - Live display of detected letter and accuracy on video feed
- **Smart Word Formation** - Automatic word completion with hand stability checking
- **Sentence Building** - Context-aware sentence construction with spacing detection

### 😊 Emotion Detection
- **Dual-Engine Emotion AI** - Combines FER (Facial Expression Recognition) + Custom Landmark Analysis
- **5 Emotion Categories** - Happy, Sad, Angry, Surprise, Neutral
- **Adaptive Fusion** - Dynamic weighting based on lighting conditions (CLAHE enhancement)
- **Temporal Smoothing** - 15-frame rolling average for stability
- **Real-time Timeline** - Live emotion tracking graph with Chart.js

### 🎨 Professional Web Dashboard
- **Modern Glassmorphism UI** - Beautiful gradient design with blur effects
- **Real-time Updates** - Socket.IO for zero-lag WebSocket communication
- **Live Statistics Panel** - FPS, letters detected, words formed, sentences, emotion changes
- **Smart Word Suggestions** - Frequency-based word predictions (top 3)
- **Interactive Controls** - Speak (TTS), Reset, Backspace buttons
- **Emotion Visualization** - Live progress bars + timeline chart

### 🔊 Text-to-Speech Integration
- **Auto-TTS** - Automatically speaks words as they're formed
- **Non-blocking Audio** - Runs in separate thread for smooth performance
- **gTTS Integration** - Natural voice synthesis via Google Text-to-Speech
- **Pygame Audio Engine** - Reliable cross-platform playback
- **Audio Cleanup Utility** - Automatic removal of temporary MP3 files

### ⚡ Performance Optimizations
- **Multi-process Architecture** - Separate processes for CV processing and web UI
- **IPC Queue Communication** - Fast inter-process data transfer
- **Dynamic Frame Management** - Maintains target 30 FPS with adaptive skipping
- **CLAHE Enhancement** - Contrast-Limited Adaptive Histogram Equalization for low-light performance
- **Camera Buffer Management** - Reduces frame latency
- **Hand Skeleton Overlay** - Visual feedback with colored landmarks and connections

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **Webcam** (720p or higher recommended)
- **4GB+ RAM**
- **Windows / Linux / macOS**

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/deepakroshan11/Indian-Sign-Language-Detection.git
cd Indian-Sign-Language-Detection
```

2. **Create Virtual Environment** (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Model File**
Ensure model.h5 exists in the project root (pre-trained ISL model included).

### Running the Application

**Method 1: Unified Launcher** (Recommended)
```bash
python launcher.py
```
This automatically starts both the core processor and web dashboard in separate processes.

**Method 2: Manual Launch**
```bash
# Terminal 1: Start core processing engine
python isl_detection.py

# Terminal 2: Start web dashboard
python isl_ui_dashboard.py
```

### Access the Dashboard
Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🎮 Usage

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl + R** | Reset entire session (clear all words and sentences) |
| **Ctrl + Backspace** | Delete last letter or word |
| **1** / **2** / **3** | Accept word suggestions 1, 2, or 3 |
| **Ctrl + C** | Stop application (in terminal) |

### Dashboard Controls

- **🔊 Speak Button** - Text-to-speech for current sentence
- **⌫ Backspace** - Remove last letter/word
- **🔄 Reset** - Clear session and start fresh

### How It Works

1. **Position your hand** in front of the webcam
2. **Make ISL letter signs** - The system detects hand landmarks
3. **Hold steady** - System confirms letter when confidence is high
4. **Word formation** - Remove hand for 15 frames to complete a word
5. **Suggestions** - Click on suggested words or press 1/2/3
6. **Emotion tracking** - Your facial emotions are tracked in real-time

---

## 🏗️ Architecture
```
┌──────────────────────────────────────────────────────────┐
│               LAUNCHER PROCESS (launcher.py)              │
│           Manages multi-process lifecycle                │
└────────────┬────────────────────────────┬─────────────────┘
             │                            │
             ▼                            ▼
┌────────────────────────┐      ┌─────────────────────────┐
│   CORE PROCESSOR       │      │    WEB DASHBOARD        │
│  (isl_detection.py)    │◄────►│  (isl_ui_dashboard.py)  │
│                        │ IPC  │                         │
│ • MediaPipe Hands      │Queue │ • Flask Server          │
│ • TensorFlow Model     │      │ • Socket.IO WebSocket   │
│ • FER + Landmarks      │      │ • Real-time Broadcast   │
│ • Letter Detection     │      │ • Chart.js Graphs       │
│ • Emotion Fusion       │      │ • Command Processing    │
│ • TTS Engine           │      │                         │
│ • Camera Buffer Mgmt   │      │                         │
└────────────────────────┘      └─────────────────────────┘
             │                            │
             ▼                            ▼
    📹 Webcam Input              🌐 Browser Client (Port 5000)
```

### Data Flow

1. **Camera Frame** → MediaPipe → Hand Landmarks Extraction
2. **Landmarks** → TensorFlow CNN Model → Letter Prediction (A-Z)
3. **Video Frame** → FER Model → Facial Emotion Scores
4. **Video Frame** → MediaPipe Face Mesh → Custom Landmark Features
5. **Emotion Fusion** → Weighted Average (FER + Landmarks) → Final Emotion
6. **Data Packet** → IPC Queue (JPEG encoded + metadata)
7. **Flask Server** → Socket.IO → Browser WebSocket
8. **Browser** → Real-time Dashboard Updates (30 FPS)

---

## 📊 Configuration

Edit isl_detection.py to customize behavior:
```python
# Camera Settings
CAM_INDEX = 0                    # Camera device index (0, 1, 2...)

# Detection Parameters
BUFFER_SIZE = 12                 # Letter stability buffer size
CONF_THRESHOLD = 0.8             # Minimum confidence to accept letter
STABILITY_THRESHOLD = 0.7        # Hand stability threshold
SPACE_THRESHOLD = 15             # Frames before word separation
SENTENCE_DELAY = 2.0             # Seconds delay for sentence formation

# Display Settings
SHOW_LETTER_OVERLAY = True       # Show detected letter on video feed
OVERLAY_POSITION = (20, 50)      # Letter overlay position (x, y)
OVERLAY_FONT_SCALE = 1.5         # Font size for overlay
OVERLAY_COLOR_LETTER = (0, 255, 255)  # Yellow (BGR)
OVERLAY_COLOR_CONF = (255, 255, 0)    # Cyan (BGR)

# Emotion Detection
SMOOTHING_FRAMES = 15            # Temporal smoothing window
BASE_FER_WEIGHT = 0.65           # FER model weight in fusion
BASE_LANDMARK_WEIGHT = 0.35      # Landmark model weight in fusion
MIN_CONF_TO_SHOW = 0.25          # Minimum confidence to display emotion

# TTS Settings
ENABLE_AUTO_TTS = True           # Auto-speak words when formed
TTS_LANGUAGE = "en"              # Voice language (en, es, fr, etc.)
```

---

## 📁 Project Structure
```
Indian-Sign-Language-Detection/
│
├── 📄 launcher.py                    # Multi-process unified launcher
├── 📄 isl_detection.py               # Core CV + ML processing engine
├── 📄 isl_ui_dashboard.py            # Flask web server + Socket.IO
├── 📄 cleanup_audio_files.py         # TTS audio cleanup utility
├── 📄 dataset_keypoint_generation.py # Data collection script
│
├── 🤖 model.h5                       # Trained TensorFlow ISL model (11.5 MB)
├── 📊 keypoint.csv                   # Training dataset landmarks
│
├── 📁 templates/
│   ├── dashboard.html                # Professional web UI
│   └── dashboard.html.backup         # Backup copy
│
├── 📁 dataset/                       # Raw training images
├── 📁 images/                        # Collected hand gesture images
├── 📁 __pycache__/                   # Python cache (ignored)
│
├── 📋 requirements.txt               # Python dependencies
├── 📋 .gitignore                     # Git ignore rules
└── 📋 README.md                      # This file
```

---

## 🔧 Troubleshooting

### Issue: Camera Not Detected
```python
# In isl_detection.py, change camera index:
CAM_INDEX = 1  # Try different values: 0, 1, 2...
```

### Issue: Low Frame Rate
- Close other resource-intensive applications
- Reduce SMOOTHING_FRAMES value
- Disable CLAHE: Comment out pply_clahe() calls
- Lower webcam resolution in system settings

### Issue: TTS Files Not Deleted
```bash
python cleanup_audio_files.py
```

### Issue: Port 5000 Already in Use
```python
# In isl_ui_dashboard.py, change:
socketio.run(app, host='0.0.0.0', port=5001)
```

### Issue: Model Not Loading
- Verify model.h5 exists in project root
- Check TensorFlow version: pip show tensorflow
- Reinstall TensorFlow: pip install --upgrade tensorflow

### Issue: MediaPipe Errors
```bash
pip uninstall mediapipe
pip install mediapipe --no-cache-dir
```

---

## 🎓 Training Your Own Model

1. **Collect Training Data**
```bash
python dataset_keypoint_generation.py
```
Follow the prompts to capture hand images for each ISL letter (A-Z).

2. **Generate Keypoint CSV**
The script automatically creates keypoint.csv with landmark coordinates.

3. **Train Model**
Use the provided Jupyter notebook ISL_classifier.ipynb or create your own training script.

4. **Replace Model**
```bash
# Backup old model
mv model.h5 model_backup.h5

# Use your new model
cp your_trained_model.h5 model.h5
```

---

## 📊 Performance Metrics

- **Frame Rate**: 25-30 FPS (real-time)
- **Letter Detection Accuracy**: ~95% (with trained model)
- **Emotion Detection Latency**: <50ms per frame
- **Hand Landmark Detection**: 21 points tracked
- **Face Landmark Detection**: 468 points tracked
- **Dashboard Update Rate**: Real-time via WebSocket

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: git checkout -b feature/AmazingFeature
3. **Commit your changes**: git commit -m 'Add AmazingFeature'
4. **Push to branch**: git push origin feature/AmazingFeature
5. **Open a Pull Request**

### Areas for Contribution
- [ ] Add more ISL gestures (words, phrases)
- [ ] Improve emotion detection accuracy
- [ ] Add multi-language support for TTS
- [ ] Create mobile app version
- [ ] Add training data augmentation
- [ ] Optimize model for edge devices

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MediaPipe** - Real-time hand and face landmark detection
- **TensorFlow** - Deep learning framework
- **FER (Facial Expression Recognition)** - Pre-trained emotion detection
- **Flask + Socket.IO** - Real-time web communication
- **Chart.js** - Beautiful data visualization
- **gTTS** - Google Text-to-Speech API
- **Pygame** - Cross-platform audio playback

---

## 📧 Contact

**Deepak Roshan**
- 🐙 GitHub: [@deepakroshan11](https://github.com/deepakroshan11)
- 📁 Repository: [Indian-Sign-Language-Detection](https://github.com/deepakroshan11/Indian-Sign-Language-Detection)

---

## 🌟 Show Your Support

If this project helped you, please consider:
- ⭐ **Star this repository**
- 🍴 **Fork and contribute**
- 📢 **Share with others**

---

<div align="center">

**Made with ❤️ for the deaf and hard-of-hearing community**

![ISL](https://img.shields.io/badge/ISL-Indian_Sign_Language-blue)
![Accessibility](https://img.shields.io/badge/Accessibility-First-green)
![Open Source](https://img.shields.io/badge/Open_Source-Yes-orange)

</div>

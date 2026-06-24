# 🎙️ Voice Mood Detector

[![GPL v3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![ModelScope](https://img.shields.io/badge/AI-ModelScope-purple.svg)](https://www.modelscope.cn/)

[中文](README.md) | English

---

## 📖 About

Voice Mood Detector is a deep learning-based desktop application that records your voice through a microphone and uses AI models to intelligently recognize emotional states in speech, evaluates emotional stability, and provides personalized adjustment suggestions.

The system adopts a **portable mode** design — all user data (recordings, models, history, logs, etc.) is saved in the program folder. You can copy the entire program folder to migrate all data without polluting system directories.

> ⚠️ **Disclaimer**: This software is for personal non-commercial reference only. The detection results are only an auxiliary reference for emotional state and do not constitute any medical diagnosis or advice. If you have persistent emotional distress, please consult a professional psychologist promptly.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 Real-time Recording | Support real-time microphone recording with duration and volume display |
| 🧠 Smart Recognition | Based on emotion2vec_plus_large model, recognizes 7 emotions |
| 📊 Emotional Stability | 0-10 scale score to evaluate emotional fluctuation |
| 🎯 Mixed Emotion Detection | Automatically detects mixed emotions for more realistic results |
| 💡 Adjustment Suggestions | Personalized suggestions based on emotional state |
| 📋 History Records | Auto-save history, up to 200 records |
| 📈 Trend Charts | Visualize emotional stability change trends |
| 💾 Portable Mode | All data saved in program folder, plug-and-play |
| 🧹 One-click Cleanup | One-click cleanup of temp files, cache, logs, etc. |

---

## 🚀 Quick Start

### Requirements

| Item | Requirement |
|------|-------------|
| Operating System | Windows 10 / Windows 11 |
| Python Version | 3.8 ~ 3.11 |
| Memory | 4GB+ recommended |
| Disk Space | At least 2GB (model ~1GB + runtime space) |
| Audio Device | Available microphone |

### Installation

1. **Clone or download the project**

```bash
git clone <repo-url>
cd Voice_Mood_Detect
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> 💡 **Tip**: If PyAudio installation fails, try using conda:
> ```bash
> conda install pyaudio -y
> ```

### Run the Program

```bash
python main.py
```

---

## 📖 User Guide

### First Run (Model Download)

1. When you start the program for the first time, a **welcome wizard** will appear introducing the system
2. Click the "🚀 Start Using (Download Model)" button
3. The program will automatically download the **emotion2vec_plus_large** model from ModelScope
4. The model is approximately **1GB**, please be patient
5. After download, the model will be saved in the `portable_data/models/` directory
6. Future launches will load the local model directly without re-downloading

### Recording and Analysis

1. Wait for the model to finish loading (status bar shows "● Ready")
2. Click the green **"🎤 Click to Start Recording"** button
3. Speak naturally about how you feel (recommended recording duration: 3-30 seconds)
4. Click the red **"⏹️ Click to Stop Recording"** button
5. Wait a few seconds to see the analysis results

### Results Explanation

- **Emotional Stability**: 0-10 points, higher score indicates greater emotional fluctuation
  - 0-2: Very Stable 🟢
  - 2-4: Good 🟢
  - 4-6: Fair 🟡
  - 6-8: Unstable 🔴
  - 8-10: Intense Emotion 🔴

- **Main Emotion**: Calm, Happy, Surprised, Sad, Angry, Fearful, Disgusted
- **Mixed Emotions**: Other emotions detected simultaneously (probability > 8%)
- **Suggestions**: Personalized suggestions based on emotional state

### History Records

1. Click the **"📋 History Report"** tab at the top
2. View all historical detection records (sorted by time, newest first)
3. The **emotional stability trend chart** is shown below
4. Record numbers are ordered chronologically (1st, 2nd, ...)

### Data Management

1. Click the menu **"File → Data Management"** (shortcut: Ctrl+D)
2. View data storage location and storage statistics
3. Supported operations:
   - **Clear all history and recordings**: Delete all detection records and audio files
   - **One-click system junk cleanup**: Clear temp files, cache files, log files
   - **Open data folder**: Open the data directory in file manager

### Uninstallation

Thanks to the portable mode, uninstallation is very simple:

1. Close the program
2. Simply delete the entire program folder
3. No residual files will be left on your system

> 💡 If you only want to clear personal data but keep the program, use "Clear History" and "One-click System Junk Cleanup" in Data Management.

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Version/Notes |
|------------|---------|---------------|
| **Python** | Development language | 3.8+ |
| **PyQt5** | GUI framework | GPL v3 license |
| **PyTorch** | Deep learning framework | Model inference |
| **ModelScope** | Model library management | Alibaba Damo Academy |
| **FunASR** | Speech recognition framework | Supports emotion2vec |
| **emotion2vec_plus_large** | Emotion recognition model | Apache 2.0 license |
| **PyAudio** | Audio recording | MIT license |
| **librosa** | Audio processing | ISC license |
| **matplotlib** | Data visualization | Trend chart plotting |
| **NumPy** | Numerical computing | Data processing |

### Project Architecture

```
Voice_Mood_Detect/
├── main.py                 # Program entry point
├── gui.py                  # GUI (main window, dialogs, widgets)
├── recorder.py             # Audio recording module
├── emotion_recognizer.py   # Emotion recognition core module
├── history_manager.py      # History record management
├── app_paths.py            # Path management (portable mode)
├── relaxation_tips.py      # Relaxation tips library
├── requirements.txt        # Python dependencies
├── LICENSE                 # GPL v3 license
├── README.md               # Chinese documentation
├── README.en.md            # English documentation
└── .gitignore              # Git ignore config
```

### Runtime Directories

The following directories are automatically created after program runs (portable mode):

```
portable_data/
├── recordings/    # Recording files (deletable)
├── models/        # AI model files (not deletable, ~1GB)
├── logs/          # Runtime logs (clearable)
├── temp/          # Temporary files (clearable)
├── cache/         # Cache files (clearable)
└── history.json   # History records (deletable)
```

---

## 📄 License

### Project License

This project is licensed under the **GNU General Public License v3.0 (GPL v3)**.

Reasons for choosing GPL v3:
- This project uses PyQt5 (GPL v3 license), GPL v3 is adopted for license consistency
- Ensures the project and its derivatives remain open source
- Protects the common interests of the open source community

See the [LICENSE](LICENSE) file for the full license text.

### Third-Party Dependencies and Licenses

| Project | License | Notes |
|---------|---------|-------|
| **emotion2vec_plus_large** | Apache License 2.0 | Emotion recognition model, provided by ModelScope |
| **FunASR** | Apache License 2.0 | Speech recognition toolkit |
| **ModelScope** | Apache License 2.0 | Model service platform |
| **PyTorch** | BSD-style | Deep learning framework |
| **PyQt5** | GPL v3 / Commercial | GUI framework (open source version uses GPL v3) |
| **PyAudio** | MIT | Audio recording library |
| **librosa** | ISC | Audio processing library |
| **matplotlib** | PSF-based | Data visualization library |
| **NumPy** | BSD | Numerical computing library |

> ✅ **Compatibility Note**: All third-party library licenses above are compatible with GPL v3.

---

## 🤝 Contributing

Contributions to this project are welcome! You can participate in the following ways:

1. **Submit Bug Reports**: Describe issues you encounter in Issues
2. **Propose Features**: Share your ideas and suggestions
3. **Submit Code**: Fork the project and submit a Pull Request
4. **Improve Documentation**: Help improve documentation and translations

### Development Setup

```bash
# Clone the project
git clone <repo-url>
cd Voice_Mood_Detect

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt

# Run the program
python main.py
```

### Code Standards

- All Python code uses UTF-8 encoding
- Core modules should include detailed Chinese comments and docstrings
- Follow PEP 8 code style
- Please ensure the code runs normally before submitting

---

## ⚠️ Disclaimer

1. This software is for personal learning and non-commercial reference only
2. Emotion detection results are for reference only and do not constitute any medical diagnosis or advice
3. If you have persistent low mood, anxiety, or other psychological distress, please consult a professional psychologist promptly
4. Any consequences arising from the use of this software shall be borne by the user
5. The AI model is provided by a third party. This project makes no guarantee regarding the accuracy or reliability of the model

---

## 📞 Contact

- Author: **Jiawei Li**
- Repository: GitHub

---

*Thank you for using Voice Mood Detector! If you find it useful, please give a Star ⭐ to show your support~*

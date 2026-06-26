# 🎙️ Voice Mood Detector

[![GPL v3 License](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![ModelScope](https://img.shields.io/badge/AI-ModelScope-purple.svg)](https://www.modelscope.cn/)

[中文](README.md) | English

---

## 📖 About

Voice Mood Detector is a deep learning-based desktop application that records your voice through a microphone, uses emotion2vec+ series large models to intelligently recognize emotional states in speech, evaluates emotional stability through a multi-factor algorithm, automatically detects compound emotion patterns, and provides personalized layered adjustment suggestions.

The system adopts a **portable mode** design — all user data (recordings, models, history, logs, etc.) is saved in the program folder. You can copy the entire program folder to migrate all data without polluting system directories.

> ⚠️ **Disclaimer**: This software is for personal non-commercial reference only. The detection results are only an auxiliary reference for emotional state and do not constitute any medical diagnosis or advice. If you have persistent emotional distress, please consult a professional psychologist promptly.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎤 Real-time Recording | Real-time microphone recording with duration and volume indicator |
| 🧠 Multi-model Recognition | Supports emotion2vec_plus_seed/base/large models, recognizes 8 basic emotions |
| 📊 Multi-factor Stability Score | Three-factor 0-10 scoring combining negative weights, Shannon entropy, and extremity |
| 🎯 Compound Emotion Detection | Auto-detects anxiety, frustration, jealousy, tension, boredom, resentment patterns |
| 🔀 Mixed Emotion Analysis | Detects and displays co-occurring emotions (probability > 8%) |
| 💡 Layered Suggestions | Immediate + long-term suggestion system with compound-emotion-specific advice |
| 📋 History Records | Auto-save history with batch management, up to 200 records |
| 📈 Trend Charts | Visualize emotional stability change trends |
| 💾 Portable Mode | All data saved in program folder, plug-and-play |
| 🧹 One-click Cleanup | One-click cleanup of temp files, cache, logs, etc. |

---

## 🧬 Core Algorithms & Technical Principles

### Emotion Recognition Model (emotion2vec+)

This system uses the **emotion2vec+** series pre-trained models developed by Alibaba DAMO Academy, trained on large-scale speech emotion data, supporting end-to-end speech emotion recognition.

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| emotion2vec_plus_seed | ~200MB | ⚡⚡⚡ Fastest | ★★★ | Low-spec devices, quick trial |
| emotion2vec_plus_base | ~500MB | ⚡⚡ Balanced | ★★★★ | Daily use, speed-accuracy balance |
| emotion2vec_plus_large | ~1GB | ⚡ Slower | ★★★★★ | Accuracy priority (recommended) |

**Model Output**: Probability distribution over 8 basic emotion categories

```
angry → Angry    disgusted → Disgusted    fearful → Fearful    happy → Happy
neutral → Calm    other → Other    sad → Sad    surprised → Surprised
```

### Emotional Stability Scoring Algorithm

The system uses a **three-factor weighted scoring model** to calculate emotional stability (0-10, higher = more unstable):

```
Final Score = Negative Weight Score × 0.40 + Entropy Score × 0.30 + Extremity Score × 0.30
```

**Factor 1: Negative Emotion Weighted Score (40%)**

Contribution weights of each emotion to instability:

| Emotion | Weight | Psychological Basis |
|---------|--------|-------------------|
| Fearful | 0.95 | High arousal, strongest stress response |
| Angry | 0.90 | High-activation negative emotion |
| Sad | 0.80 | Low activation but sustained impact |
| Disgusted | 0.70 | Moderate negative impact |
| Surprised | 0.30 | Brief activation, limited impact |
| Other | 0.15 | Unrecognized, slight contribution |
| Happy/Calm | 0.00 | Positive/neutral emotions have no negative contribution |

**Factor 2: Emotion Dispersion / Shannon Entropy (30%)**

Uses Shannon information entropy to measure the disorder of emotion probability distribution:

```
H = -Σ p(i) × log₂(p(i))    normalized to 0-10
```

High entropy means multiple emotions co-exist, emotional state is ambiguous, suggesting psychological conflict.

**Factor 3: Emotion Extremity (30%)**

Detects extremely high-probability negative emotions:
- Single negative emotion probability > 60%: Extreme instability
- Single negative emotion probability > 40%: Moderate instability
- Positive emotions dominating automatically reduces instability

**Stability Level Classification:**

| Score Range | Level | Meaning |
|-------------|-------|---------|
| 0.0 - 2.0 | Very Stable 🟢 | Peaceful, emotionally healthy |
| 2.0 - 3.5 | Good 🟢 | Minor fluctuation, normal range |
| 3.5 - 5.0 | Fair 🟡 | Some emotional fluctuation |
| 5.0 - 6.5 | Mild Fluctuation 🟠 | Attention needed |
| 6.5 - 8.0 | Unstable 🔴 | Noticeable emotional fluctuation |
| 8.0 - 10.0 | Intense 🔴 | Strong emotions, immediate relaxation advised |

### Compound Emotion Detection Mechanism

Compound emotions are formed by specific combinations of basic emotions. When all component emotions exceed the threshold (10%), compound emotion detection is triggered:

| Compound Emotion | Component Pattern | Psychological Description |
|-----------------|-------------------|--------------------------|
| **Anxiety** | Fear + Sadness | Interweaving of fear and sadness, uncertainty about the future |
| **Frustration** | Anger + Sadness | Co-existence of anger and sadness, from unmet expectations |
| **Jealousy** | Anger + Sadness + Fear | Triple interweaving, involving insecurity about one's position |
| **Tension** | Fear + Surprise | Stress response when facing unknown challenges |
| **Boredom** | Disgust + Sadness | From prolonged repetition or lack of meaning |
| **Resentment** | Anger + Disgust | Strong rejection and confrontational emotions |

When multiple compound emotions are simultaneously qualified, the system selects the one with the highest total component probability.

### Real-time Voice Emotion Analysis Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│              Real-time Voice Emotion Analysis Pipeline            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐   ┌─────────────┐   ┌────────────────┐          │
│  │ Microphone│──→│  Recorder   │──→│  WAV File Save │          │
│  │   Input   │   │  (16kHz)    │   │  (16-bit PCM)  │          │
│  └───────────┘   └─────────────┘   └───────┬────────┘          │
│                                              │                   │
│                                              ▼                   │
│                                   ┌──────────────────┐          │
│                                   │ emotion2vec+ Infer│          │
│                                   │ (CPU/GPU, FunASR) │          │
│                                   └────────┬─────────┘          │
│                                            │                     │
│                                            ▼                     │
│                                ┌───────────────────┐            │
│                                │ Label Normalize &  │            │
│                                │ Probability Dist.  │            │
│                                └─────────┬─────────┘            │
│                                          │                       │
│                ┌─────────────────────────┼──────────────────┐   │
│                ▼                         ▼                   ▼   │
│  ┌──────────────────┐  ┌────────────────────┐  ┌────────────┐  │
│  │ 3-Factor Stability│  │ Compound Emotion   │  │ Mixed Emo  │  │
│  │ Score Calculation │  │ Pattern Matching   │  │ Detection  │  │
│  └────────┬─────────┘  └─────────┬──────────┘  └──────┬─────┘  │
│           └───────────────────────┼──────────────────────┘      │
│                                   ▼                              │
│                        ┌────────────────────┐                   │
│                        │ Combined Result +  │                   │
│                        │ Layered Suggestions│                   │
│                        └────────┬───────────┘                   │
│                                 ▼                                │
│                       ┌──────────────────┐                      │
│                       │ GUI Display +    │                      │
│                       │ History Storage  │                      │
│                       └──────────────────┘                      │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

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

1. When you start the program for the first time, a **welcome wizard** will appear
2. Click the "🚀 Start Using (Download Model)" button
3. The program will automatically download the **emotion2vec_plus_large** model (default) from ModelScope
4. The model is approximately **1GB**, please be patient
5. After download, the model will be saved in the `portable_data/models/` directory
6. Future launches will load the local model directly without re-downloading

### Recording and Analysis

1. Wait for the model to finish loading (status bar shows "● Ready")
2. Click the green **"🎤 Click to Start Recording"** button
3. Speak naturally about how you feel (recommended: 3-30 seconds)
4. Click the red **"⏹️ Click to Stop Recording"** button
5. Wait a few seconds to see the analysis results

### Results Explanation

- **Emotional Stability**: 0-10 points, higher score indicates greater emotional fluctuation
  - 0-2: Very Stable 🟢
  - 2-3.5: Good 🟢
  - 3.5-5: Fair 🟡
  - 5-6.5: Mild Fluctuation 🟠
  - 6.5-8: Unstable 🔴
  - 8-10: Intense 🔴

- **Main Emotion**: Angry, Disgusted, Fearful, Happy, Calm, Sad, Surprised, Other
- **Mixed Emotions**: Other emotions detected simultaneously (probability > 8%)
- **Compound Emotions**: Higher-order emotional states composed of basic emotions
- **Suggestions**: Layered suggestions based on emotional state (immediate + long-term)

### Model Switching

The system supports three model sizes:
- **Seed (Smallest)**: ~200MB, for low-spec devices
- **Base (Standard)**: ~500MB, speed-accuracy balance
- **Large (Largest)**: ~1GB, highest accuracy (recommended)

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

---

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Version/Notes |
|------------|---------|---------------|
| **Python** | Development language | 3.8+ |
| **PyQt5** | GUI framework | GPL v3 license |
| **PyTorch** | Deep learning framework | CPU inference (enforced) |
| **ModelScope** | Model library management | Alibaba DAMO Academy |
| **FunASR** | Speech recognition framework | Supports emotion2vec+ series |
| **emotion2vec+** | Emotion recognition model | seed/base/large variants, Apache 2.0 |
| **PyAudio** | Audio recording | 16kHz mono, MIT license |
| **librosa** | Audio processing | ISC license |
| **matplotlib** | Data visualization | Trend chart plotting |
| **NumPy / SciPy** | Numerical computing | Data processing & algorithms |

### Project Architecture

```
Voice_Mood_Detect/
├── main.py                 # Entry point: env config + PyQt5 app launch
├── gui.py                  # GUI: main window, widgets, interaction logic
├── emotion_recognizer.py   # Emotion recognition core: model + inference + algorithms
├── recorder.py             # Audio recording: threaded recording + WAV save
├── history_manager.py      # History management: atomic write + data sanitization
├── app_paths.py            # Path management: portable mode + security checks
├── relaxation_tips.py      # Relaxation tips: layered suggestion library
├── requirements.txt        # Python dependencies
├── LICENSE                 # GPL v3 license
├── README.md               # Chinese documentation
├── README.en.md            # English documentation
└── .gitignore              # Git ignore config
```

### Module Dependencies

```
main.py
  ├── app_paths.py          (environment setup)
  └── gui.py                (main interface)
        ├── emotion_recognizer.py  (emotion recognition)
        │     └── app_paths.py     (model paths)
        ├── recorder.py            (recording)
        ├── history_manager.py     (history records)
        │     └── app_paths.py     (storage paths)
        └── relaxation_tips.py     (adjustment tips)
              └── emotion_recognizer.py  (compound emotion defs)
```

### Runtime Directories

The following directories are automatically created after program runs (portable mode):

```
portable_data/
├── recordings/        # Recording files (WAV format, deletable)
├── models/            # AI model files (not deletable, ~1GB)
│   └── models/iic/   # ModelScope model cache structure
├── logs/              # Runtime logs (clearable)
├── temp/              # Temporary files (clearable)
├── cache/             # Cache files (clearable)
├── history.json       # History records (JSON format)
└── model_config.json  # Model config (currently selected model)
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
| **emotion2vec+** | Apache License 2.0 | Emotion recognition model, by Alibaba DAMO Academy |
| **FunASR** | Apache License 2.0 | Speech recognition toolkit |
| **ModelScope** | Apache License 2.0 | Model service platform |
| **PyTorch** | BSD-style | Deep learning framework |
| **PyQt5** | GPL v3 / Commercial | GUI framework (open source uses GPL v3) |
| **PyAudio** | MIT | Audio recording library |
| **librosa** | ISC | Audio processing library |
| **matplotlib** | PSF-based | Data visualization library |
| **NumPy** | BSD | Numerical computing library |
| **SciPy** | BSD | Scientific computing library |

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
# Activate conda environment
conda activate audio
# Or use full path:
# C:\Users\xiaol\anaconda3\envs\audio\python.exe

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Test

```bash
# Run all tests
python -m pytest tests/ -v

# Run a single test
python -m pytest tests/test_emotion.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Architecture

This is a PyQt5 desktop application for speech emotion recognition using Alibaba's emotion2vec+ models via ModelScope/FunASR.

### Core Pipeline

```
Microphone → AudioRecorder → WAV file → EmotionRecognizer → Result dict → GUI display
                                                              ↓
                                                       HistoryManager (JSON)
```

### Key Modules

| Module | Role |
|--------|------|
| `main.py` | Entry point — env setup, QApplication init, launches MainWindow |
| `gui.py` | Main GUI (~3100 lines) — MainWindow, dialogs, ScoreCard, MplCanvas, threads |
| `gui_widgets/` | Reusable widget package — ResultCardWidget, ResearchRadarChart, BaselinePanel, StatsPanel |
| `emotion_recognizer.py` | AI inference wrapper — loads emotion2vec+ via ModelScope/FunASR, runs predict() |
| `recorder.py` | AudioRecorder — PyAudio-based microphone capture to WAV |
| `history_manager.py` | JSON-based history CRUD, auto-cleanup, statistics aggregation |
| `audio_features.py` | Acoustic feature extraction (F0, jitter, shimmer, HNR, MFCC) via Parselmouth |
| `audio_quality.py` | Real-time audio quality analysis (volume, clipping, noise, speech ratio) |
| `baseline.py` | Personal baseline modeling — multi-sample calibration, deviation detection |
| `statistics.py` | Statistical analysis (Cohen's d, trend detection, anomaly, MCID) |
| `export_manager.py` | CSV/JSON research data export |
| `vad_dimensions.py` | Valence-Arousal-Dominance dimension estimation from emotion probabilities |
| `app_paths.py` | Portable mode path management — all data under `portable_data/` |

### GUI Architecture

```
MainWindow
├── Header (title + status + model switch)
├── QTabWidget
│   ├── "实时检测" tab
│   │   ├── Left panel: Recording controls + Result cards + Probability bars
│   │   └── Right panel: Quick guide (steps + features + tips)
│   └── "历史报告" tab
│       ├── History list + Export toolbar
│       └── Trend chart (matplotlib)
└── Bottom bar (data management + credits)
```

### Design Style

Soviet Constructivist (苏联构成主义): thick black borders (#2B2B2B), brick-red accents (#C44B4F), warm off-white background (#F2EDE4/#E8E3DA), 45° diagonal textures, triangular wedge decorations, no rounded corners, bold typography.

### Data Flow

- All user data stored in `portable_data/` (recordings, history.json, model cache, logs)
- `model_config.json` tracks which model is selected
- History capped at 200 records, auto-cleans oldest

## Git Workflow

- **Branch**: feature branches from `main`
- **Strategy**: small frequent commits, auto-push to GitHub
- **No automatic PRs** — manual review before merging
- **Remote**: `origin` → `https://github.com/lijiawei255/voice-mood-detector.git`

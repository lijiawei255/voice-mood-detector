# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

```bash
# Create and activate an Anaconda environment (recommended, named `audio`)
conda create -n audio python=3.10 -y
conda activate audio

# Install libraries that need local compilation (PyAudio) via conda first
conda install -c conda-forge pyaudio -y

# Install dependencies
pip install -r requirements.txt
# Optional: test dependencies
pip install -r requirements-dev.txt
```

Supported Python versions: 3.8 ~ 3.11 (3.10 tested). PyQt5 is required for GUI tests; headless/CI tests skip GUI imports if PyQt5 is unavailable.

## Run

```bash
python main.py
```

## Test

```bash
# Run all tests
python -m pytest tests/ -v

# Latest status:
#   - With PyQt5 installed (anaconda `audio` env): 153 passed, 7 skipped
#   - Headless/CI without PyQt5: ~150 passed, 10 skipped (GUI tests auto-skip)

# Run a single test
python -m pytest tests/test_emotion.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing
```

## Architecture

PyQt5 desktop application for speech emotion recognition using Alibaba's emotion2vec+ models via ModelScope/FunASR. CPU inference is enforced (`device="cpu"`); GPU errors are caught and cache is cleared.

### Core Pipeline

```
Microphone → AudioRecorder → WAV file → AudioQualityAnalyzer
                                              ↓
                                EmotionRecognizer.predict()
                                              ↓
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
   VADDimensions + Stability     Compound Emotions    Acoustic Features + Psychological Indicators
              │                               │                               │
              └───────────────────────────────┼───────────────────────────────┘
                                              ▼
                              Reliability + Personal Baseline Deviation
                                              ↓
                                        Result dict
                                              ↓
                              GUI display + HistoryManager (JSON) + Export (CSV/JSON)
```

### Key Modules

| Module | Role |
|--------|------|
| `main.py` | Entry point — env setup, QApplication init, launches MainWindow |
| `gui.py` | Main GUI (~3000 lines) — MainWindow, tabs, dialogs, threads, result layout |
| `gui_widgets/` | Reusable widget package — ResultCardWidget, ResearchRadarChart, BaselinePanel, StatsPanel, ScoreCard, assessment cards |
| `emotion_recognizer.py` | AI inference wrapper — singleton, loads emotion2vec+ via FunASR, runs `predict()` with full P0/P1/P2 enrichment |
| `research_session.py` | Research-mode orchestrator — quality gates, multi-sample recording, dual-model validation |
| `recorder.py` | AudioRecorder — PyAudio-based microphone capture to WAV |
| `history_manager.py` | JSON-based history CRUD, auto-cleanup, statistics aggregation |
| `audio_features.py` | Acoustic feature extraction (F0, jitter, shimmer, HNR, MFCC) via librosa + praat-parselmouth fallback |
| `audio_quality.py` | Audio quality analysis (duration, clipping, noise, speech ratio) and preprocessing |
| `baseline.py` | Personal baseline modeling — multi-sample calibration, deviation detection, personalized stability score |
| `reliability.py` | Assessment reliability — ICC, multi-sample consistency, model-agreement/scale-sensitivity, composite level |
| `statistics.py` | Statistical analysis (descriptive stats, Cohen's d, trend detection, anomaly, MCID) |
| `export_manager.py` | CSV/JSON research data export |
| `vad_dimensions.py` | Valence-Arousal-Dominance dimension estimation from emotion probabilities |
| `relaxation_tips.py` | Layered relaxation/suggestion texts keyed by emotion and stability level |
| `app_paths.py` | Portable mode path management — all data under `portable_data/` |
| `version.py` | App version, algorithm version, model metadata |

### GUI Architecture

```
MainWindow (1500×1000 default)
├── Header (title + status + model switch)
├── QTabWidget
│   ├── "实时检测" / Real-time Detection tab
│   │   ├── Left panel (scrollable)
│   │   │   ├── Recording controls (mode switch, duration, quality, progress)
│   │   │   └── Result area — score/level/emotion/compound cards,
│   │   │       ResultCardWidget (VAD bars + stability factors),
│   │   │       radar chart, reliability badge,
│   │   │       acoustic feature card, psychological indicator card,
│   │   │       baseline deviation card, probability bars
│   │   └── Right panel — Quick guide + Personal baseline management
│   └── "历史报告" / History tab
│       ├── History list + Export toolbar (CSV/JSON)
│       ├── Emotion trend chart (matplotlib)
│       └── Statistics panel (descriptive stats + VAD means)
└── Bottom bar (data management + credits)
```

### Design Style

极简主义（瑞士/包豪斯现代极简）：纯白背景 (#FFFFFF)，浅灰次背景 (#F5F5F7)，文字主色 #1D1D1F / 次色 #86868B，蓝色强调 #1A73E8，细线边框 (#D2D2D7)，适度圆角 (4-8px)，状态色（成功 #34A853 / 警告 #F9AB00 / 错误 #EA4335）。无装饰图案、无三角形/楔形、无斜线纹理，功能优先。

历史背景：本应用早期采用「苏联构成主义」风格（粗炭黑边框、砖红强调、三角装饰），于 2026-06-27 重构为极简主义风格。

### Data Flow

- All user data stored in `portable_data/` (recordings, history.json, baseline.json, model cache, logs, temp)
- `model_config.json` tracks which model is selected
- History capped at 200 records, auto-cleans oldest
- Baseline requires 3–10 calm-state samples

### Development Notes

- Keep CPU-only inference path robust; do not assume GPU availability.
- Maintain the test suite when adding backend fields or GUI cards.
- Avoid adding heavy runtime dependencies; prefer librosa + parselmouth over proprietary toolkits.
- When modifying GUI layouts, capture screenshots manually via the running app (requires a display) and inspect `portable_data/temp/screenshots/`.

## Git Workflow

- **Branch**: feature branches from `main`
- **Strategy**: small frequent commits, auto-push to GitHub
- **No automatic PRs** — manual review before merging
- **Remote**: `origin` → `https://github.com/lijiawei255/voice-mood-detector.git`

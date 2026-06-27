"""
Headless GUI screenshot capture for CI/visual validation.
Requires PyQt5 and a display backend (Windows desktop or xvfb on Linux).
"""
import os
import sys

# Project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "windows")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from gui import MainWindow


def save_screenshot(widget, name):
    out_dir = os.path.join(ROOT, "portable_data", "temp", "screenshots")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    pixmap = widget.grab()
    ok = pixmap.save(path)
    print(f"[screenshot] {name}: {path} ({pixmap.width()}x{pixmap.height()}) saved={ok}")
    return path


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    test_audio = os.path.join(
        ROOT, "portable_data", "models", "models", "iic", "emotion2vec_plus_large", "example", "test.wav"
    )
    if not os.path.exists(test_audio):
        test_audio = os.path.join(ROOT, "portable_data", "recordings", "recording_20260624_185611.wav")

    def scroll_results(ratio):
        if hasattr(window, "result_scroll_area") and window.result_scroll_area:
            vbar = window.result_scroll_area.verticalScrollBar()
            if vbar and vbar.maximum() > 0:
                vbar.setValue(int(vbar.maximum() * ratio))
                app.processEvents()

    def wait_for_model(callback, attempts=0):
        if window.recognizer and window.recognizer.is_ready():
            callback()
        elif attempts < 60:
            QTimer.singleShot(1000, lambda: wait_for_model(callback, attempts + 1))
        else:
            print("[warn] model did not become ready in time")
            callback()

    def capture_result_sections(sections=None, idx=0):
        ratios = sections or [0.0, 0.30, 0.52, 0.72, 0.90]
        if idx < len(ratios):
            scroll_results(ratios[idx])
            QTimer.singleShot(400, lambda: (
                save_screenshot(window, f"05_realtime_result_section_{idx}"),
                capture_result_sections(ratios, idx + 1)
            ))
        else:
            # Switch to research mode UI
            window.mode_research.setChecked(True)
            QTimer.singleShot(400, lambda: (
                save_screenshot(window, "07_research_mode_ui"),
                window.mode_quick.setChecked(True),
                QTimer.singleShot(400, lambda: (
                    window.tab_widget.setCurrentIndex(1),
                    QTimer.singleShot(800, lambda: (
                        save_screenshot(window, "08_history_with_result"),
                        app.quit()
                    ))
                ))
            ))

    def run_analysis():
        if os.path.exists(test_audio):
            print(f"[info] analyzing {test_audio}")
            window.start_analysis(test_audio)

            def poll_analysis():
                if not window.is_analyzing and window.recognizer.is_ready():
                    QTimer.singleShot(800, lambda: capture_result_sections())
                else:
                    QTimer.singleShot(500, poll_analysis)
            QTimer.singleShot(2000, poll_analysis)
        else:
            print("[warn] no test audio found")
            app.quit()

    def step(idx=0):
        if idx == 0:
            save_screenshot(window, "01_main_initial")
            QTimer.singleShot(800, lambda: step(1))
        elif idx == 1:
            if hasattr(window, "baseline_panel"):
                window.baseline_panel.setVisible(True)
            save_screenshot(window, "02_main_panels_expanded")
            QTimer.singleShot(500, lambda: step(2))
        elif idx == 2:
            window.tab_widget.setCurrentIndex(1)
            QTimer.singleShot(800, lambda: step(3))
        elif idx == 3:
            save_screenshot(window, "03_history_tab")
            window.tab_widget.setCurrentIndex(0)
            QTimer.singleShot(500, lambda: wait_for_model(run_analysis))

    QTimer.singleShot(1500, lambda: step(0))
    app.exec_()


if __name__ == "__main__":
    main()

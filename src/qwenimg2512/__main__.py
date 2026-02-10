"""Entry point for Qwen-Image-2512 GUI."""

from __future__ import annotations

import logging
import queue
import sys
from logging.handlers import QueueHandler, QueueListener

from PySide6.QtWidgets import QApplication

from qwenimg2512.main_window import MainWindow
from qwenimg2512.styles.dark_theme import apply_dark_theme

_log_listener: QueueListener | None = None


def setup_thread_safe_logging() -> None:
    """Use QueueHandler/QueueListener to serialize log writes from multiple threads."""
    global _log_listener  # noqa: PLW0603
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    _log_listener = QueueListener(log_queue, stream_handler, respect_handler_level=True)
    _log_listener.start()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))


def main() -> None:
    setup_thread_safe_logging()

    app = QApplication(sys.argv)
    app.setApplicationName("Qwen-Image-2512")
    app.setOrganizationName("qwenimg2512")

    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

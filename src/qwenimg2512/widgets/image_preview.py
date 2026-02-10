"""Image preview widget for displaying generated images."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)


class ImagePreviewWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Preview")
        self._current_path: str | None = None
        self._pixmap: QPixmap | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.image_label = QLabel("No image generated yet")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setStyleSheet("border: 1px dashed #2a2a4a; border-radius: 8px;")
        layout.addWidget(self.image_label)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setProperty("class", "muted")
        layout.addWidget(self.info_label)

    def set_image(self, path: str) -> None:
        self._current_path = path
        self._pixmap = QPixmap(path)
        if self._pixmap.isNull():
            self.image_label.setText(f"Failed to load: {path}")
            return

        self._update_scaled_pixmap()

        file_size = Path(path).stat().st_size
        size_str = f"{file_size / 1024:.0f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
        self.info_label.setText(
            f"{self._pixmap.width()}x{self._pixmap.height()} | {size_str} | {Path(path).name}"
        )

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def clear(self) -> None:
        self._current_path = None
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image generated yet")
        self.info_label.setText("")

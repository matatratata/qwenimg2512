"""Image preview widget for displaying generated images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _pil_to_pixmap(pil_image: Image.Image) -> QPixmap:
    """Convert a PIL RGB image to QPixmap."""
    arr = np.asarray(pil_image)
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, c * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ImagePreviewWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Preview")
        self._current_path: str | None = None
        self._pixmap: QPixmap | None = None
        self._mode: str = "empty"  # "empty", "fit_preview", "generated"
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

        # Quick Actions Row
        self.actions_widget = QWidget()
        act_layout = QHBoxLayout(self.actions_widget)
        act_layout.setContentsMargins(0, 10, 0, 0)

        self.btn_send_edit = QPushButton("✏️ Send to Edit")
        self.btn_send_wan = QPushButton("🎬 Make Cinematic (Wan)")
        self.btn_send_seedvr2 = QPushButton("🔍 Upscale (SeedVR2)")

        act_layout.addWidget(self.btn_send_edit)
        act_layout.addWidget(self.btn_send_wan)
        act_layout.addWidget(self.btn_send_seedvr2)

        layout.addWidget(self.actions_widget)
        self.actions_widget.setVisible(False)

    def set_image(self, path: str) -> None:
        """Show a generated output image (replaces any fit preview)."""
        self._mode = "generated"
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
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            self.actions_widget.setVisible(True)
        else:
            self.actions_widget.setVisible(False)

    def show_fit_preview(self, images: list[tuple[Image.Image, str]], target_w: int, target_h: int) -> None:
        """Show cover-crop preview of input/control images at target dimensions.

        Args:
            images: List of (pil_image, label) pairs, already cropped to target dims.
            target_w: Target output width.
            target_h: Target output height.
        """
        if not images:
            self.clear_fit_preview()
            return

        self._mode = "fit_preview"
        self._current_path = None

        gap = 4
        if len(images) == 1:
            cropped, label = images[0]
            composite = cropped
            info = f"{label} — {target_w}x{target_h} cover crop"
        else:
            total_h = target_h * len(images) + gap * (len(images) - 1)
            composite = Image.new("RGB", (target_w, total_h), (30, 30, 30))
            info_parts = []
            for i, (cropped, label) in enumerate(images):
                y = i * (target_h + gap)
                composite.paste(cropped, (0, y))
                info_parts.append(f"{label}: {target_w}x{target_h}")
            info = " | ".join(info_parts)

        self._pixmap = _pil_to_pixmap(composite)
        self._update_scaled_pixmap()
        self.info_label.setText(info)

    def clear_fit_preview(self) -> None:
        """Clear fit preview and return to placeholder (only if not showing generated output)."""
        if self._mode == "generated":
            return
        self._mode = "empty"
        self._current_path = None
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image generated yet")
        self.info_label.setText("")
        self.actions_widget.setVisible(False)

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
        self._mode = "empty"
        self._current_path = None
        self._pixmap = None
        self.image_label.clear()
        self.image_label.setText("No image generated yet")
        self.info_label.setText("")
        self.actions_widget.setVisible(False)

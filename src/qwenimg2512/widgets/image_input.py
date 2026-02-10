"""Image input widget for img2img: browse, drag-drop, thumbnail, caption, strength."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ImageInputWidget(QGroupBox):
    image_loaded = Signal(str)  # image path
    image_cleared = Signal()
    caption_requested = Signal(str)  # image path
    strength_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__("Input Image (img2img)")
        self._image_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()
        self._update_visibility()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Thumbnail preview
        self.thumbnail_label = QLabel("Drop image here or click Browse")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(80)
        self.thumbnail_label.setStyleSheet("border: 1px dashed #4a4a6a; border-radius: 6px; padding: 8px;")
        layout.addWidget(self.thumbnail_label)

        # Buttons row
        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_image)
        btn_row.addWidget(self.browse_btn)

        self.caption_btn = QPushButton("Caption")
        self.caption_btn.setToolTip("Generate a prompt description from the image using VL model")
        self.caption_btn.clicked.connect(self._request_caption)
        self.caption_btn.setEnabled(False)
        btn_row.addWidget(self.caption_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

        # Strength spinbox row
        self.strength_row = QHBoxLayout()
        self.strength_label = QLabel("Strength:")
        self.strength_row.addWidget(self.strength_label)
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.0, 1.0)
        self.strength_spin.setValue(0.7)
        self.strength_spin.setSingleStep(0.05)
        self.strength_spin.setDecimals(2)
        self.strength_spin.setToolTip("How much to transform the input image (0=keep original, 1=full generation)")
        self.strength_spin.valueChanged.connect(self.strength_changed.emit)
        self.strength_row.addWidget(self.strength_spin, 1)

        self.strength_widget = QLabel()  # placeholder for layout
        layout.addLayout(self.strength_row)

    def _update_visibility(self) -> None:
        has_image = bool(self._image_path)
        self.caption_btn.setEnabled(has_image)
        self.clear_btn.setEnabled(has_image)
        self.strength_label.setVisible(has_image)
        self.strength_spin.setVisible(has_image)

    def _browse_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.set_image(path)

    def set_image(self, path: str) -> None:
        if not Path(path).is_file():
            return
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(160, Qt.TransformationMode.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setStyleSheet("")
        self._update_visibility()
        self.image_loaded.emit(path)

    def clear_image(self) -> None:
        self._image_path = ""
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("Drop image here or click Browse")
        self.thumbnail_label.setStyleSheet("border: 1px dashed #4a4a6a; border-radius: 6px; padding: 8px;")
        self._update_visibility()
        self.image_cleared.emit()

    def _request_caption(self) -> None:
        if self._image_path:
            self.caption_requested.emit(self._image_path)

    def get_image_path(self) -> str:
        return self._image_path

    def get_strength(self) -> float:
        return self.strength_spin.value()

    def set_strength(self, value: float) -> None:
        self.strength_spin.setValue(value)

    def set_captioning(self, active: bool) -> None:
        self.caption_btn.setEnabled(not active)
        self.caption_btn.setText("Captioning..." if active else "Caption")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.set_image(path)
                return

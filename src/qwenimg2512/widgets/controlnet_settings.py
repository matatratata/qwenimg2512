"""ControlNet Union settings widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

CONTROL_TYPES = ["Canny", "HED", "Depth", "Pose", "MLSD", "Scribble", "Gray"]


class ControlNetSettingsWidget(QGroupBox):
    settings_changed = Signal()
    caption_requested = Signal(str, str)  # image path, control type

    def __init__(self) -> None:
        super().__init__("ControlNet Union")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Enable checkbox
        self.enable_check = QCheckBox("Enable ControlNet")
        self.enable_check.toggled.connect(lambda _: self.settings_changed.emit())
        layout.addWidget(self.enable_check)

        # Control type
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Control Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(CONTROL_TYPES)
        self.type_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        type_row.addWidget(self.type_combo, 1)
        layout.addLayout(type_row)

        # Control image
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Control Image:"))
        self.image_edit = QLineEdit()
        self.image_edit.setPlaceholderText("No control image selected")
        self.image_edit.setReadOnly(True)
        img_row.addWidget(self.image_edit, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_image)
        img_row.addWidget(self.browse_btn)

        self.caption_btn = QPushButton("Caption")
        self.caption_btn.setToolTip("Generate a prompt description from the control image")
        self.caption_btn.clicked.connect(self._request_caption)
        self.caption_btn.setEnabled(False)
        img_row.addWidget(self.caption_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_image)
        img_row.addWidget(self.clear_btn)
        layout.addLayout(img_row)

        # Thumbnail preview
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedHeight(120)
        self.thumbnail_label.setScaledContents(False)
        self.thumbnail_label.hide()
        layout.addWidget(self.thumbnail_label)

        # Conditioning scale
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Conditioning Scale:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 2.0)
        self.scale_spin.setValue(0.80)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setDecimals(2)
        self.scale_spin.setToolTip("How strongly the control image influences generation")
        self.scale_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        scale_row.addWidget(self.scale_spin, 1)
        layout.addLayout(scale_row)

        # Guidance start / end
        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance Start:"))
        self.guidance_start_spin = QDoubleSpinBox()
        self.guidance_start_spin.setRange(0.0, 1.0)
        self.guidance_start_spin.setValue(0.0)
        self.guidance_start_spin.setSingleStep(0.05)
        self.guidance_start_spin.setDecimals(2)
        self.guidance_start_spin.setToolTip("Fraction of denoising where control begins")
        self.guidance_start_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        guidance_row.addWidget(self.guidance_start_spin, 1)

        guidance_row.addWidget(QLabel("End:"))
        self.guidance_end_spin = QDoubleSpinBox()
        self.guidance_end_spin.setRange(0.0, 1.0)
        self.guidance_end_spin.setValue(1.0)
        self.guidance_end_spin.setSingleStep(0.05)
        self.guidance_end_spin.setDecimals(2)
        self.guidance_end_spin.setToolTip("Fraction of denoising where control ends")
        self.guidance_end_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        guidance_row.addWidget(self.guidance_end_spin, 1)
        layout.addLayout(guidance_row)

    def _browse_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Control Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.image_edit.setText(path)
            self._update_thumbnail(path)
            self.settings_changed.emit()

    def _clear_image(self) -> None:
        self.image_edit.clear()
        self.thumbnail_label.clear()
        self.thumbnail_label.hide()
        self.caption_btn.setEnabled(False)
        self.settings_changed.emit()

    def _update_thumbnail(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.thumbnail_label.hide()
            self.caption_btn.setEnabled(False)
            return
        scaled = pixmap.scaledToHeight(120)
        self.thumbnail_label.setPixmap(scaled)
        self.thumbnail_label.show()
        self.caption_btn.setEnabled(True)

    def _request_caption(self) -> None:
        path = self.image_edit.text()
        if path and Path(path).is_file():
            control_type = self.type_combo.currentText()
            self.caption_requested.emit(path, control_type)

    def set_captioning(self, active: bool) -> None:
        self.caption_btn.setEnabled(not active)
        self.caption_btn.setText("Captioning..." if active else "Caption")

    # --- Getters / Setters ---

    def is_enabled(self) -> bool:
        return self.enable_check.isChecked()

    def set_enabled(self, enabled: bool) -> None:
        self.enable_check.setChecked(enabled)

    def get_control_type(self) -> str:
        return self.type_combo.currentText().lower()

    def set_control_type(self, value: str) -> None:
        idx = self.type_combo.findText(value, flags=Qt.MatchFlag.MatchFixedString)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)

    def get_control_image_path(self) -> str:
        return self.image_edit.text()

    def set_control_image_path(self, path: str) -> None:
        self.image_edit.setText(path)
        if path and Path(path).is_file():
            self._update_thumbnail(path)
        else:
            self.thumbnail_label.hide()

    def get_conditioning_scale(self) -> float:
        return self.scale_spin.value()

    def set_conditioning_scale(self, value: float) -> None:
        self.scale_spin.setValue(value)

    def get_guidance_start(self) -> float:
        return self.guidance_start_spin.value()

    def set_guidance_start(self, value: float) -> None:
        self.guidance_start_spin.setValue(value)

    def get_guidance_end(self) -> float:
        return self.guidance_end_spin.value()

    def set_guidance_end(self, value: float) -> None:
        self.guidance_end_spin.setValue(value)

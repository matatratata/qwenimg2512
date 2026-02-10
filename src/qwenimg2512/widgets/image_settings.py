"""Image generation settings widget (resolution, steps, CFG, model)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from qwenimg2512.config import ASPECT_RATIOS, MODEL_VARIANTS


class ImageSettingsWidget(QGroupBox):
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Image Settings")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Model variant
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_VARIANTS.keys())
        self.model_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        model_row.addWidget(self.model_combo, 1)
        layout.addLayout(model_row)

        # Aspect ratio
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Aspect Ratio:"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(ASPECT_RATIOS.keys())
        self.ratio_combo.currentTextChanged.connect(lambda: self.settings_changed.emit())
        ratio_row.addWidget(self.ratio_combo, 1)
        layout.addLayout(ratio_row)

        # Inference steps
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(10, 100)
        self.steps_spin.setValue(50)
        self.steps_spin.setSingleStep(5)
        self.steps_spin.setToolTip("Number of denoising steps (50 recommended)")
        self.steps_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        steps_row.addWidget(self.steps_spin, 1)
        layout.addLayout(steps_row)

        # True CFG scale
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("CFG Scale:"))
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 10.0)
        self.cfg_spin.setValue(4.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setDecimals(1)
        self.cfg_spin.setToolTip("Classifier-free guidance scale (4.0 recommended)")
        self.cfg_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        cfg_row.addWidget(self.cfg_spin, 1)
        layout.addLayout(cfg_row)

        # Guidance scale (usually fixed at 1.0)
        guidance_row = QHBoxLayout()
        guidance_row.addWidget(QLabel("Guidance:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(0.0, 5.0)
        self.guidance_spin.setValue(1.0)
        self.guidance_spin.setSingleStep(0.1)
        self.guidance_spin.setDecimals(1)
        self.guidance_spin.setToolTip("Guidance scale (1.0 recommended, usually kept fixed)")
        self.guidance_spin.valueChanged.connect(lambda: self.settings_changed.emit())
        guidance_row.addWidget(self.guidance_spin, 1)
        layout.addLayout(guidance_row)

        # Resolution preview
        self.resolution_label = QLabel()
        self.resolution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resolution_label.setProperty("class", "muted")
        self._update_resolution_label()
        self.ratio_combo.currentTextChanged.connect(lambda: self._update_resolution_label())
        layout.addWidget(self.resolution_label)

    def _update_resolution_label(self) -> None:
        w, h = self.get_resolution()
        megapixels = (w * h) / 1_000_000
        self.resolution_label.setText(f"{w} x {h} ({megapixels:.1f} MP)")

    def get_resolution(self) -> tuple[int, int]:
        return ASPECT_RATIOS[self.ratio_combo.currentText()]

    def get_steps(self) -> int:
        return self.steps_spin.value()

    def get_cfg_scale(self) -> float:
        return self.cfg_spin.value()

    def get_guidance_scale(self) -> float:
        return self.guidance_spin.value()

    def get_model_variant(self) -> str:
        return self.model_combo.currentText()

    def get_aspect_ratio(self) -> str:
        return self.ratio_combo.currentText()

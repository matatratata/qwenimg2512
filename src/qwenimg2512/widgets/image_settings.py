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
from qwenimg2512.samplers import SAMPLER_NAMES, SAMPLER_DESCRIPTIONS
from qwenimg2512.schedules import SCHEDULE_NAMES, SCHEDULE_DESCRIPTIONS


class ImageSettingsWidget(QGroupBox):
    settings_changed = Signal()

    def __init__(self) -> None:
        super().__init__("Image Settings")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Model variant
        model_row = QHBoxLayout()
        self.model_label = QLabel("Model:")
        model_row.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_VARIANTS.keys())
        self.model_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        model_row.addWidget(self.model_combo, 1)
        layout.addLayout(model_row)

        # Aspect ratio
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Aspect Ratio:"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(ASPECT_RATIOS.keys())
        self.ratio_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        ratio_row.addWidget(self.ratio_combo, 1)
        layout.addLayout(ratio_row)

        # Sampler
        sampler_row = QHBoxLayout()
        sampler_row.addWidget(QLabel("Sampler:"))
        self.sampler_combo = QComboBox()
        for name in SAMPLER_NAMES:
            self.sampler_combo.addItem(name)
            
        for i, name in enumerate(SAMPLER_NAMES):
            self.sampler_combo.setItemData(i, SAMPLER_DESCRIPTIONS.get(name, ""), Qt.ItemDataRole.ToolTipRole)
            
        self.sampler_combo.setToolTip("Select the generation sampler")
        self.sampler_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        sampler_row.addWidget(self.sampler_combo, 1)
        layout.addLayout(sampler_row)

        # Schedule
        schedule_row = QHBoxLayout()
        schedule_row.addWidget(QLabel("Schedule:"))
        self.schedule_combo = QComboBox()
        for name in SCHEDULE_NAMES:
            self.schedule_combo.addItem(name)
            
        for i, name in enumerate(SCHEDULE_NAMES):
            self.schedule_combo.setItemData(i, SCHEDULE_DESCRIPTIONS.get(name, ""), Qt.ItemDataRole.ToolTipRole)
            
        self.schedule_combo.setToolTip("Select the sigma schedule (beta57/bong_tangent)")
        self.schedule_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        schedule_row.addWidget(self.schedule_combo, 1)
        layout.addLayout(schedule_row)

        # Inference steps
        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(50)
        self.steps_spin.setSingleStep(5)
        self.steps_spin.setToolTip("Number of denoising steps (50 recommended)")
        self.steps_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        steps_row.addWidget(self.steps_spin, 1)
        layout.addLayout(steps_row)

        # Effective steps indicator (visible only during img2img)
        self.effective_steps_label = QLabel()
        self.effective_steps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.effective_steps_label.setStyleSheet("color: #e0a040; font-size: 11px;")
        self.effective_steps_label.setVisible(False)
        layout.addWidget(self.effective_steps_label)

        # True CFG scale
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("CFG Scale:"))
        self.cfg_spin = QDoubleSpinBox()
        self.cfg_spin.setRange(1.0, 20.0)
        self.cfg_spin.setValue(4.0)
        self.cfg_spin.setSingleStep(0.5)
        self.cfg_spin.setDecimals(1)
        self.cfg_spin.setToolTip("Classifier-free guidance scale (4.0 recommended)")
        self.cfg_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
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
        self.guidance_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        guidance_row.addWidget(self.guidance_spin, 1)
        layout.addLayout(guidance_row)

        # Resolution preview
        self.resolution_label = QLabel()
        self.resolution_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resolution_label.setProperty("class", "muted")
        self._update_resolution_label()
        self.ratio_combo.currentTextChanged.connect(lambda _: self._update_resolution_label())
        layout.addWidget(self.resolution_label)

    def _update_resolution_label(self) -> None:
        w, h = self.get_resolution()
        megapixels = (w * h) / 1_000_000
        self.resolution_label.setText(f"{w} x {h} ({megapixels:.1f} MP)")

    def get_resolution(self) -> tuple[int, int]:
        return ASPECT_RATIOS[self.ratio_combo.currentText()]

    def get_steps(self) -> int:
        return self.steps_spin.value()

    def get_sampler_name(self) -> str:
        return self.sampler_combo.currentText()

    def get_schedule_name(self) -> str:
        return self.schedule_combo.currentText()

    def get_cfg_scale(self) -> float:
        return self.cfg_spin.value()

    def get_guidance_scale(self) -> float:
        return self.guidance_spin.value()

    def get_model_variant(self) -> str:
        return self.model_combo.currentText()

    def get_aspect_ratio(self) -> str:
        return self.ratio_combo.currentText()

    def update_effective_steps(self, has_image: bool, strength: float) -> None:
        """Update the effective steps indicator for img2img mode."""
        steps = self.steps_spin.value()
        if not has_image or strength >= 1.0:
            self.effective_steps_label.setVisible(False)
            return
        effective = max(1, int(steps * strength))
        self.effective_steps_label.setText(f"⚠ Effective: {effective} steps (strength {strength:.2f})")
        self.effective_steps_label.setVisible(True)

"""Wan Cinematic I2V tab widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.widgets.generation_controls import GenerationControlsWidget
from qwenimg2512.widgets.prompt_input import PromptInputWidget
from qwenimg2512.widgets.seedvr2_tab import SeedVR2InputWidget


class WanTabWidget(QWidget):
    generate_requested = Signal()
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 4, 4)

        self.input_widget = SeedVR2InputWidget("Input Image (Starting Frame)")
        content_layout.addWidget(self.input_widget)

        self.prompt_widget = PromptInputWidget()
        self.prompt_widget.positive_input.setPlaceholderText(
            "Cinematic prompt to guide the motion and lighting.\n"
            "Example: 'Cinematic slow pan, highly detailed, dramatic volumetric lighting, anamorphic lens flare'"
        )
        content_layout.addWidget(self.prompt_widget)

        settings_group = QGroupBox("Cinematic Settings")
        settings_layout = QVBoxLayout(settings_group)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.res_combo = QComboBox()
        self.res_combo.addItems(["832x480", "1280x720", "480x832", "720x1280"])
        res_row.addWidget(self.res_combo, 1)

        res_row.addWidget(QLabel("Frames:"))
        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 100)
        self.frames_spin.setValue(33)
        self.frames_spin.setToolTip(
            "33 recommended for cinematic stills (allows the 3D lighting to settle)."
        )
        res_row.addWidget(self.frames_spin, 1)
        settings_layout.addLayout(res_row)

        steps_row = QHBoxLayout()
        steps_row.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(1, 100)
        self.steps_spin.setValue(40)
        steps_row.addWidget(self.steps_spin, 1)

        steps_row.addWidget(QLabel("Guidance Scale:"))
        self.guidance_spin = QDoubleSpinBox()
        self.guidance_spin.setRange(1.0, 10.0)
        self.guidance_spin.setValue(5.0)
        self.guidance_spin.setSingleStep(0.5)
        steps_row.addWidget(self.guidance_spin, 1)

        steps_row.addWidget(QLabel("Flow Shift:"))
        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(1.0, 15.0)
        self.shift_spin.setValue(5.0)
        self.shift_spin.setSingleStep(0.5)
        steps_row.addWidget(self.shift_spin, 1)
        settings_layout.addLayout(steps_row)

        self.extract_still_check = QCheckBox("Extract Cinematic Still (PNG)")
        self.extract_still_check.setChecked(True)
        self.extract_still_check.setToolTip(
            "Automatically extract and save the middle frame as a cinematic still image."
        )
        settings_layout.addWidget(self.extract_still_check)

        content_layout.addWidget(settings_group)

        self.gen_controls = GenerationControlsWidget()
        self.gen_controls.generate_clicked.connect(self.generate_requested)
        self.gen_controls.cancel_clicked.connect(self.cancel_requested)
        content_layout.addWidget(self.gen_controls)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def set_generating(self, generating: bool) -> None:
        self.gen_controls.set_generating(generating)
        self.input_widget.setEnabled(not generating)
        self.prompt_widget.setEnabled(not generating)
        self.res_combo.setEnabled(not generating)
        self.frames_spin.setEnabled(not generating)
        self.steps_spin.setEnabled(not generating)
        self.guidance_spin.setEnabled(not generating)
        self.shift_spin.setEnabled(not generating)
        self.extract_still_check.setEnabled(not generating)

    def set_progress(self, current: int, total: int, message: str) -> None:
        self.gen_controls.set_progress(current, total, message)

    def set_stage(self, stage: str) -> None:
        self.gen_controls.set_stage(stage)

    def set_vram(self, gb: float) -> None:
        self.gen_controls.set_vram(gb)

    def set_finished(self, output_path: str) -> None:
        self.gen_controls.set_finished(output_path)

    def set_error(self, error: str) -> None:
        self.gen_controls.set_error(error)

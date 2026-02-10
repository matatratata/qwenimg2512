"""LoRA adapter settings widget: path, scale ramp, step range."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class LoraSettingsWidget(QGroupBox):
    def __init__(self) -> None:
        super().__init__("LoRA Adapter")
        self._setup_ui()
        self._update_visibility()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # LoRA path row
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("LoRA:"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No LoRA loaded")
        self.path_edit.setReadOnly(True)
        path_row.addWidget(self.path_edit, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_lora)
        path_row.addWidget(self.browse_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_lora)
        path_row.addWidget(self.clear_btn)

        layout.addLayout(path_row)

        # Controls container (hidden when no LoRA loaded)
        self._controls = QWidget()
        controls_layout = QVBoxLayout(self._controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        # Scale row
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale Start:"))
        self.scale_start_spin = QDoubleSpinBox()
        self.scale_start_spin.setRange(0.0, 2.0)
        self.scale_start_spin.setValue(1.0)
        self.scale_start_spin.setSingleStep(0.05)
        self.scale_start_spin.setDecimals(2)
        self.scale_start_spin.setToolTip("LoRA scale at the beginning of the active step range")
        scale_row.addWidget(self.scale_start_spin, 1)

        scale_row.addWidget(QLabel("End:"))
        self.scale_end_spin = QDoubleSpinBox()
        self.scale_end_spin.setRange(0.0, 2.0)
        self.scale_end_spin.setValue(1.0)
        self.scale_end_spin.setSingleStep(0.05)
        self.scale_end_spin.setDecimals(2)
        self.scale_end_spin.setToolTip("LoRA scale at the end of the active step range")
        scale_row.addWidget(self.scale_end_spin, 1)
        controls_layout.addLayout(scale_row)

        # Step range row
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step Start:"))
        self.step_start_spin = QSpinBox()
        self.step_start_spin.setRange(0, 100)
        self.step_start_spin.setValue(0)
        self.step_start_spin.setToolTip("First step where LoRA is active")
        step_row.addWidget(self.step_start_spin, 1)

        step_row.addWidget(QLabel("End:"))
        self.step_end_spin = QSpinBox()
        self.step_end_spin.setRange(-1, 100)
        self.step_end_spin.setValue(-1)
        self.step_end_spin.setToolTip("Last step where LoRA is active (-1 = last step)")
        step_row.addWidget(self.step_end_spin, 1)
        controls_layout.addLayout(step_row)

        layout.addWidget(self._controls)

    def _update_visibility(self) -> None:
        has_lora = bool(self.path_edit.text())
        self._controls.setVisible(has_lora)
        self.clear_btn.setEnabled(has_lora)

    def _browse_lora(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LoRA File",
            str(Path.home()),
            "Safetensors (*.safetensors)",
        )
        if path:
            self.path_edit.setText(path)
            self._update_visibility()

    def _clear_lora(self) -> None:
        self.path_edit.clear()
        self._update_visibility()

    def get_lora_path(self) -> str:
        return self.path_edit.text()

    def set_lora_path(self, path: str) -> None:
        self.path_edit.setText(path)
        self._update_visibility()

    def get_scale_start(self) -> float:
        return self.scale_start_spin.value()

    def set_scale_start(self, value: float) -> None:
        self.scale_start_spin.setValue(value)

    def get_scale_end(self) -> float:
        return self.scale_end_spin.value()

    def set_scale_end(self, value: float) -> None:
        self.scale_end_spin.setValue(value)

    def get_step_start(self) -> int:
        return self.step_start_spin.value()

    def set_step_start(self, value: int) -> None:
        self.step_start_spin.setValue(value)

    def get_step_end(self) -> int:
        return self.step_end_spin.value()

    def set_step_end(self, value: int) -> None:
        self.step_end_spin.setValue(value)

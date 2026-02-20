"""Generation controls: seed, output path, generate/cancel buttons, progress."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class GenerationControlsWidget(QGroupBox):
    generate_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self) -> None:
        super().__init__("Generation")
        self._generating = False
        self._start_time: float | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Seed row
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2147483647)
        self.seed_spin.setValue(-1)
        self.seed_spin.setSpecialValueText("Random")
        self.seed_spin.setToolTip("-1 = random seed")
        seed_row.addWidget(self.seed_spin, 1)
        layout.addLayout(seed_row)

        # Output directory
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.output_edit = QLineEdit()
        self.output_edit.setText(str(Path.home() / "Pictures" / "qwenimg2512"))
        output_row.addWidget(self.output_edit, 1)
        self.browse_btn = QPushButton("...")
        self.browse_btn.setMaximumWidth(40)
        self.browse_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self.browse_btn)
        layout.addLayout(output_row)

        # Generate / Cancel button
        self.generate_btn = QPushButton("GENERATE")
        self.generate_btn.setProperty("class", "primary")
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_btn.clicked.connect(self._on_generate_clicked)
        layout.addWidget(self.generate_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        layout.addWidget(self.progress_bar)

        # Status row
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setProperty("class", "muted")
        status_row.addWidget(self.status_label, 1)
        self.vram_label = QLabel("")
        self.vram_label.setProperty("class", "muted")
        self.vram_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_row.addWidget(self.vram_label)
        self.time_label = QLabel("")
        self.time_label.setProperty("class", "muted")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_row.addWidget(self.time_label)
        layout.addLayout(status_row)

    def _browse_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_edit.text())
        if directory:
            self.output_edit.setText(directory)

    def _on_generate_clicked(self) -> None:
        if self._generating:
            self.cancel_clicked.emit()
        else:
            self.generate_clicked.emit()

    def set_generating(self, generating: bool) -> None:
        self._generating = generating
        if generating:
            self._start_time = time.monotonic()
            self.generate_btn.setText("CANCEL")
            self.generate_btn.setProperty("class", "danger")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Starting...")
        else:
            self._start_time = None
            self.generate_btn.setText("GENERATE")
            self.generate_btn.setProperty("class", "primary")

        self.generate_btn.style().unpolish(self.generate_btn)
        self.generate_btn.style().polish(self.generate_btn)

    def set_progress(self, current: int, total: int, message: str) -> None:
        percentage = int(100 * current / max(total, 1))
        self.progress_bar.setValue(percentage)
        self.progress_bar.setFormat(f"{percentage}% ({current}/{total})")
        self.status_label.setText(message)

        if self._start_time:
            elapsed = int(time.monotonic() - self._start_time)
            minutes, seconds = divmod(elapsed, 60)
            self.time_label.setText(f"{minutes}:{seconds:02d}")

    def set_stage(self, stage: str) -> None:
        self.status_label.setText(stage)
        self.progress_bar.setFormat(stage)

    def set_vram(self, gb: float) -> None:
        self.vram_label.setText(f"VRAM: {gb:.1f} GB")

    def set_finished(self, output_path: str) -> None:
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("Complete!")
        self.status_label.setText(f"Saved: {output_path}")

    def set_error(self, error: str) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Error")
        self.status_label.setText(f"Error: {error}")

    def get_seed(self) -> int:
        return self.seed_spin.value()

    def get_output_dir(self) -> str:
        return self.output_edit.text()

    def set_output_dir(self, output_dir: str) -> None:
        self.output_edit.setText(output_dir)

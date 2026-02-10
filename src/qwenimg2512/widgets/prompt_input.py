"""Prompt and negative prompt input widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)


class PromptInputWidget(QGroupBox):
    prompt_changed = Signal(str)
    negative_prompt_changed = Signal(str)

    def __init__(self) -> None:
        super().__init__("Prompts")
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Positive Prompt"))
        self.positive_input = QPlainTextEdit()
        self.positive_input.setPlaceholderText("Describe the image you want to generate...")
        self.positive_input.setMinimumHeight(80)
        self.positive_input.setMaximumHeight(150)
        self.positive_input.textChanged.connect(lambda: self.prompt_changed.emit(self.positive_input.toPlainText()))
        layout.addWidget(self.positive_input)

        self.positive_counter = QLabel("0 chars")
        self.positive_counter.setProperty("class", "muted")
        self.positive_input.textChanged.connect(
            lambda: self.positive_counter.setText(f"{len(self.positive_input.toPlainText())} chars")
        )
        layout.addWidget(self.positive_counter)

        layout.addWidget(QLabel("Negative Prompt"))
        self.negative_input = QPlainTextEdit()
        self.negative_input.setPlaceholderText("What to avoid in the image...")
        self.negative_input.setMinimumHeight(50)
        self.negative_input.setMaximumHeight(100)
        self.negative_input.textChanged.connect(
            lambda: self.negative_prompt_changed.emit(self.negative_input.toPlainText())
        )
        layout.addWidget(self.negative_input)

    def get_prompt(self) -> str:
        return self.positive_input.toPlainText().strip()

    def get_negative_prompt(self) -> str:
        return self.negative_input.toPlainText().strip()

    def set_prompt(self, text: str) -> None:
        self.positive_input.setPlainText(text)

    def set_negative_prompt(self, text: str) -> None:
        self.negative_input.setPlainText(text)

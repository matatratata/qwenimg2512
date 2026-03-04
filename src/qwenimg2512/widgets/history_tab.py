"""History tab widget."""

import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QTextEdit, 
    QSplitter, QPushButton
)
from PySide6.QtCore import Signal, Qt
from qwenimg2512.history import HistoryManager

class HistoryTabWidget(QWidget):
    # Emits (output_path, ref1_path_or_empty, fit_mode)
    image_selected = Signal(str, str, str)
    # Emits (tab_name, params_dict)
    load_settings_requested = Signal(str, dict)

    def __init__(self, history_manager: HistoryManager) -> None:
        super().__init__()
        self.history_manager = history_manager
        self._setup_ui()
        self.refresh()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        top_layout = QHBoxLayout()
        self.btn_load_settings = QPushButton("Load Settings")
        self.btn_load_settings.setEnabled(False)
        self.btn_load_settings.clicked.connect(self._on_load_settings)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)
        
        top_layout.addWidget(self.btn_load_settings)
        top_layout.addStretch()
        top_layout.addWidget(refresh_btn)
        layout.addLayout(top_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_item_selected)

        self.details_edit = QTextEdit()
        self.details_edit.setReadOnly(True)

        splitter.addWidget(self.list_widget)
        splitter.addWidget(self.details_edit)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)

    def refresh(self) -> None:
        self.list_widget.clear()
        self.history_manager.load()
        for entry in self.history_manager.get_history():
            ts = entry.get("timestamp", "")
            tab = entry.get("tab_name", "")
            params = entry.get("params", {})
            prompt = params.get("prompt", "")
            if not prompt and "input_image" in params:
                prompt = "[Image Input] " + str(params.get("input_image", ""))
            elif not prompt:
                prompt = "[No prompt]"
                
            display = f"{ts[:19].replace('T', ' ')} | {tab}\n{str(prompt)[:60]}..."
            self.list_widget.addItem(display)

    def _on_item_selected(self, index: int) -> None:
        history = self.history_manager.get_history()
        if index < 0 or index >= len(history):
            self.details_edit.clear()
            self.btn_load_settings.setEnabled(False)
            return
            
        self.btn_load_settings.setEnabled(True)
        entry = history[index]
        formatted = json.dumps(entry, indent=2, ensure_ascii=False)
        self.details_edit.setText(formatted)
        
        output_path = entry.get("output_path", "")
        if output_path:
            params = entry.get("params", {})
            ref1 = params.get("ref_image_1", "")
            fit_mode = params.get("ref_fit_mode_1", "cover")
            self.image_selected.emit(output_path, ref1 or "", fit_mode)

    def _on_load_settings(self) -> None:
        index = self.list_widget.currentRow()
        history = self.history_manager.get_history()
        if 0 <= index < len(history):
            entry = history[index]
            tab_name = entry.get("tab_name", "")
            params = entry.get("params", {})
            self.load_settings_requested.emit(tab_name, params)


"""Dialog for configuring local model file paths."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from qwenimg2512.config import ModelPaths


class ModelPathsDialog(QDialog):
    def __init__(self, model_paths: ModelPaths, parent: object = None) -> None:
        super().__init__(parent)
        self._model_paths = model_paths
        self.setWindowTitle("Model Paths")
        self.setMinimumWidth(600)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Configure paths to local GGUF model files:"))

        self._fields: dict[str, QLineEdit] = {}

        labels = {
            "diffusion_gguf": "Diffusion GGUF:",
            "vl_model": "VL Model:",
            "mmproj": "Vision Projector (mmproj):",
            "vae": "VAE:",
            "controlnet_path": "ControlNet Union:",
        }

        filters = {
            "diffusion_gguf": "GGUF Files (*.gguf)",
            "vl_model": "GGUF Files (*.gguf)",
            "mmproj": "GGUF Files (*.gguf)",
            "vae": "SafeTensors Files (*.safetensors)",
            "controlnet_path": "SafeTensors Files (*.safetensors)",
        }

        for field_name, label_text in labels.items():
            layout.addWidget(QLabel(label_text))
            row = QHBoxLayout()

            edit = QLineEdit()
            edit.setText(getattr(self._model_paths, field_name))
            edit.textChanged.connect(lambda text, fn=field_name: self._validate_path(fn, text))
            row.addWidget(edit, 1)
            self._fields[field_name] = edit

            browse_btn = QPushButton("...")
            browse_btn.setMaximumWidth(40)
            file_filter = filters[field_name]
            browse_btn.clicked.connect(lambda checked=False, e=edit, f=file_filter: self._browse(e, f))
            row.addWidget(browse_btn)

            layout.addLayout(row)

            # Status label for validation
            status = QLabel()
            status.setProperty("class", "muted")
            status.setObjectName(f"status_{field_name}")
            layout.addWidget(status)
            self._validate_path(field_name, getattr(self._model_paths, field_name))

        # Base model directory (for fully offline GGUF pipeline)
        layout.addWidget(QLabel(""))
        hint = QLabel("Base model directory (optional, for offline GGUF mode):")
        hint.setToolTip(
            "Local directory containing the full Qwen-Image-2512 model "
            "(text_encoder, vae, scheduler, tokenizer). When set, the GGUF "
            "pipeline loads entirely from disk without HuggingFace downloads."
        )
        layout.addWidget(hint)
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setText(self._model_paths.base_model_dir)
        edit.setPlaceholderText("Leave empty to download from HuggingFace")
        edit.textChanged.connect(lambda text: self._validate_dir("base_model_dir", text))
        row.addWidget(edit, 1)
        self._fields["base_model_dir"] = edit

        browse_btn = QPushButton("...")
        browse_btn.setMaximumWidth(40)
        browse_btn.clicked.connect(lambda checked=False, e=edit: self._browse_dir(e))
        row.addWidget(browse_btn)
        layout.addLayout(row)

        status = QLabel()
        status.setProperty("class", "muted")
        status.setObjectName("status_base_model_dir")
        layout.addWidget(status)
        self._validate_dir("base_model_dir", self._model_paths.base_model_dir)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self, edit: QLineEdit, file_filter: str) -> None:
        current = edit.text()
        start_dir = str(Path(current).parent) if current and Path(current).parent.exists() else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", start_dir, file_filter)
        if path:
            edit.setText(path)

    def _browse_dir(self, edit: QLineEdit) -> None:
        current = edit.text()
        start_dir = current if current and Path(current).is_dir() else str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Select Base Model Directory", start_dir)
        if path:
            edit.setText(path)

    def _validate_dir(self, field_name: str, text: str) -> None:
        status_label = self.findChild(QLabel, f"status_{field_name}")
        if not status_label:
            return
        if not text:
            status_label.setText("Not set (will use HuggingFace)")
        elif Path(text).is_dir():
            expected = ["model_index.json", "text_encoder", "vae", "scheduler", "tokenizer"]
            found = [e for e in expected if (Path(text) / e).exists()]
            if len(found) == len(expected):
                status_label.setText("Valid model directory")
            else:
                missing = [e for e in expected if e not in found]
                status_label.setText(f"Missing: {', '.join(missing)}")
        else:
            status_label.setText("Directory not found")

    def _validate_path(self, field_name: str, text: str) -> None:
        status_label = self.findChild(QLabel, f"status_{field_name}")
        if not status_label:
            return
        path = Path(text)
        if not text:
            status_label.setText("Not set")
        elif path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            status_label.setText(f"Found ({size_mb:.0f} MB)")
        else:
            status_label.setText("File not found")

    def _apply_and_accept(self) -> None:
        for field_name, edit in self._fields.items():
            setattr(self._model_paths, field_name, edit.text())
        self.accept()

    def get_model_paths(self) -> ModelPaths:
        return self._model_paths

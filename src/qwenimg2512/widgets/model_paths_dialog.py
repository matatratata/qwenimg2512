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
            "edit_gguf": "Edit 2511 GGUF:",
            "llama_cpp_cli": "llama-mtmd-cli Path:",
            "edit_2509_gguf": "Edit 2509 GGUF:",
            "telestyle_lora": "TeleStyle LoRA:",
            "telestyle_speedup": "TeleStyle Speedup LoRA:",
            "seedvr2_gguf": "SeedVR2 GGUF:",
            "seedvr2_vae": "SeedVR2 VAE:",
            "seedvr2_cli": "SeedVR2 CLI (inference_cli.py):",
            "wan_gguf": "Wan 2.2 GGUF:",
        }

        filters = {
            "diffusion_gguf": "GGUF Files (*.gguf)",
            "vl_model": "GGUF Files (*.gguf)",
            "mmproj": "GGUF Files (*.gguf)",
            "vae": "SafeTensors Files (*.safetensors)",
            "controlnet_path": "SafeTensors Files (*.safetensors)",
            "edit_gguf": "GGUF Files (*.gguf)",
            "llama_cpp_cli": "All Files (*)",
            "edit_2509_gguf": "GGUF Files (*.gguf)",
            "telestyle_lora": "SafeTensors Files (*.safetensors)",
            "telestyle_speedup": "SafeTensors Files (*.safetensors)",
            "seedvr2_gguf": "GGUF Files (*.gguf)",
            "seedvr2_vae": "PyTorch Files (*.pth)",
            "seedvr2_cli": "Python Files (*.py)",
            "wan_gguf": "GGUF Files (*.gguf)",
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

        # Edit-2511 Base model directory
        layout.addWidget(QLabel(""))
        hint_edit = QLabel("Edit-2511 Base model directory (optional):")
        hint_edit.setToolTip(
            "Local directory containing the Qwen-Image-Edit-2511 model components. "
            "Used for the Edit tab."
        )
        layout.addWidget(hint_edit)
        row_edit = QHBoxLayout()
        edit_dir = QLineEdit()
        edit_dir.setText(self._model_paths.edit_base_model_dir)
        edit_dir.setPlaceholderText("Leave empty to download from HuggingFace")
        edit_dir.textChanged.connect(lambda text: self._validate_dir("edit_base_model_dir", text))
        row_edit.addWidget(edit_dir, 1)
        self._fields["edit_base_model_dir"] = edit_dir

        browse_btn_edit = QPushButton("...")
        browse_btn_edit.setMaximumWidth(40)
        browse_btn_edit.clicked.connect(lambda checked=False, e=edit_dir: self._browse_dir(e))
        row_edit.addWidget(browse_btn_edit)
        layout.addLayout(row_edit)

        status_edit = QLabel()
        status_edit.setProperty("class", "muted")
        status_edit.setObjectName("status_edit_base_model_dir")
        layout.addWidget(status_edit)
        self._validate_dir("edit_base_model_dir", self._model_paths.edit_base_model_dir)

        # Edit-2509 Base model directory
        layout.addWidget(QLabel(""))
        hint_edit_2509 = QLabel("Edit-2509 Base model directory (optional):")
        hint_edit_2509.setToolTip(
            "Local directory containing the Qwen-Image-Edit-2509 model components. "
            "Used for the Edit (2509) tab."
        )
        layout.addWidget(hint_edit_2509)
        row_edit_2509 = QHBoxLayout()
        edit_dir_2509 = QLineEdit()
        edit_dir_2509.setText(getattr(self._model_paths, "edit_2509_base_model_dir"))
        edit_dir_2509.setPlaceholderText("Leave empty to download from HuggingFace")
        edit_dir_2509.textChanged.connect(lambda text: self._validate_dir("edit_2509_base_model_dir", text))
        row_edit_2509.addWidget(edit_dir_2509, 1)
        self._fields["edit_2509_base_model_dir"] = edit_dir_2509

        browse_btn_edit_2509 = QPushButton("...")
        browse_btn_edit_2509.setMaximumWidth(40)
        browse_btn_edit_2509.clicked.connect(lambda checked=False, e=edit_dir_2509: self._browse_dir(e))
        row_edit_2509.addWidget(browse_btn_edit_2509)
        layout.addLayout(row_edit_2509)

        status_edit_2509 = QLabel()
        status_edit_2509.setProperty("class", "muted")
        status_edit_2509.setObjectName("status_edit_2509_base_model_dir")
        layout.addWidget(status_edit_2509)
        self._validate_dir("edit_2509_base_model_dir", getattr(self._model_paths, "edit_2509_base_model_dir"))

        # Edit-2509 TeleStyle Fused model directory
        layout.addWidget(QLabel(""))
        hint_fused = QLabel("Edit-2509 TeleStyle Fused model directory:")
        hint_fused.setToolTip(
            "Directory containing the fused TeleStyle+Lightning model. "
            "Created by running fuse_telestyle.py once."
        )
        layout.addWidget(hint_fused)
        row_fused = QHBoxLayout()
        edit_dir_fused = QLineEdit()
        edit_dir_fused.setText(getattr(self._model_paths, "edit_2509_telestyle_fused_dir", ""))
        edit_dir_fused.setPlaceholderText("Path to fused model (run fuse_telestyle.py first)")
        edit_dir_fused.textChanged.connect(lambda text: self._validate_dir("edit_2509_telestyle_fused_dir", text))
        row_fused.addWidget(edit_dir_fused, 1)
        self._fields["edit_2509_telestyle_fused_dir"] = edit_dir_fused

        browse_btn_fused = QPushButton("...")
        browse_btn_fused.setMaximumWidth(40)
        browse_btn_fused.clicked.connect(lambda checked=False, e=edit_dir_fused: self._browse_dir(e))
        row_fused.addWidget(browse_btn_fused)
        layout.addLayout(row_fused)

        status_fused = QLabel()
        status_fused.setProperty("class", "muted")
        status_fused.setObjectName("status_edit_2509_telestyle_fused_dir")
        layout.addWidget(status_fused)
        self._validate_dir("edit_2509_telestyle_fused_dir", getattr(self._model_paths, "edit_2509_telestyle_fused_dir", ""))

        # SeedVR2 Model directory
        layout.addWidget(QLabel(""))
        hint_seedvr2 = QLabel("SeedVR2 Model directory (GGUF + VAE):")
        hint_seedvr2.setToolTip(
            "Directory containing SeedVR2 model files (seedvr2_ema_7b-Q8_0.gguf, ema_vae.pth)."
        )
        layout.addWidget(hint_seedvr2)
        row_seedvr2 = QHBoxLayout()
        edit_dir_seedvr2 = QLineEdit()
        edit_dir_seedvr2.setText(getattr(self._model_paths, "seedvr2_model_dir", ""))
        edit_dir_seedvr2.setPlaceholderText("Directory with GGUF + VAE files")
        edit_dir_seedvr2.textChanged.connect(lambda text: self._validate_dir("seedvr2_model_dir", text))
        row_seedvr2.addWidget(edit_dir_seedvr2, 1)
        self._fields["seedvr2_model_dir"] = edit_dir_seedvr2

        browse_btn_seedvr2 = QPushButton("...")
        browse_btn_seedvr2.setMaximumWidth(40)
        browse_btn_seedvr2.clicked.connect(lambda checked=False, e=edit_dir_seedvr2: self._browse_dir(e))
        row_seedvr2.addWidget(browse_btn_seedvr2)
        layout.addLayout(row_seedvr2)

        status_seedvr2 = QLabel()
        status_seedvr2.setProperty("class", "muted")
        status_seedvr2.setObjectName("status_seedvr2_model_dir")
        layout.addWidget(status_seedvr2)
        self._validate_dir("seedvr2_model_dir", getattr(self._model_paths, "seedvr2_model_dir", ""))

        # Wan Diffusers Base Directory
        layout.addWidget(QLabel(""))
        hint_wan = QLabel("Wan Diffusers Base Directory (Encoders/VAE):")
        hint_wan.setToolTip("Directory containing Wan 2.2 Diffusers components (text_encoder, vae, etc.)")
        layout.addWidget(hint_wan)
        row_wan = QHBoxLayout()
        edit_dir_wan = QLineEdit()
        edit_dir_wan.setText(getattr(self._model_paths, "wan_base_model_dir", ""))
        edit_dir_wan.setPlaceholderText("Directory with Wan components")
        edit_dir_wan.textChanged.connect(lambda text: self._validate_dir("wan_base_model_dir", text))
        row_wan.addWidget(edit_dir_wan, 1)
        self._fields["wan_base_model_dir"] = edit_dir_wan

        browse_btn_wan = QPushButton("...")
        browse_btn_wan.setMaximumWidth(40)
        browse_btn_wan.clicked.connect(lambda checked=False, e=edit_dir_wan: self._browse_dir(e))
        row_wan.addWidget(browse_btn_wan)
        layout.addLayout(row_wan)

        status_wan = QLabel()
        status_wan.setProperty("class", "muted")
        status_wan.setObjectName("status_wan_base_model_dir")
        layout.addWidget(status_wan)
        self._validate_dir("wan_base_model_dir", getattr(self._model_paths, "wan_base_model_dir", ""))

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

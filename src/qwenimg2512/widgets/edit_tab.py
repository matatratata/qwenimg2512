"""Edit tab for Qwen-Image-Edit-2511 mode."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.widgets.generation_controls import GenerationControlsWidget
from qwenimg2512.widgets.image_settings import ImageSettingsWidget
from qwenimg2512.widgets.lora_settings import LoraSettingsWidget
from qwenimg2512.widgets.prompt_input import PromptInputWidget


class ReferenceImageWidget(QGroupBox):
    image_changed = Signal(str)  # image path

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._image_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.thumbnail_label = QLabel("Drop or Browse")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(100)
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 4px;"
        )
        layout.addWidget(self.thumbnail_label)

        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_image)
        btn_row.addWidget(self.browse_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

    def _browse_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Reference Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if path:
            self.set_image(path)

    def set_image(self, path: str) -> None:
        if not Path(path).is_file():
            return
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setStyleSheet("")
        self.clear_btn.setEnabled(True)
        self.image_changed.emit(path)

    def clear_image(self) -> None:
        self._image_path = ""
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("Drop or Browse")
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 4px;"
        )
        self.clear_btn.setEnabled(False)
        self.image_changed.emit("")

    def get_image_path(self) -> str:
        return self._image_path

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                self.set_image(path)
                return


class EditTabWidget(QWidget):
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

        # 1. Prompt Input
        self.prompt_widget = PromptInputWidget()
        content_layout.addWidget(self.prompt_widget)

        # 2. Reference Images (3 slots)
        ref_group = QGroupBox("Reference Images (1-3 optional)")
        ref_layout = QHBoxLayout(ref_group)
        self.ref_widgets: list[ReferenceImageWidget] = []
        for i in range(3):
            w = ReferenceImageWidget(f"Ref {i+1}")
            ref_layout.addWidget(w)
            self.ref_widgets.append(w)
        content_layout.addWidget(ref_group)

        # 3. Settings (Resolution, Steps, etc.)
        self.settings_widget = ImageSettingsWidget()
        # Edit model doesn't support variant switching in the same way, or maybe it does? 
        # For now, let's keep the widget but maybe hide the model variant if it's fixed, 
        # or adapt it if we have quantization variants for Edit too.
        # The prompt says we have GGUF model, so we might want to hide the "Model" dropdown 
        # or repurposed it. For simplicity, we'll keep it as is, but maybe the user can 
        # implement variant selection later if needed.
        # NOTE: Model dropdown in ImageSettingsWidget is hardcoded to 2512 variants.
        # We might want to customize this or just ignore it for Edit tab.
        self.settings_widget.model_combo.setVisible(False)
        self.settings_widget.model_label.setVisible(False)
        # Actually better to just hide the whole row 0 if possible, but let's just leave it for now
        # and ignore the value.
        content_layout.addWidget(self.settings_widget)

        # 4. LoRA
        self.lora_widget = LoraSettingsWidget()
        content_layout.addWidget(self.lora_widget)

        # 5. Generation Controls
        self.gen_controls = GenerationControlsWidget()
        self.gen_controls.generate_clicked.connect(self.generate_requested)
        self.gen_controls.cancel_clicked.connect(self.cancel_requested)
        content_layout.addWidget(self.gen_controls)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def get_reference_images(self) -> list[str]:
        return [w.get_image_path() for w in self.ref_widgets]

    def set_reference_images(self, paths: list[str]) -> None:
        for i, path in enumerate(paths):
            if i < len(self.ref_widgets):
                self.ref_widgets[i].set_image(path)

    def set_generating(self, generating: bool) -> None:
        self.gen_controls.set_generating(generating)
        self.prompt_widget.setEnabled(not generating)
        for w in self.ref_widgets:
            w.setEnabled(not generating)
        self.settings_widget.setEnabled(not generating)
        self.lora_widget.setEnabled(not generating)

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

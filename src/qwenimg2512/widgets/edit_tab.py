"""Edit tab for Qwen-Image-Edit-2511 mode."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.config import ASPECT_RATIOS
from qwenimg2512.resize_utils import resize_with_fit_mode
from qwenimg2512.widgets.generation_controls import GenerationControlsWidget
from qwenimg2512.widgets.image_settings import ImageSettingsWidget
from qwenimg2512.widgets.lora_settings import LoraSettingsWidget
from qwenimg2512.widgets.prompt_input import PromptInputWidget

logger = logging.getLogger(__name__)


def _pil_to_qpixmap(pil_img) -> QPixmap:
    """Convert a PIL Image to a QPixmap."""
    from PIL import Image as PILImage

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height, 3 * pil_img.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class ReferenceImageWidget(QGroupBox):
    image_changed = Signal(str)  # image path

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self._image_path = ""
        self._target_w = 0
        self._target_h = 0
        self._resized_pil = None  # cached resized PIL image for save
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Original thumbnail
        self.thumbnail_label = QLabel("Drop or Browse")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(100)
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 4px;"
        )
        layout.addWidget(self.thumbnail_label)

        # Resized preview thumbnail
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(80)
        self.preview_label.setStyleSheet(
            "border: 1px solid #2a6a3a; border-radius: 6px; padding: 2px;"
        )
        self.preview_label.setVisible(False)
        layout.addWidget(self.preview_label)

        # Fit mode combo
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self.fit_combo = QComboBox()
        self.fit_combo.addItem("Cover (Crop)", "cover")
        self.fit_combo.addItem("Contain (Pad)", "contain")
        self.fit_combo.addItem("Stretch", "stretch")
        self.fit_combo.addItem("Center", "center")
        self.fit_combo.setToolTip(
            "How to resize to target resolution:\n"
            "• Cover: Scale + crop to fill\n"
            "• Contain: Scale + pad with black\n"
            "• Stretch: Distort to fill\n"
            "• Center: No scale, crop/pad"
        )
        self.fit_combo.currentIndexChanged.connect(self._update_preview)
        fit_row.addWidget(self.fit_combo, 1)
        layout.addLayout(fit_row)

        # Buttons row
        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_image)
        btn_row.addWidget(self.browse_btn)

        self.save_btn = QPushButton("Save Resized")
        self.save_btn.setToolTip("Save the resized image to disk (next to the original)")
        self.save_btn.clicked.connect(self._save_resized)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_image)
        self.clear_btn.setEnabled(False)
        btn_row.addWidget(self.clear_btn)

        layout.addLayout(btn_row)

        # Resolution info label
        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #6a6a80; font-size: 11px;")
        self.info_label.setVisible(False)
        layout.addWidget(self.info_label)

    def set_target_resolution(self, width: int, height: int) -> None:
        """Set the target output resolution for preview."""
        if self._target_w == width and self._target_h == height:
            return
        self._target_w = width
        self._target_h = height
        self._update_preview()

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
        if not path or not Path(path).is_file():
            return
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(100, Qt.TransformationMode.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setStyleSheet("")
        self.clear_btn.setEnabled(True)
        self._update_preview()
        self.image_changed.emit(path)

    def clear_image(self) -> None:
        self._image_path = ""
        self._resized_pil = None
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("Drop or Browse")
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 4px;"
        )
        self.preview_label.clear()
        self.preview_label.setVisible(False)
        self.info_label.setVisible(False)
        self.clear_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.image_changed.emit("")

    def get_image_path(self) -> str:
        return self._image_path

    def get_fit_mode(self) -> str:
        return self.fit_combo.currentData() or "cover"

    def set_fit_mode(self, mode: str) -> None:
        idx = self.fit_combo.findData(mode)
        if idx >= 0:
            self.fit_combo.setCurrentIndex(idx)

    def _update_preview(self) -> None:
        """Recompute and show the resized preview thumbnail."""
        if not self._image_path or not Path(self._image_path).is_file():
            self.preview_label.setVisible(False)
            self.info_label.setVisible(False)
            self.save_btn.setEnabled(False)
            self._resized_pil = None
            return

        if self._target_w <= 0 or self._target_h <= 0:
            self.preview_label.setVisible(False)
            self.info_label.setVisible(False)
            self.save_btn.setEnabled(False)
            self._resized_pil = None
            return

        try:
            from PIL import Image as PILImage

            img = PILImage.open(self._image_path).convert("RGB")
            mode = self.get_fit_mode()
            resized = resize_with_fit_mode(img, self._target_w, self._target_h, mode)
            self._resized_pil = resized

            # Show preview thumbnail (scaled down for display)
            pixmap = _pil_to_qpixmap(resized)
            scaled = pixmap.scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            self.preview_label.setPixmap(scaled)
            self.preview_label.setVisible(True)
            self.preview_label.setToolTip(
                f"Preview: {resized.width}×{resized.height} ({mode})"
            )
            self.info_label.setText(
                f"{img.width}×{img.height} → {resized.width}×{resized.height} ({mode})"
            )
            self.info_label.setVisible(True)
            self.save_btn.setEnabled(True)
        except Exception as exc:
            logger.warning("Failed to generate resize preview: %s", exc)
            self.preview_label.setVisible(False)
            self.info_label.setVisible(False)
            self.save_btn.setEnabled(False)
            self._resized_pil = None

    def _save_resized(self) -> None:
        """Save the current resized image to disk."""
        if self._resized_pil is None:
            return
        src = Path(self._image_path)
        out_path = src.parent / f"{src.stem}_resized{src.suffix}"
        counter = 1
        while out_path.exists():
            out_path = src.parent / f"{src.stem}_resized_{counter}{src.suffix}"
            counter += 1
        try:
            self._resized_pil.save(str(out_path))
            QMessageBox.information(self, "Saved", f"Resized image saved to:\n{out_path}")
            logger.info("Saved resized image to %s", out_path)
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

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
        self.prompt_widget.positive_input.setPlaceholderText(
            "Describe the edit. For multiple images, reference them as 'Picture 1', 'Picture 2', etc.\n"
            "Example: 'Replace the character in Picture 1 with the person from Picture 2.'"
        )
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
        self.settings_widget.model_combo.setVisible(False)
        self.settings_widget.model_label.setVisible(False)
        content_layout.addWidget(self.settings_widget)

        # Connect ratio changes → update ref widget target resolution
        self.settings_widget.ratio_combo.currentTextChanged.connect(
            lambda _: self._push_target_resolution()
        )

        # 3b. TeleStyle toggle
        self.telestyle_check = QCheckBox("TeleStyle (fused model — no LoRAs needed, 4 steps)")
        self.telestyle_check.toggled.connect(self._on_telestyle_toggled)
        content_layout.addWidget(self.telestyle_check)

        # 4. LoRA
        self.lora_widget = LoraSettingsWidget()
        content_layout.addWidget(self.lora_widget)

        # 4b. LoRA 2
        self.lora_widget_2 = LoraSettingsWidget()
        self.lora_widget_2.setTitle("LoRA Adapter 2")
        content_layout.addWidget(self.lora_widget_2)

        # 5. Generation Controls
        self.gen_controls = GenerationControlsWidget()
        self.gen_controls.generate_clicked.connect(self.generate_requested)
        self.gen_controls.cancel_clicked.connect(self.cancel_requested)
        content_layout.addWidget(self.gen_controls)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Push initial resolution
        self._push_target_resolution()

    def _push_target_resolution(self) -> None:
        """Push the current target resolution to all ref widgets."""
        w, h = self.settings_widget.get_resolution()
        for ref in self.ref_widgets:
            ref.set_target_resolution(w, h)

    def get_reference_images(self) -> list[str]:
        return [w.get_image_path() for w in self.ref_widgets]

    def set_reference_images(self, paths: list[str]) -> None:
        for i, path in enumerate(paths):
            if i < len(self.ref_widgets):
                self.ref_widgets[i].set_image(path)

    def get_fit_modes(self) -> list[str]:
        return [w.get_fit_mode() for w in self.ref_widgets]

    def set_fit_modes(self, modes: list[str]) -> None:
        for i, mode in enumerate(modes):
            if i < len(self.ref_widgets):
                self.ref_widgets[i].set_fit_mode(mode)

    def set_generating(self, generating: bool) -> None:
        self.gen_controls.set_generating(generating)
        self.prompt_widget.setEnabled(not generating)
        for w in self.ref_widgets:
            w.setEnabled(not generating)
        self.settings_widget.setEnabled(not generating)
        self.lora_widget.setEnabled(not generating)
        self.lora_widget_2.setEnabled(not generating)

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

    def set_telestyle(self, enabled: bool) -> None:
        self.telestyle_check.setChecked(enabled)

    def get_telestyle(self) -> bool:
        return self.telestyle_check.isChecked()

    def _on_telestyle_toggled(self, enabled: bool) -> None:
        """When TeleStyle is on, hide LoRA widgets and set fast defaults."""
        self.lora_widget.setVisible(not enabled)
        self.lora_widget_2.setVisible(not enabled)
        if enabled:
            # Store previous values so unchecking can restore them
            self._prev_steps = self.settings_widget.steps_spin.value()
            self._prev_cfg = self.settings_widget.cfg_spin.value()
            self.settings_widget.steps_spin.setValue(4)
            self.settings_widget.cfg_spin.setValue(1.0)
        else:
            self.settings_widget.steps_spin.setValue(getattr(self, "_prev_steps", 40))
            self.settings_widget.cfg_spin.setValue(getattr(self, "_prev_cfg", 4.0))

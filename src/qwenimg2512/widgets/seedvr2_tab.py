"""SeedVR2 image upscaler tab widget."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.widgets.generation_controls import GenerationControlsWidget


RESOLUTION_OPTIONS = {
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "2160p (4K)": 2160,
}

COLOR_CORRECTION_OPTIONS = ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]


class SeedVR2InputWidget(QGroupBox):
    """Single input image widget with drag-and-drop support."""

    image_changed = Signal(str)

    def __init__(self, title: str = "Input Image") -> None:
        super().__init__(title)
        self._image_path = ""
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.thumbnail_label = QLabel("Drop or Browse an image to upscale")
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail_label.setMinimumHeight(120)
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 8px;"
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
            "Select Image to Upscale",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tiff)",
        )
        if path:
            self.set_image(path)

    def set_image(self, path: str) -> None:
        if not path or not Path(path).is_file():
            return
        self._image_path = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            scaled = pixmap.scaledToHeight(120, Qt.TransformationMode.SmoothTransformation)
            self.thumbnail_label.setPixmap(scaled)
            self.thumbnail_label.setStyleSheet("")
        self.clear_btn.setEnabled(True)
        self.image_changed.emit(path)

    def clear_image(self) -> None:
        self._image_path = ""
        self.thumbnail_label.clear()
        self.thumbnail_label.setText("Drop or Browse an image to upscale")
        self.thumbnail_label.setStyleSheet(
            "border: 1px dashed #4a4a6a; border-radius: 6px; padding: 8px;"
        )
        self.clear_btn.setEnabled(False)
        self.image_changed.emit("")

    def get_image_path(self) -> str:
        return self._image_path

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.toLocalFile().lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")
                ):
                    event.acceptProposedAction()
                    return

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff")):
                self.set_image(path)
                return


class SeedVR2TabWidget(QWidget):
    """Tab widget for SeedVR2 image upscaling."""

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

        # 1. Input image
        self.input_widget = SeedVR2InputWidget("Input Image")
        content_layout.addWidget(self.input_widget)

        # 1b. Depth Map (Optional)
        self.depth_map_widget = SeedVR2InputWidget("Depth Map (Optional)")
        self.depth_map_widget.setToolTip(
            "Upload a depth map to guide the upscaling blending.\n"
            "Lighter areas = more AI upscaling (SeedVR2 details).\n"
            "Darker areas = more original image (Lanczos upscale)."
        )
        content_layout.addWidget(self.depth_map_widget)

        # 2. Quality Controls
        quality_group = QGroupBox("Quality Controls")
        quality_layout = QVBoxLayout(quality_group)

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        for label in RESOLUTION_OPTIONS:
            self.resolution_combo.addItem(label)
        self.resolution_combo.setCurrentText("1080p")
        res_row.addWidget(self.resolution_combo, 1)
        quality_layout.addLayout(res_row)

        # Seed
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(-1, 2**31 - 1)
        self.seed_spin.setValue(42)
        self.seed_spin.setSpecialValueText("Random")
        seed_row.addWidget(self.seed_spin, 1)
        quality_layout.addLayout(seed_row)

        # Input Noise Scale (Grain)
        grain_row = QHBoxLayout()
        grain_row.addWidget(QLabel("Grain (Input Noise):"))
        self.input_noise_spin = QDoubleSpinBox()
        self.input_noise_spin.setRange(0.0, 1.0)
        self.input_noise_spin.setSingleStep(0.05)
        self.input_noise_spin.setValue(0.0)
        self.input_noise_spin.setToolTip(
            "Inject noise into the input image to extract more detail. "
            "Higher values = more detail but less fidelity. Try 0.1–0.3."
        )
        grain_row.addWidget(self.input_noise_spin, 1)
        quality_layout.addLayout(grain_row)

        # Latent Noise Scale
        latent_row = QHBoxLayout()
        latent_row.addWidget(QLabel("Latent Noise:"))
        self.latent_noise_spin = QDoubleSpinBox()
        self.latent_noise_spin.setRange(0.0, 1.0)
        self.latent_noise_spin.setSingleStep(0.05)
        self.latent_noise_spin.setValue(0.0)
        self.latent_noise_spin.setToolTip(
            "Noise in latent space. Softens details if needed. Usually keep at 0."
        )
        latent_row.addWidget(self.latent_noise_spin, 1)
        quality_layout.addLayout(latent_row)

        # Color Correction
        cc_row = QHBoxLayout()
        cc_row.addWidget(QLabel("Color Correction:"))
        self.color_correction_combo = QComboBox()
        self.color_correction_combo.addItems(COLOR_CORRECTION_OPTIONS)
        self.color_correction_combo.setCurrentText("lab")
        self.color_correction_combo.setToolTip(
            "'lab' is recommended (perceptual). Use 'none' to disable."
        )
        cc_row.addWidget(self.color_correction_combo, 1)
        quality_layout.addLayout(cc_row)

        content_layout.addWidget(quality_group)

        # 3. Memory Optimization
        mem_group = QGroupBox("Memory Optimization")
        mem_layout = QVBoxLayout(mem_group)

        # VAE Tiling
        self.vae_tiling_check = QCheckBox("Enable VAE Tiling")
        self.vae_tiling_check.setChecked(True)
        self.vae_tiling_check.setToolTip(
            "Process VAE in tiles to save VRAM. preventing OOM on high resolutions. "
            "Slightly slower but essential for <24GB VRAM cards."
        )
        mem_layout.addWidget(self.vae_tiling_check)

        # Blocks to Swap
        swap_row = QHBoxLayout()
        swap_row.addWidget(QLabel("Blocks to Swap:"))
        self.blocks_swap_spin = QSpinBox()
        self.blocks_swap_spin.setRange(0, 36)
        self.blocks_swap_spin.setValue(0)
        self.blocks_swap_spin.setToolTip(
            "Offload N transformer blocks to CPU to save VRAM. "
            "Higher = less VRAM, slower speed. Try 10-20 if OOM."
        )
        swap_row.addWidget(self.blocks_swap_spin, 1)
        mem_layout.addLayout(swap_row)

        content_layout.addWidget(mem_group)

        # 3. Generation Controls (output dir, generate/cancel)
        self.gen_controls = GenerationControlsWidget()
        self.gen_controls.generate_clicked.connect(self.generate_requested)
        self.gen_controls.cancel_clicked.connect(self.cancel_requested)
        content_layout.addWidget(self.gen_controls)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    # --- Public accessors ---

    def get_depth_map_path(self) -> str:
        return self.depth_map_widget.get_image_path()

    def get_resolution(self) -> int:
        return RESOLUTION_OPTIONS[self.resolution_combo.currentText()]

    def set_resolution(self, value: int) -> None:
        for label, res in RESOLUTION_OPTIONS.items():
            if res == value:
                self.resolution_combo.setCurrentText(label)
                return

    def set_generating(self, generating: bool) -> None:
        self.gen_controls.set_generating(generating)
        self.input_widget.setEnabled(not generating)
        self.depth_map_widget.setEnabled(not generating)

    def set_progress(self, current: int, total: int, message: str) -> None:
        self.gen_controls.set_progress(current, total, message)

    def set_stage(self, stage: str) -> None:
        self.gen_controls.set_stage(stage)

    def set_finished(self, output_path: str) -> None:
        self.gen_controls.set_finished(output_path)

    def set_error(self, error: str) -> None:
        self.gen_controls.set_error(error)

"""Main application window."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from qwenimg2512.captioning_worker import CaptioningWorker
from qwenimg2512.config import Config
from qwenimg2512.widgets.controlnet_settings import ControlNetSettingsWidget
from qwenimg2512.widgets.generation_controls import GenerationControlsWidget
from qwenimg2512.widgets.image_input import ImageInputWidget
from qwenimg2512.widgets.image_preview import ImagePreviewWidget
from qwenimg2512.widgets.image_settings import ImageSettingsWidget
from qwenimg2512.widgets.lora_settings import LoraSettingsWidget
from qwenimg2512.widgets.model_paths_dialog import ModelPathsDialog
from qwenimg2512.widgets.prompt_input import PromptInputWidget
from qwenimg2512.worker import (
    GenerationWorker, 
    load_image_with_alpha_fill, 
    resize_and_center_crop
)
from qwenimg2512.edit_worker import EditWorker
from qwenimg2512.edit_2509_worker import Edit2509Worker
from qwenimg2512.seedvr2_worker import SeedVR2Worker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._config = Config.load()
        self._worker: GenerationWorker | None = None
        self._captioning_worker: CaptioningWorker | None = None
        self._caption_context: str = "input"  # "input" or "controlnet"
        self._control_caption_type: str = ""
        self._setup_ui()
        self._setup_menu()
        self._connect_fit_preview()
        self._load_settings()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Qwen-Image-2512")
        self.setMinimumSize(1600, 900)
        self.resize(1920, 1080)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Left panel: Tabs
        self.tabs = QTabWidget()
        
        # --- Tab 1: Generate (2512) ---
        generate_tab = QWidget()
        gen_layout = QVBoxLayout(generate_tab)
        gen_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)

        self.prompt_widget = PromptInputWidget()
        scroll_layout.addWidget(self.prompt_widget)

        self.image_input = ImageInputWidget()
        self.image_input.image_loaded.connect(self._on_image_loaded)
        self.image_input.image_cleared.connect(self._on_image_cleared)
        self.image_input.caption_requested.connect(self._start_captioning)
        scroll_layout.addWidget(self.image_input)

        self.settings_widget = ImageSettingsWidget()
        scroll_layout.addWidget(self.settings_widget)

        # Wire effective steps update for img2img
        self.image_input.strength_changed.connect(lambda _: self._update_effective_steps())
        self.image_input.image_loaded.connect(lambda _: self._update_effective_steps())
        self.image_input.image_cleared.connect(self._update_effective_steps)
        self.settings_widget.steps_spin.valueChanged.connect(lambda _: self._update_effective_steps())

        self.lora_widget = LoraSettingsWidget()
        scroll_layout.addWidget(self.lora_widget)

        self.controlnet_widget = ControlNetSettingsWidget()
        self.controlnet_widget.enable_check.toggled.connect(self._on_controlnet_toggled)
        self.controlnet_widget.caption_requested.connect(self._start_controlnet_captioning)
        scroll_layout.addWidget(self.controlnet_widget)

        self.gen_controls = GenerationControlsWidget()
        self.gen_controls.generate_clicked.connect(self._start_generation)
        self.gen_controls.cancel_clicked.connect(self._cancel_generation)
        scroll_layout.addWidget(self.gen_controls)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        gen_layout.addWidget(scroll)

        self.tabs.addTab(generate_tab, "Generate (2512)")

        # --- Tab 2: Edit (2511) ---
        from qwenimg2512.widgets.edit_tab import EditTabWidget
        self.edit_tab = EditTabWidget()
        self.edit_tab.generate_requested.connect(self._start_edit_generation)
        self.edit_tab.cancel_requested.connect(self._cancel_edit_generation)
        self.tabs.addTab(self.edit_tab, "Edit (2511)")

        # --- Tab 3: Edit (2509) ---
        self.edit_2509_tab = EditTabWidget()
        self.edit_2509_tab.generate_requested.connect(self._start_edit_2509_generation)
        self.edit_2509_tab.cancel_requested.connect(self._cancel_edit_2509_generation)
        self.tabs.addTab(self.edit_2509_tab, "Edit (2509)")

        # --- Tab 4: SeedVR2 ---
        from qwenimg2512.widgets.seedvr2_tab import SeedVR2TabWidget
        self.seedvr2_tab = SeedVR2TabWidget()
        self.seedvr2_tab.generate_requested.connect(self._start_seedvr2_generation)
        self.seedvr2_tab.cancel_requested.connect(self._cancel_seedvr2_generation)
        self.tabs.addTab(self.seedvr2_tab, "SeedVR2")

        # Right panel: preview
        self.preview_widget = ImagePreviewWidget()

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.preview_widget)
        splitter.setSizes([900, 1020])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter)

        self.statusBar().showMessage("Ready")

    def _setup_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("&File")
        file_menu.addAction("&Model Paths...", self._open_model_paths)
        file_menu.addSeparator()
        file_menu.addAction("&Open Output Folder", self._open_output_folder)
        file_menu.addSeparator()
        file_menu.addAction("&Save Settings", self._save_settings)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)



    def _load_settings(self) -> None:
        # Generate settings
        gs = self._config.generation
        self.prompt_widget.set_prompt(gs.prompt)
        self.prompt_widget.set_negative_prompt(gs.negative_prompt)
        self.gen_controls.seed_spin.setValue(gs.seed)
        self.gen_controls.output_edit.setText(gs.output_dir)
        self.settings_widget.steps_spin.setValue(gs.num_inference_steps)
        self.settings_widget.cfg_spin.setValue(gs.true_cfg_scale)
        self.settings_widget.guidance_spin.setValue(gs.guidance_scale)

        # Set combo boxes by text
        idx = self.settings_widget.ratio_combo.findText(gs.aspect_ratio)
        if idx >= 0:
            self.settings_widget.ratio_combo.setCurrentIndex(idx)
        idx = self.settings_widget.model_combo.findText(gs.model_variant)
        if idx >= 0:
            self.settings_widget.model_combo.setCurrentIndex(idx)
        self.settings_widget.sampler_combo.setCurrentText(gs.sampler_name)

        # img2img settings
        self.image_input.set_strength(gs.img2img_strength)
        if gs.input_image_path and Path(gs.input_image_path).is_file():
            self.image_input.set_image(gs.input_image_path)
        self.image_input.set_alpha_fill(gs.alpha_fill)

        # LoRA settings
        self.lora_widget.set_lora_path(gs.lora_path)
        self.lora_widget.set_scale_start(gs.lora_scale_start)
        self.lora_widget.set_scale_end(gs.lora_scale_end)
        self.lora_widget.set_step_start(gs.lora_step_start)
        self.lora_widget.set_step_end(gs.lora_step_end)

        # ControlNet settings
        self.controlnet_widget.set_enabled(gs.controlnet_enabled)
        self.controlnet_widget.set_control_type(gs.control_type)
        self.controlnet_widget.set_control_image_path(gs.control_image_path)
        self.controlnet_widget.set_conditioning_scale(gs.controlnet_conditioning_scale)
        self.controlnet_widget.set_guidance_start(gs.control_guidance_start)
        self.controlnet_widget.set_guidance_end(gs.control_guidance_end)

        # Edit settings
        es = self._config.edit
        self.edit_tab.prompt_widget.set_prompt(es.prompt)
        self.edit_tab.prompt_widget.set_negative_prompt(es.negative_prompt)
        self.edit_tab.gen_controls.seed_spin.setValue(es.seed)
        self.edit_tab.gen_controls.output_edit.setText(es.output_dir)
        self.edit_tab.settings_widget.steps_spin.setValue(es.num_inference_steps)
        self.edit_tab.settings_widget.cfg_spin.setValue(es.true_cfg_scale)
        self.edit_tab.settings_widget.guidance_spin.setValue(es.guidance_scale)
        
        idx = self.edit_tab.settings_widget.ratio_combo.findText(es.aspect_ratio)
        if idx >= 0:
            self.edit_tab.settings_widget.ratio_combo.setCurrentIndex(idx)
        self.edit_tab.settings_widget.sampler_combo.setCurrentText(es.sampler_name)

        self.edit_tab.set_reference_images([es.ref_image_1, es.ref_image_2, es.ref_image_3])
        self.edit_tab.set_fit_modes([es.ref_fit_mode_1, es.ref_fit_mode_2, es.ref_fit_mode_3])

        self.edit_tab.lora_widget.set_lora_path(es.lora_path)
        self.edit_tab.lora_widget.set_scale_start(es.lora_scale_start)
        self.edit_tab.lora_widget.set_scale_end(es.lora_scale_end)
        self.edit_tab.lora_widget.set_step_start(es.lora_step_start)
        self.edit_tab.lora_widget.set_step_start(es.lora_step_start)
        self.edit_tab.lora_widget.set_step_end(es.lora_step_end)

        # Edit 2509 settings
        es2 = self._config.edit_2509
        self.edit_2509_tab.prompt_widget.set_prompt(es2.prompt)
        self.edit_2509_tab.prompt_widget.set_negative_prompt(es2.negative_prompt)
        self.edit_2509_tab.gen_controls.seed_spin.setValue(es2.seed)
        self.edit_2509_tab.gen_controls.output_edit.setText(es2.output_dir)
        self.edit_2509_tab.settings_widget.steps_spin.setValue(es2.num_inference_steps)
        self.edit_2509_tab.settings_widget.cfg_spin.setValue(es2.true_cfg_scale)
        self.edit_2509_tab.settings_widget.guidance_spin.setValue(es2.guidance_scale)

        idx = self.edit_2509_tab.settings_widget.ratio_combo.findText(es2.aspect_ratio)
        if idx >= 0:
            self.edit_2509_tab.settings_widget.ratio_combo.setCurrentIndex(idx)
        self.edit_2509_tab.settings_widget.sampler_combo.setCurrentText(es2.sampler_name)

        self.edit_2509_tab.set_reference_images([es2.ref_image_1, es2.ref_image_2, es2.ref_image_3])
        self.edit_2509_tab.set_fit_modes([es2.ref_fit_mode_1, es2.ref_fit_mode_2, es2.ref_fit_mode_3])

        self.edit_2509_tab.set_telestyle(es2.use_telestyle)

        self.edit_2509_tab.lora_widget.set_lora_path(es2.lora_path)
        self.edit_2509_tab.lora_widget.set_scale_start(es2.lora_scale_start)
        self.edit_2509_tab.lora_widget.set_scale_end(es2.lora_scale_end)
        self.edit_2509_tab.lora_widget.set_step_start(es2.lora_step_start)
        self.edit_2509_tab.lora_widget.set_step_end(es2.lora_step_end)

        self.edit_2509_tab.lora_widget_2.set_lora_path(es2.lora_path_2)
        self.edit_2509_tab.lora_widget_2.set_scale_start(es2.lora_scale_start_2)
        self.edit_2509_tab.lora_widget_2.set_scale_end(es2.lora_scale_end_2)
        self.edit_2509_tab.lora_widget_2.set_step_start(es2.lora_step_start_2)
        self.edit_2509_tab.lora_widget_2.set_step_end(es2.lora_step_end_2)

        # SeedVR2 settings
        sv = self._config.seedvr2
        self.seedvr2_tab.input_widget.set_image(self._config.seedvr2.input_image)
        self.seedvr2_tab.depth_map_widget.set_image(self._config.seedvr2.depth_map_path)
        self.seedvr2_tab.gen_controls.set_output_dir(self._config.seedvr2.output_dir)
        self.seedvr2_tab.set_resolution(self._config.seedvr2.resolution)
        self.seedvr2_tab.seed_spin.setValue(sv.seed)
        self.seedvr2_tab.input_noise_spin.setValue(sv.input_noise_scale)
        self.seedvr2_tab.latent_noise_spin.setValue(sv.latent_noise_scale)
        idx = self.seedvr2_tab.color_correction_combo.findText(sv.color_correction)
        if idx >= 0:
            self.seedvr2_tab.color_correction_combo.setCurrentIndex(idx)
        self.seedvr2_tab.vae_tiling_check.setChecked(sv.vae_tiling)
        self.seedvr2_tab.blocks_swap_spin.setValue(sv.blocks_to_swap)

    def _collect_settings(self) -> None:
        gs = self._config.generation
        gs.prompt = self.prompt_widget.get_prompt()
        gs.negative_prompt = self.prompt_widget.get_negative_prompt()
        gs.aspect_ratio = self.settings_widget.get_aspect_ratio()
        gs.sampler_name = self.settings_widget.get_sampler_name()
        gs.num_inference_steps = self.settings_widget.get_steps()
        gs.true_cfg_scale = self.settings_widget.get_cfg_scale()
        gs.guidance_scale = self.settings_widget.get_guidance_scale()
        gs.seed = self.gen_controls.get_seed()
        gs.model_variant = self.settings_widget.get_model_variant()
        gs.output_dir = self.gen_controls.get_output_dir()
        gs.input_image_path = self.image_input.get_image_path()
        gs.img2img_strength = self.image_input.get_strength()
        gs.alpha_fill = self.image_input.get_alpha_fill()
        gs.lora_path = self.lora_widget.get_lora_path()
        gs.lora_scale_start = self.lora_widget.get_scale_start()
        gs.lora_scale_end = self.lora_widget.get_scale_end()
        gs.lora_step_start = self.lora_widget.get_step_start()
        gs.lora_step_end = self.lora_widget.get_step_end()
        gs.controlnet_enabled = self.controlnet_widget.is_enabled()
        gs.control_type = self.controlnet_widget.get_control_type()
        gs.control_image_path = self.controlnet_widget.get_control_image_path()
        gs.controlnet_conditioning_scale = self.controlnet_widget.get_conditioning_scale()
        gs.control_guidance_start = self.controlnet_widget.get_guidance_start()
        gs.control_guidance_end = self.controlnet_widget.get_guidance_end()

        # Edit settings
        es = self._config.edit
        es.prompt = self.edit_tab.prompt_widget.get_prompt()
        es.negative_prompt = self.edit_tab.prompt_widget.get_negative_prompt()
        es.aspect_ratio = self.edit_tab.settings_widget.get_aspect_ratio()
        es.sampler_name = self.edit_tab.settings_widget.get_sampler_name()
        es.num_inference_steps = self.edit_tab.settings_widget.get_steps()
        es.true_cfg_scale = self.edit_tab.settings_widget.get_cfg_scale()
        es.guidance_scale = self.edit_tab.settings_widget.get_guidance_scale()
        es.seed = self.edit_tab.gen_controls.get_seed()
        es.output_dir = self.edit_tab.gen_controls.get_output_dir()
        
        refs = self.edit_tab.get_reference_images()
        es.ref_image_1 = refs[0] if len(refs) > 0 else ""
        es.ref_image_2 = refs[1] if len(refs) > 1 else ""
        es.ref_image_3 = refs[2] if len(refs) > 2 else ""

        es.lora_path = self.edit_tab.lora_widget.get_lora_path()

        fit_modes = self.edit_tab.get_fit_modes()
        es.ref_fit_mode_1 = fit_modes[0] if len(fit_modes) > 0 else "cover"
        es.ref_fit_mode_2 = fit_modes[1] if len(fit_modes) > 1 else "cover"
        es.ref_fit_mode_3 = fit_modes[2] if len(fit_modes) > 2 else "cover"
        es.lora_scale_start = self.edit_tab.lora_widget.get_scale_start()
        es.lora_scale_end = self.edit_tab.lora_widget.get_scale_end()
        es.lora_step_start = self.edit_tab.lora_widget.get_step_start()
        es.lora_step_end = self.edit_tab.lora_widget.get_step_end()

        # Edit 2509 settings
        es2 = self._config.edit_2509
        es2.prompt = self.edit_2509_tab.prompt_widget.get_prompt()
        es2.negative_prompt = self.edit_2509_tab.prompt_widget.get_negative_prompt()
        es2.aspect_ratio = self.edit_2509_tab.settings_widget.get_aspect_ratio()
        es2.sampler_name = self.edit_2509_tab.settings_widget.get_sampler_name()
        es2.num_inference_steps = self.edit_2509_tab.settings_widget.get_steps()
        es2.true_cfg_scale = self.edit_2509_tab.settings_widget.get_cfg_scale()
        es2.guidance_scale = self.edit_2509_tab.settings_widget.get_guidance_scale()
        es2.seed = self.edit_2509_tab.gen_controls.get_seed()
        es2.output_dir = self.edit_2509_tab.gen_controls.get_output_dir()

        refs2 = self.edit_2509_tab.get_reference_images()
        es2.ref_image_1 = refs2[0] if len(refs2) > 0 else ""
        es2.ref_image_2 = refs2[1] if len(refs2) > 1 else ""
        es2.ref_image_3 = refs2[2] if len(refs2) > 2 else ""

        es2.use_telestyle = self.edit_2509_tab.get_telestyle()

        es2.lora_path = self.edit_2509_tab.lora_widget.get_lora_path()

        fit_modes2 = self.edit_2509_tab.get_fit_modes()
        es2.ref_fit_mode_1 = fit_modes2[0] if len(fit_modes2) > 0 else "cover"
        es2.ref_fit_mode_2 = fit_modes2[1] if len(fit_modes2) > 1 else "cover"
        es2.ref_fit_mode_3 = fit_modes2[2] if len(fit_modes2) > 2 else "cover"
        es2.lora_scale_start = self.edit_2509_tab.lora_widget.get_scale_start()
        es2.lora_scale_end = self.edit_2509_tab.lora_widget.get_scale_end()
        es2.lora_step_start = self.edit_2509_tab.lora_widget.get_step_start()
        es2.lora_step_end = self.edit_2509_tab.lora_widget.get_step_end()

        es2.lora_path_2 = self.edit_2509_tab.lora_widget_2.get_lora_path()
        es2.lora_scale_start_2 = self.edit_2509_tab.lora_widget_2.get_scale_start()
        es2.lora_scale_end_2 = self.edit_2509_tab.lora_widget_2.get_scale_end()
        es2.lora_step_start_2 = self.edit_2509_tab.lora_widget_2.get_step_start()
        es2.lora_step_end_2 = self.edit_2509_tab.lora_widget_2.get_step_end()

        # SeedVR2 settings
        sv = self._config.seedvr2
        sv.input_image = self.seedvr2_tab.input_widget.get_image_path()
        sv.depth_map_path = self.seedvr2_tab.get_depth_map_path()
        sv.output_dir = self.seedvr2_tab.gen_controls.get_output_dir()
        sv.resolution = self.seedvr2_tab.get_resolution()
        sv.seed = self.seedvr2_tab.seed_spin.value()
        sv.input_noise_scale = self.seedvr2_tab.input_noise_spin.value()
        sv.latent_noise_scale = self.seedvr2_tab.latent_noise_spin.value()
        sv.color_correction = self.seedvr2_tab.color_correction_combo.currentText()
        sv.vae_tiling = self.seedvr2_tab.vae_tiling_check.isChecked()
        sv.blocks_to_swap = self.seedvr2_tab.blocks_swap_spin.value()

    def _save_settings(self) -> None:
        self._collect_settings()
        self._config.save()
        self.statusBar().showMessage("Settings saved")

    def _is_busy(self) -> bool:
        gen_running = self._worker and self._worker.isRunning()
        try:
            cap_running = self._captioning_worker and self._captioning_worker.isRunning()
        except RuntimeError:
            self._captioning_worker = None
            cap_running = False
        return bool(gen_running or cap_running)

    def _start_edit_generation(self) -> None:
        prompt = self.edit_tab.prompt_widget.get_prompt()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt before generating.")
            return

        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._collect_settings()
        self._config.save()

        self.edit_tab.set_generating(True)
        self.statusBar().showMessage("Generating (Edit)...")

        self._worker = EditWorker(self._config.edit, self._config.model_paths)
        self._worker.progress_updated.connect(self._on_edit_progress)
        self._worker.stage_changed.connect(self._on_edit_stage)
        self._worker.finished_success.connect(self._on_edit_finished)
        self._worker.error_occurred.connect(self._on_edit_error)
        self._worker.vram_updated.connect(self._on_edit_vram)
        self._worker.start()

    def _cancel_edit_generation(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.edit_tab.set_generating(False)
            self.edit_tab.set_stage("Cancelled")
            self.statusBar().showMessage("Generation cancelled")

    def _on_edit_progress(self, current: int, total: int, message: str) -> None:
        self.edit_tab.set_progress(current, total, message)

    def _on_edit_stage(self, stage: str) -> None:
        self.edit_tab.set_stage(stage)
        self.statusBar().showMessage(stage)

    def _on_edit_finished(self, output_path: str) -> None:
        self.edit_tab.set_generating(False)
        self.edit_tab.set_finished(output_path)
        self.preview_widget.set_image(output_path)
        self.statusBar().showMessage(f"Image saved: {output_path}")

    def _on_edit_error(self, error: str) -> None:
        self.edit_tab.set_generating(False)
        self.edit_tab.set_error(error)
        self.statusBar().showMessage(f"Error: {error}")
        QMessageBox.critical(self, "Generation Error", error)

    def _on_edit_vram(self, gb: float) -> None:
        self.edit_tab.set_vram(gb)

    # --- Edit (2509) ---

    def _start_edit_2509_generation(self) -> None:
        prompt = self.edit_2509_tab.prompt_widget.get_prompt()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt before generating.")
            return

        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._collect_settings()
        self._config.save()

        self.edit_2509_tab.set_generating(True)
        self.statusBar().showMessage("Generating (Edit 2509)...")

        self._worker = Edit2509Worker(self._config.edit_2509, self._config.model_paths)
        self._worker.progress_updated.connect(self._on_edit_2509_progress)
        self._worker.stage_changed.connect(self._on_edit_2509_stage)
        self._worker.finished_success.connect(self._on_edit_2509_finished)
        self._worker.error_occurred.connect(self._on_edit_2509_error)
        self._worker.vram_updated.connect(self._on_edit_2509_vram)
        self._worker.start()

    def _cancel_edit_2509_generation(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.edit_2509_tab.set_generating(False)
            self.edit_2509_tab.set_stage("Cancelled")
            self.statusBar().showMessage("Generation cancelled")

    def _on_edit_2509_progress(self, current: int, total: int, message: str) -> None:
        self.edit_2509_tab.set_progress(current, total, message)

    def _on_edit_2509_stage(self, stage: str) -> None:
        self.edit_2509_tab.set_stage(stage)
        self.statusBar().showMessage(stage)

    def _on_edit_2509_finished(self, output_path: str) -> None:
        self.edit_2509_tab.set_generating(False)
        self.edit_2509_tab.set_finished(output_path)
        self.preview_widget.set_image(output_path)
        self.statusBar().showMessage(f"Image saved: {output_path}")

    def _on_edit_2509_error(self, error: str) -> None:
        self.edit_2509_tab.set_generating(False)
        self.edit_2509_tab.set_error(error)
        self.statusBar().showMessage(f"Error: {error}")
        QMessageBox.critical(self, "Generation Error", error)

    def _on_edit_2509_vram(self, gb: float) -> None:
        self.edit_2509_tab.set_vram(gb)

    # --- SeedVR2 ---

    def _start_seedvr2_generation(self) -> None:
        if not self.seedvr2_tab.input_widget.get_image_path():
            QMessageBox.warning(self, "Missing Input", "Please select an image to upscale.")
            return

        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._collect_settings()
        self._config.save()

        self.seedvr2_tab.set_generating(True)
        self.statusBar().showMessage("Upscaling (SeedVR2)...")

        self._worker = SeedVR2Worker(self._config.seedvr2, self._config.model_paths)
        self._worker.progress_updated.connect(self._on_seedvr2_progress)
        self._worker.stage_changed.connect(self._on_seedvr2_stage)
        self._worker.finished_success.connect(self._on_seedvr2_finished)
        self._worker.error_occurred.connect(self._on_seedvr2_error)
        self._worker.start()

    def _cancel_seedvr2_generation(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.seedvr2_tab.set_generating(False)
            self.seedvr2_tab.set_stage("Cancelled")
            self.statusBar().showMessage("Upscaling cancelled")

    def _on_seedvr2_progress(self, current: int, total: int, message: str) -> None:
        self.seedvr2_tab.set_progress(current, total, message)

    def _on_seedvr2_stage(self, stage: str) -> None:
        self.seedvr2_tab.set_stage(stage)
        self.statusBar().showMessage(stage)

    def _on_seedvr2_finished(self, output_path: str) -> None:
        self.seedvr2_tab.set_generating(False)
        self.seedvr2_tab.set_finished(output_path)
        self.preview_widget.set_image(output_path)
        self.statusBar().showMessage(f"Image saved: {output_path}")

    def _on_seedvr2_error(self, error: str) -> None:
        self.seedvr2_tab.set_generating(False)
        self.seedvr2_tab.set_error(error)
        self.statusBar().showMessage(f"Error: {error}")
        QMessageBox.critical(self, "SeedVR2 Error", error)

    def _start_generation(self) -> None:
        prompt = self.prompt_widget.get_prompt()
        if not prompt:
            QMessageBox.warning(self, "Missing Prompt", "Please enter a prompt before generating.")
            return

        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._collect_settings()
        self._config.save()


        self.gen_controls.set_generating(True)
        self.statusBar().showMessage("Generating...")

        self._worker = GenerationWorker(self._config.generation, self._config.model_paths)
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.stage_changed.connect(self._on_stage)
        self._worker.finished_success.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.vram_updated.connect(self._on_vram)
        self._worker.start()

    def _cancel_generation(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.gen_controls.set_generating(False)
            self.gen_controls.set_stage("Cancelled")
            self.statusBar().showMessage("Generation cancelled")

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.gen_controls.set_progress(current, total, message)

    def _on_stage(self, stage: str) -> None:
        self.gen_controls.set_stage(stage)
        self.statusBar().showMessage(stage)

    def _on_finished(self, output_path: str) -> None:
        self.gen_controls.set_generating(False)
        self.gen_controls.set_finished(output_path)
        self.preview_widget.set_image(output_path)
        self.statusBar().showMessage(f"Image saved: {output_path}")

    def _on_error(self, error: str) -> None:
        self.gen_controls.set_generating(False)
        self.gen_controls.set_error(error)
        self.statusBar().showMessage(f"Error: {error}")
        QMessageBox.critical(self, "Generation Error", error)

    def _on_vram(self, gb: float) -> None:
        self.gen_controls.set_vram(gb)

    # --- ControlNet ---

    def _on_controlnet_toggled(self, enabled: bool) -> None:
        if enabled:
            self.statusBar().showMessage("ControlNet enabled")
        else:
            self.statusBar().showMessage("ControlNet disabled")

    # --- Image input ---

    def _on_image_loaded(self, path: str) -> None:
        self.statusBar().showMessage(f"Input image: {Path(path).name}")

    def _on_image_cleared(self) -> None:
        self.statusBar().showMessage("Input image cleared (txt2img mode)")

    def _update_effective_steps(self) -> None:
        has_image = bool(self.image_input.get_image_path())
        strength = self.image_input.get_strength()
        self.settings_widget.update_effective_steps(has_image, strength)

    # --- Fit preview ---

    def _connect_fit_preview(self) -> None:
        self.settings_widget.ratio_combo.currentTextChanged.connect(lambda _: self._update_fit_preview())
        self.image_input.image_loaded.connect(lambda _: self._update_fit_preview())
        self.image_input.image_cleared.connect(self._update_fit_preview)
        self.controlnet_widget.settings_changed.connect(self._update_fit_preview)
        self.image_input.alpha_fill_changed.connect(lambda _: self._update_fit_preview())

    def _update_fit_preview(self) -> None:
        from PIL import Image

        target_w, target_h = self.settings_widget.get_resolution()
        images: list[tuple[Image.Image, str]] = []

        input_path = self.image_input.get_image_path()
        if input_path and Path(input_path).is_file():
            fill_mode = self.image_input.get_alpha_fill()
            img = load_image_with_alpha_fill(input_path, fill_mode)
            cropped = resize_and_center_crop(img, target_w, target_h)
            images.append((cropped, "Input"))

        if self.controlnet_widget.is_enabled():
            cn_path = self.controlnet_widget.get_control_image_path()
            if cn_path and Path(cn_path).is_file():
                # ControlNet also respects the alpha fill setting for now, 
                # or we could default to grey/black. Let's use the setting.
                fill_mode = self.image_input.get_alpha_fill()
                img = load_image_with_alpha_fill(cn_path, fill_mode)
                cropped = resize_and_center_crop(img, target_w, target_h)
                images.append((cropped, "Control"))

        if images:
            self.preview_widget.show_fit_preview(images, target_w, target_h)
        else:
            self.preview_widget.clear_fit_preview()

    # --- Captioning ---

    def _start_captioning(self, image_path: str) -> None:
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._caption_context = "input"
        self.image_input.set_captioning(True)
        self.statusBar().showMessage("Captioning...")

        self._captioning_worker = CaptioningWorker(image_path, self._config.model_paths)
        self._captioning_worker.caption_ready.connect(self._on_caption_ready)
        self._captioning_worker.stage_changed.connect(self._on_caption_stage)
        self._captioning_worker.error_occurred.connect(self._on_caption_error)
        self._captioning_worker.finished.connect(self._on_captioning_finished)
        self._captioning_worker.start()

    def _start_controlnet_captioning(self, image_path: str, control_type: str) -> None:
        if self._is_busy():
            QMessageBox.warning(self, "Busy", "Please wait for the current operation to finish.")
            return

        self._caption_context = "controlnet"
        self._control_caption_type = control_type
        self.controlnet_widget.set_captioning(True)
        self.statusBar().showMessage(f"Captioning {control_type}...")

        prompt = (
            f"Describe the {control_type} clearly. "
            "Focus on the main subject's pose, position, action. "
            "Do not describe colors, lighting, or style. "
            "Output only the description."
        )

        self._captioning_worker = CaptioningWorker(
            image_path,
            self._config.model_paths,
            custom_prompt=prompt
        )
        self._captioning_worker.caption_ready.connect(self._on_caption_ready)
        self._captioning_worker.stage_changed.connect(self._on_caption_stage)
        self._captioning_worker.error_occurred.connect(self._on_caption_error)
        self._captioning_worker.finished.connect(self._on_captioning_finished)
        self._captioning_worker.start()

    def _on_captioning_finished(self) -> None:
        """Clean up the captioning worker reference before Qt deletes the C++ object."""
        if self._captioning_worker:
            self._captioning_worker.deleteLater()
            self._captioning_worker = None

    def _on_caption_ready(self, caption: str) -> None:
        if self._caption_context == "controlnet":
            # Format: "{control_type}: {caption}"
            # Ensure control type is lower case for the prefix
            prefix = self._control_caption_type.lower()
            formatted = f"{prefix}: {caption}"
            self.controlnet_widget.set_captioning(False)
            self.prompt_widget.append_to_prompt(formatted)
            self.statusBar().showMessage(f"ControlNet caption appended")
        else:
            self.image_input.set_captioning(False)
            self.prompt_widget.set_prompt(caption)
            self.statusBar().showMessage("Caption generated")



    def _on_caption_stage(self, stage: str) -> None:
        self.statusBar().showMessage(stage)

    def _on_caption_error(self, error: str) -> None:
        if self._caption_context == "controlnet":
            self.controlnet_widget.set_captioning(False)
        else:
            self.image_input.set_captioning(False)

        self.statusBar().showMessage(f"Caption error: {error}")
        QMessageBox.critical(self, "Captioning Error", error)


    # --- Model paths ---

    def _open_model_paths(self) -> None:
        dialog = ModelPathsDialog(self._config.model_paths, self)
        if dialog.exec():
            self._config.model_paths = dialog.get_model_paths()
            self._config.save()
            self.statusBar().showMessage("Model paths updated")

    def _open_output_folder(self) -> None:
        output_dir = Path(self.gen_controls.get_output_dir())
        output_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(str(output_dir))  # noqa: S606
        elif sys.platform == "linux":
            subprocess.Popen(["xdg-open", str(output_dir)])  # noqa: S603, S607
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])  # noqa: S603, S607

    def closeEvent(self, event: object) -> None:
        if self._worker:
            try:
                if self._worker.isRunning():
                    self._worker.cancel()
                    self._worker.wait(5000)
            except RuntimeError:
                pass  # Worker already deleted

        if self._captioning_worker:
            try:
                if self._captioning_worker.isRunning():
                    self._captioning_worker.cancel()
                    self._captioning_worker.wait(5000)
            except RuntimeError:
                pass  # Worker already deleted
        self._collect_settings()
        self._config.save()
        super().closeEvent(event)

"""Background worker for SeedVR2 image upscaling via inference_cli.py subprocess."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ModelPaths, SeedVR2Settings

logger = logging.getLogger(__name__)


class SeedVR2Worker(QThread):
    """Runs SeedVR2 image upscaling in a background thread via subprocess."""

    progress_updated = Signal(int, int, str)
    stage_changed = Signal(str)
    finished_success = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, settings: SeedVR2Settings, model_paths: ModelPaths) -> None:
        super().__init__()
        self._settings = settings
        self._model_paths = model_paths
        self._process: subprocess.Popen | None = None
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True
        if self._process:
            try:
                self._process.terminate()
            except Exception:
                pass

    def run(self) -> None:
        try:
            self._run_upscale()
        except Exception as e:
            if self._is_cancelled:
                logger.info("SeedVR2 upscaling cancelled")
            else:
                logger.exception("SeedVR2 upscaling failed")
                self.error_occurred.emit(str(e))

    def _run_upscale(self) -> None:
        cli_path = Path(self._model_paths.seedvr2_cli)
        if not cli_path.is_file():
            raise FileNotFoundError(
                f"SeedVR2 inference_cli.py not found at: {cli_path}\n"
                "Clone the repo: git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
            )

        input_path = self._settings.input_image
        if not input_path or not Path(input_path).is_file():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Build output path
        output_dir = Path(self._settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_stem = Path(input_path).stem
        output_path = output_dir / f"{input_stem}_seedvr2_{self._settings.resolution}p.png"
        counter = 1
        while output_path.exists():
            output_path = output_dir / f"{input_stem}_seedvr2_{self._settings.resolution}p_v{counter:03d}.png"
            counter += 1

        self.stage_changed.emit("Starting SeedVR2 upscaling...")

        # Build command
        cmd = [
            sys.executable, str(cli_path),
            str(input_path),
            "--output", str(output_path),
            "--model_dir", self._model_paths.seedvr2_model_dir,
            "--dit_model", Path(self._model_paths.seedvr2_gguf).name,
            "--resolution", str(self._settings.resolution),
            "--seed", str(self._settings.seed),
            "--color_correction", self._settings.color_correction,
            "--input_noise_scale", str(self._settings.input_noise_scale),
            "--latent_noise_scale", str(self._settings.latent_noise_scale),
        ]

        if self._settings.vae_tiling:
            cmd.extend(["--vae_encode_tiled", "--vae_decode_tiled"])

        if self._settings.blocks_to_swap > 0:
            cmd.extend(["--blocks_to_swap", str(self._settings.blocks_to_swap)])
            # Block swapping requires offloading to work effectively
            cmd.extend(["--dit_offload_device", "cpu", "--vae_offload_device", "cpu"])

        logger.info("SeedVR2 command: %s", " ".join(cmd))
        self.stage_changed.emit("Running SeedVR2 model...")

        # Set up environment — ensure the CLI's parent dir is on PYTHONPATH
        env = os.environ.copy()
        cli_dir = str(cli_path.parent)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{cli_dir}:{existing}" if existing else cli_dir

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cli_path.parent),
            env=env,
        )

        # Monitor output — collect all lines for error reporting
        output_lines: list[str] = []
        progress_re = re.compile(r"(\d+)%|Processing|Encoding|Decoding|Saving|Loading|Upscal", re.IGNORECASE)
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            output_lines.append(line)
            logger.info("SeedVR2: %s", line)

            if self._is_cancelled:
                self._process.terminate()
                return

            # Try to extract progress percentage
            pct_match = re.search(r"(\d+)%", line)
            if pct_match:
                pct = int(pct_match.group(1))
                self.progress_updated.emit(pct, 100, f"{pct}% complete")

            # Update stage for key events
            if progress_re.search(line):
                self.stage_changed.emit(line[:80])

        self._process.wait()

        if self._is_cancelled:
            return

        if self._process.returncode != 0:
            # Show the last 20 lines of output for debugging
            tail = "\n".join(output_lines[-20:])
            raise RuntimeError(
                f"SeedVR2 process exited with code {self._process.returncode}\n\n"
                f"--- Last output ---\n{tail}"
            )

        if not output_path.exists():
            # The CLI may have generated the output with a different name; search
            candidates = list(output_dir.glob(f"{input_stem}_seedvr2*"))
            if candidates:
                output_path = max(candidates, key=lambda p: p.stat().st_mtime)
            else:
                raise FileNotFoundError(f"SeedVR2 output not found at: {output_path}")

        # Depth-guided blending (optional)
        if self._settings.depth_map_path and Path(self._settings.depth_map_path).is_file():
            self.stage_changed.emit("Blending with depth map...")
            try:
                self._apply_depth_blend(str(output_path), str(input_path), self._settings.depth_map_path)
            except Exception as e:
                logger.exception("Failed to apply depth blend")
                self.error_occurred.emit(f"Depth blend failed: {e}")
                return

        self.progress_updated.emit(100, 100, "Complete!")
        self.finished_success.emit(str(output_path))

    def _apply_depth_blend(self, seedvr2_path: str, input_path: str, depth_path: str) -> None:
        """Blend SeedVR2 output with Lanczos-upscaled original using depth map."""
        # Load images
        # seedvr2_output is the AI upscaled image (high detail)
        ai_img = cv2.imread(seedvr2_path)
        if ai_img is None:
            raise ValueError(f"Could not load SeedVR2 output: {seedvr2_path}")

        # input_path is the original LQ image
        orig_img = cv2.imread(input_path)
        if orig_img is None:
            raise ValueError(f"Could not load input image: {input_path}")
        
        # depth_path is the depth map
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            raise ValueError(f"Could not load depth map: {depth_path}")

        h, w = ai_img.shape[:2]

        # Upscale original using Lanczos4 to match AI output size
        lanczos_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # Resize depth map to output size
        depth_resized = cv2.resize(depth_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize depth to [0, 1] float. 
        # Lighter (255) = Near/Salient -> Keep AI (more detail)
        # Darker (0) = Far/Background -> Keep Original (less hallucination)
        alpha = depth_resized.astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = alpha[..., np.newaxis]  # broadcast to [H, W, 1] to match 3 channels

        # Blend: final = alpha * AI + (1 - alpha) * Lanczos
        # If alpha=1 (white), we get AI. If alpha=0 (black), we get Lanczos.
        ai_float = ai_img.astype(np.float32)
        lanczos_float = lanczos_img.astype(np.float32)
        
        blended = (alpha * ai_float + (1.0 - alpha) * lanczos_float)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Save over the original output or create a new one?
        # Creating a new one is safer but maybe user wants just one result.
        # Let's overwrite as implied by the requirement "provided depth image as a mask for upscaler"
        # actually, let's save as _blended to compare if they want? 
        # But standard behavior for upscalers is usually one output.
        # Let's overwrite for now to match the "feature" description of being PART of the process.
        cv2.imwrite(seedvr2_path, blended)
        logger.info("Applied depth-guided blending to %s", seedvr2_path)

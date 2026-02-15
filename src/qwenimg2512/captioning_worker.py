"""Background worker for VL captioning via local llama.cpp CLI."""

from __future__ import annotations

import logging
import re
import subprocess

from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ModelPaths

logger = logging.getLogger(__name__)

LLAMA_MTMD = "/home/matatrata/AI/llama.cpp/build/bin/llama-mtmd-cli"

CAPTION_PROMPT = (
    "Describe this image in detail as a text-to-image generation prompt. "
    "Include subject, composition, lighting, style, colors, and mood. "
    "Be specific and descriptive. Output only the prompt, no preamble."
)


class CaptioningWorker(QThread):
    """Runs Qwen2.5-VL captioning via llama-mtmd-cli subprocess."""

    caption_ready = Signal(str)  # generated caption
    stage_changed = Signal(str)  # status message
    error_occurred = Signal(str)  # error message

    def __init__(self, image_path: str, model_paths: ModelPaths) -> None:
        super().__init__()
        self._image_path = image_path
        self._model_paths = model_paths
        self._process: subprocess.Popen | None = None

    def run(self) -> None:
        try:
            self.stage_changed.emit("Captioning image...")

            cmd = [
                LLAMA_MTMD,
                "-m", self._model_paths.vl_model,
                "--mmproj", self._model_paths.mmproj,
                "--image", self._image_path,
                "-p", CAPTION_PROMPT,
                "-n", "512",
                "--temp", "0.3",
                "-ngl", "99",
                "-c", "4096",
            ]

            logger.info("Running llama-mtmd-cli for captioning")

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = self._process.communicate(timeout=120)

            # llama-mtmd-cli writes generated text to stderr, mixed with logs.
            # The caption appears after the last "image decoded" line and before
            # "llama_perf_context_print". Extract it.
            caption = self._extract_caption(stdout, stderr)

            if not caption:
                logger.error("Could not extract caption from output")
                logger.error("stdout: %r", stdout[:500])
                logger.error("stderr (tail): %r", stderr[-1000:])
                self.error_occurred.emit(
                    f"llama-mtmd-cli returned no caption (exit {self._process.returncode}). "
                    "Check terminal logs for details."
                )
                return

            logger.info("Caption generated: %s", caption[:100])
            self.caption_ready.emit(caption)

        except subprocess.TimeoutExpired:
            if self._process:
                self._process.kill()
            self.error_occurred.emit("Captioning timed out (120s)")
        except Exception as e:
            logger.exception("Captioning failed")
            self.error_occurred.emit(str(e))
        finally:
            self._process = None
            self.stage_changed.emit("Ready")

    @staticmethod
    def _extract_caption(stdout: str, stderr: str) -> str:
        # Prefer stdout if it has meaningful content
        if stdout.strip():
            return stdout.strip()

        # Extract from stderr: text between last "image decoded" line
        # and "llama_perf_context_print"
        text = stderr

        # Cut after the last "image decoded" log line
        match = re.search(r"image decoded[^\n]*\n", text)
        if match:
            text = text[match.end():]

        # Cut before perf stats
        perf_idx = text.find("llama_perf_context_print")
        if perf_idx >= 0:
            text = text[:perf_idx]

        # Strip trailing special tokens
        for tok in ("<|im_end|>", "<|endoftext|>", "[end of text]"):
            text = text.split(tok)[0]

        caption = text.strip()
        # Remove any remaining log lines (they start with known prefixes)
        lines = []
        for line in caption.split("\n"):
            if re.match(r"^(main:|llama_|ggml_|alloc_|warmup:|WARN:)", line):
                continue
            lines.append(line)

        return "\n".join(lines).strip()

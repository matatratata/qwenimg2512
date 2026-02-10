"""Background worker for VL captioning via llama-cpp-python."""

from __future__ import annotations

import base64
import gc
import logging

from PySide6.QtCore import QThread, Signal

from qwenimg2512.config import ModelPaths

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe this image in detail as a text-to-image generation prompt. "
    "Include subject, composition, lighting, style, colors, and mood. "
    "Be specific and descriptive. Output only the prompt, no preamble."
)


class CaptioningWorker(QThread):
    """Runs Qwen2.5-VL captioning via llama-cpp-python in a background thread."""

    caption_ready = Signal(str)  # generated caption
    stage_changed = Signal(str)  # status message
    error_occurred = Signal(str)  # error message

    def __init__(self, image_path: str, model_paths: ModelPaths) -> None:
        super().__init__()
        self._image_path = image_path
        self._model_paths = model_paths

    def run(self) -> None:
        model = None
        try:
            self.stage_changed.emit("Loading VL model...")

            from llama_cpp import Llama

            model = Llama(
                model_path=self._model_paths.vl_model,
                chat_handler=None,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )

            # Load mmproj for vision
            from llama_cpp.llama_chat_format import Llava16ChatHandler

            chat_handler = Llava16ChatHandler(clip_model_path=self._model_paths.mmproj)
            model.chat_handler = chat_handler

            self.stage_changed.emit("Captioning image...")

            # Read and encode image as base64 data URI
            with open(self._image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine mime type from extension
            ext = self._image_path.rsplit(".", 1)[-1].lower()
            mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "bmp": "image/bmp", "webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/png")
            data_uri = f"data:{mime_type};base64,{image_data}"

            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_uri}},
                            {"type": "text", "text": CAPTION_PROMPT},
                        ],
                    }
                ],
                max_tokens=512,
                temperature=0.3,
            )

            caption = response["choices"][0]["message"]["content"].strip()
            logger.info("Caption generated: %s", caption[:100])
            self.caption_ready.emit(caption)

        except Exception as e:
            logger.exception("Captioning failed")
            self.error_occurred.emit(str(e))
        finally:
            del model
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.stage_changed.emit("Ready")

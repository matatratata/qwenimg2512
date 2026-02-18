"""Configuration management with JSON persistence."""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / ".config" / "qwenimg2512"
CONFIG_FILE = CONFIG_DIR / "settings.json"

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 (1328x1328)": (1328, 1328),
    "16:9 (1664x928)": (1664, 928),
    "9:16 (928x1664)": (928, 1664),
    "4:3 (1472x1104)": (1472, 1104),
    "3:4 (1104x1472)": (1104, 1472),
    "3:2 (1584x1056)": (1584, 1056),
    "2:3 (1056x1584)": (1056, 1584),
}

DEFAULT_NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, deformed fingers, "
    "oversaturated, wax figure look, no face detail, overly smooth, "
    "AI-looking, chaotic composition, blurry text, distorted text"
)

MODEL_VARIANTS = {
    "4-bit (BnB, ~13GB VRAM)": "unsloth/Qwen-Image-2512-unsloth-bnb-4bit",
    "Full precision (~24GB VRAM)": "Qwen/Qwen-Image-2512",
    "GGUF Q8_0 (local)": "gguf_local",
}

_MODELS_DIR = Path.home() / "AI" / "Models"


@dataclass
class ModelPaths:
    diffusion_gguf: str = str(_MODELS_DIR / "Qwen-Image-2512-GGUF" / "qwen-image-2512-Q8_0.gguf")
    vl_model: str = str(_MODELS_DIR / "Qwen2.5-VL-7B-Instruct-GGUF" / "Qwen2.5-VL-7B-Instruct-UD-Q8_K_XL.gguf")
    mmproj: str = str(_MODELS_DIR / "Qwen2.5-VL-7B-Instruct-GGUF" / "mmproj-BF16.gguf")
    vae: str = str(_MODELS_DIR / "Qwen2.5-VL-7B-Instruct-GGUF" / "qwen_image_vae.safetensors")
    controlnet_path: str = str(_MODELS_DIR / "Qwen-Image-2512-Fun-Controlnet-Union" / "Qwen-Image-2512-Fun-Controlnet-Union-2602.safetensors")
    base_model_dir: str = str(_MODELS_DIR / "Qwen-Image-2512")
    edit_gguf: str = str(_MODELS_DIR / "Qwen-Image-Edit-2511-GGUF" / "qwen-image-edit-2511-Q8_0.gguf")
    edit_base_model_dir: str = str(_MODELS_DIR / "Qwen-Image-Edit-2511")
    llama_cpp_cli: str = "llama-mtmd-cli"


@dataclass
class GenerationSettings:
    prompt: str = ""
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    aspect_ratio: str = "1:1 (1328x1328)"
    num_inference_steps: int = 50
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = -1
    model_variant: str = "4-bit (BnB, ~13GB VRAM)"
    output_dir: str = str(Path.home() / "Pictures" / "qwenimg2512")
    input_image_path: str = ""
    img2img_strength: float = 0.7
    alpha_fill: str = "grey"
    lora_path: str = ""
    lora_scale_start: float = 1.0
    lora_scale_end: float = 1.0
    lora_step_start: int = 0
    lora_step_end: int = -1
    controlnet_enabled: bool = False
    control_type: str = "canny"
    control_image_path: str = ""
    controlnet_conditioning_scale: float = 0.80
    control_guidance_start: float = 0.0
    control_guidance_end: float = 1.0


@dataclass
class EditSettings:
    prompt: str = ""
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    aspect_ratio: str = "1:1 (1328x1328)"
    num_inference_steps: int = 40
    true_cfg_scale: float = 4.0
    guidance_scale: float = 1.0
    seed: int = -1
    output_dir: str = str(Path.home() / "Pictures" / "qwenimg2512")
    ref_image_1: str = ""
    ref_image_2: str = ""
    ref_image_3: str = ""
    lora_path: str = ""
    lora_scale_start: float = 1.0
    lora_scale_end: float = 1.0
    lora_step_start: int = 0
    lora_step_end: int = -1


@dataclass
class Config:
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    edit: EditSettings = field(default_factory=EditSettings)
    model_paths: ModelPaths = field(default_factory=ModelPaths)

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))
        logger.info("Settings saved to %s", CONFIG_FILE)

    @classmethod
    def load(cls) -> Config:
        if not CONFIG_FILE.exists():
            return cls()
        try:
            data = json.loads(CONFIG_FILE.read_text())
            gen = _filter_kwargs(GenerationSettings, data.get("generation", {}))
            edit = _filter_kwargs(EditSettings, data.get("edit", {}))
            paths = _filter_kwargs(ModelPaths, data.get("model_paths", {}))
            return cls(
                generation=GenerationSettings(**gen),
                edit=EditSettings(**edit),
                model_paths=ModelPaths(**paths),
            )
        except Exception:
            logger.exception("Failed to load config, using defaults")
            return cls()


def _filter_kwargs(cls: type, kwargs_dict: dict) -> dict:
    """Filter a dict to only keys that are valid fields of *cls*."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs_dict.items() if k in valid}

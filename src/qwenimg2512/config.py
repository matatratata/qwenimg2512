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
    # Original
    "1:1 (1328x1328)": (1328, 1328),
    "16:9 (1664x928)": (1664, 928),
    "9:16 (928x1664)": (928, 1664),
    "4:3 (1472x1104)": (1472, 1104),
    "3:4 (1104x1472)": (1104, 1472),
    "3:2 (1584x1056)": (1584, 1056),
    "2:3 (1056x1584)": (1056, 1584),
    
    # Original Half
    "1:1 Half (664x664)": (664, 664),
    "16:9 Half (832x464)": (832, 464),
    "9:16 Half (464x832)": (464, 832),
    "4:3 Half (736x552)": (736, 552),
    "3:4 Half (552x736)": (552, 736),
    "3:2 Half (792x528)": (792, 528),
    "2:3 Half (528x792)": (528, 792),

    # Square standard
    "1:1 Square (1024x1024)": (1024, 1024),
    "1:1 Square Half (512x512)": (512, 512),

    # Wan cinematic resolutions
    "16:9 Wan (1280x720)": (1280, 720),
    "9:16 Wan (720x1280)": (720, 1280),
    "16:9 Wan Low (832x480)": (832, 480),
    "9:16 Wan Low (480x832)": (480, 832),
    
    # LTX-2 resolutions
    "LTX-2 (1216x704)": (1216, 704),
    "LTX-2 Double (2432x1408)": (2432, 1408),
    "LTX-2 Half (608x352)": (608, 352),
    "LTX-2 Portrait (704x1216)": (704, 1216),
    "LTX-2 Portrait Half (352x608)": (352, 608),
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
    edit_2509_gguf: str = str(_MODELS_DIR / "Qwen-Image-Edit-2509-GGUF" / "qwen-image-edit-2509-Q8_0.gguf")
    edit_2509_base_model_dir: str = str(_MODELS_DIR / "Qwen-Image-Edit-2509")
    edit_2509_telestyle_fused_dir: str = str(_MODELS_DIR / "Qwen-Image-Edit-2509-TeleStyle-Fused")
    telestyle_lora: str = str(_MODELS_DIR / "TeleStyle" / "diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors")
    telestyle_speedup: str = str(_MODELS_DIR / "TeleStyle" / "diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors")
    llama_cpp_cli: str = str(Path.home() / "AI/llama.cpp/build/bin/llama-mtmd-cli")
    seedvr2_gguf: str = str(_MODELS_DIR / "SeedVR2" / "seedvr2_ema_7b-Q8_0.gguf")
    seedvr2_vae: str = str(_MODELS_DIR / "SeedVR2" / "ema_vae.pth")
    seedvr2_model_dir: str = str(_MODELS_DIR / "SeedVR2")
    seedvr2_cli: str = str(Path.home() / "AI" / "ComfyUI-SeedVR2_VideoUpscaler" / "inference_cli.py")
    wan_gguf: str = str(_MODELS_DIR / "Wan2.2-I2V-A14B-GGUF" / "wan2.2_i2v_low_noise_14B_Q8_0.gguf")
    wan_base_model_dir: str = str(_MODELS_DIR / "Wan2.2-I2V-A14B-Diffusers")


@dataclass
class GenerationSettings:
    prompt: str = ""
    sampler_name: str = "euler"
    schedule_name: str = "default"
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
    sampler_name: str = "euler"
    schedule_name: str = "default"
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
    ref_fit_mode_1: str = "cover"
    ref_fit_mode_2: str = "cover"
    ref_fit_mode_3: str = "cover"
    lora_path: str = ""
    lora_scale_start: float = 1.0
    lora_scale_end: float = 1.0
    lora_step_start: int = 0
    lora_step_end: int = -1
    ref_strength_1: float = 1.0
    ref_strength_2: float = 1.0
    ref_strength_3: float = 1.0
    ffn_chunk_size: int = 0    # 0 = disabled; try 2048 for large resolutions
    blocks_to_swap: int = 0   # 0 = disabled; N = move last N blocks to CPU
    attn_chunk_size: int = 0  # 0 = disabled; try 4096 for large resolutions


@dataclass
class Edit2509Settings:
    use_telestyle: bool = False
    prompt: str = ""
    sampler_name: str = "euler"
    schedule_name: str = "default"
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
    ref_fit_mode_1: str = "cover"
    ref_fit_mode_2: str = "cover"
    ref_fit_mode_3: str = "cover"
    lora_path: str = ""
    lora_scale_start: float = 1.0
    lora_scale_end: float = 1.0
    lora_step_start: int = 0
    lora_step_end: int = -1
    lora_path_2: str = ""
    lora_scale_start_2: float = 1.0
    lora_scale_end_2: float = 1.0
    lora_step_start_2: int = 0
    lora_step_end_2: int = -1


@dataclass
class SeedVR2Settings:
    input_image: str = ""
    sampler_name: str = "euler"
    depth_map_path: str = ""
    output_dir: str = str(Path.home() / "Pictures" / "qwenimg2512")
    resolution: int = 1080
    seed: int = 42
    input_noise_scale: float = 0.0
    latent_noise_scale: float = 0.0
    color_correction: str = "lab"
    vae_tiling: bool = True
    blocks_to_swap: int = 0


@dataclass
class WanSettings:
    input_image: str = ""
    prompt: str = "Cinematic slow pan, volumetric fog, anamorphic lens flare, 35mm film grain, 8k resolution, highly detailed movie still"
    negative_prompt: str = "low quality, worst quality, deformed, distorted, watermark"
    resolution: str = "832x480"
    frames: int = 33
    num_inference_steps: int = 40
    guidance_scale: float = 5.0
    shift: float = 5.0
    sampler_name: str = "euler"
    schedule_name: str = "default"
    seed: int = -1
    output_dir: str = str(Path.home() / "Pictures" / "qwenimg2512")
    extract_still: bool = True


@dataclass
class Config:
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    edit: EditSettings = field(default_factory=EditSettings)
    edit_2509: Edit2509Settings = field(default_factory=Edit2509Settings)
    seedvr2: SeedVR2Settings = field(default_factory=SeedVR2Settings)
    wan: WanSettings = field(default_factory=WanSettings)
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
            edit_2509 = _filter_kwargs(Edit2509Settings, data.get("edit_2509", {}))
            seedvr2 = _filter_kwargs(SeedVR2Settings, data.get("seedvr2", {}))
            wan = _filter_kwargs(WanSettings, data.get("wan", {}))
            paths = _filter_kwargs(ModelPaths, data.get("model_paths", {}))
            return cls(
                generation=GenerationSettings(**gen),
                edit=EditSettings(**edit),
                edit_2509=Edit2509Settings(**edit_2509),
                seedvr2=SeedVR2Settings(**seedvr2),
                wan=WanSettings(**wan),
                model_paths=ModelPaths(**paths),
            )
        except Exception:
            logger.exception("Failed to load config, using defaults")
            return cls()


def _filter_kwargs(cls: type, kwargs_dict: dict) -> dict:
    """Filter a dict to only keys that are valid fields of *cls*."""
    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in kwargs_dict.items() if k in valid}

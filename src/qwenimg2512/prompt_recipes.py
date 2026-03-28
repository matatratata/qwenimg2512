"""Built-in and custom prompt recipes for the Prompt Crafter tab.

Each recipe pairs a descriptive template with its associated LoRA adapter
(when applicable) so the user can craft high-quality prompts quickly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CUSTOM_RECIPES_FILE = Path.home() / ".config" / "qwenimg2512" / "custom_recipes.json"


@dataclass
class PromptRecipe:
    """One prompt recipe (template + optional LoRA coordinates)."""

    name: str
    category: str
    template: str
    placeholders: list[str] = field(default_factory=list)
    placeholder_defaults: dict[str, str] = field(default_factory=dict)
    lora_repo: str = ""
    lora_weights: str = ""
    tip: str = ""
    num_images: int = 1
    builtin: bool = True  # False for user-created recipes


# ---------------------------------------------------------------------------
#  Prompt recipe catalogue – derived from the curated examples in
#  github.com/PRITHIVSAKTHIUR/Qwen-Image-Edit-2511-LoRAs-Fast-Lazy-Load
# ---------------------------------------------------------------------------

BUILTIN_RECIPES: list[PromptRecipe] = [
    # ── Style Transfer ─────────────────────────────────────────────────
    PromptRecipe(
        name="Photo → Anime",
        category="Style Transfer",
        template="Transform into anime{detail_note}.",
        placeholders=["detail_note"],
        placeholder_defaults={
            "detail_note": ", preserving background details"
        },
        lora_repo="autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime",
        lora_weights="Qwen-Image-Edit-2509-Photo-to-Anime_000001000.safetensors",
        tip="Works best with a single reference image of a person or scene.",
        num_images=1,
    ),
    PromptRecipe(
        name="Anime V2 (preserve realism)",
        category="Style Transfer",
        template="Transform into anime (while preserving the background and remaining elements maintaining realism and original details.){extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Anime",
        lora_weights="Qwen-Image-Edit-2511-Anime-2000.safetensors",
        tip="V2 is trained on the 2511 base and keeps background realism.",
        num_images=1,
    ),
    PromptRecipe(
        name="Style Transfer (two images)",
        category="Style Transfer",
        template="Convert Image 1 to the style of Image 2.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="zooeyy/Style-Transfer",
        lora_weights="Style Transfer-Alpha-V0.1.safetensors",
        tip="Upload the content image as Ref 1 and the style image as Ref 2.",
        num_images=2,
    ),
    PromptRecipe(
        name="Anything → Realistic Photo",
        category="Style Transfer",
        template="Change the picture to realistic photograph.{detail}",
        placeholders=["detail"],
        placeholder_defaults={"detail": ""},
        lora_repo="lrzjason/Anything2Real_2601",
        lora_weights="anything2real_2601.safetensors",
        tip="Good for converting illustrations/3D renders back to photorealism.",
        num_images=1,
    ),
    PromptRecipe(
        name="Pixar-Inspired 3D",
        category="Style Transfer",
        template="Transform it into Pixar-inspired 3D.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Pixar-Inspired-3D",
        lora_weights="PI3_20.safetensors",
        tip="Works well on characters and portrait shots.",
        num_images=1,
    ),
    PromptRecipe(
        name="Noir Comic Book",
        category="Style Transfer",
        template="Transform into a noir comic book style.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Noir-Comic-Book-Panel",
        lora_weights="Noir-Comic-Book-Panel_20.safetensors",
        tip="High-contrast black-and-white comic panel look.",
        num_images=1,
    ),
    PromptRecipe(
        name="Manga Tone",
        category="Style Transfer",
        template="Paint with manga tone.{detail}",
        placeholders=["detail"],
        placeholder_defaults={"detail": ""},
        lora_repo="nappa114514/Qwen-Image-Edit-2509-Manga-Tone",
        lora_weights="tone001.safetensors",
        tip="B&W manga screen-tone aesthetic.",
        num_images=1,
    ),
    PromptRecipe(
        name="Polaroid Photo",
        category="Style Transfer",
        template="cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed by hf‪‪‬ preserving realistic texture and details{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Polaroid-Photo",
        lora_weights="Qwen-Image-Edit-2511-Polaroid-Photo.safetensors",
        tip="Adds a retro Polaroid frame with warm tones.",
        num_images=1,
    ),
    PromptRecipe(
        name="Midnight Noir Eyes Spotlight",
        category="Style Transfer",
        template="Transform into Midnight Noir Eyes Spotlight.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight",
        lora_weights="Qwen-Image-Edit-2511-Midnight-Noir-Eyes-Spotlight.safetensors",
        tip="Dramatic noir lighting with spotlight focus on eyes.",
        num_images=1,
    ),

    # ── Camera / Angle ─────────────────────────────────────────────────
    PromptRecipe(
        name="Rotate Camera (dx8152)",
        category="Camera / Angle",
        template="Rotate the camera {angle} degrees to the {direction}.{extra}",
        placeholders=["angle", "direction", "extra"],
        placeholder_defaults={"angle": "45", "direction": "right", "extra": ""},
        lora_repo="dx8152/Qwen-Edit-2509-Multiple-angles",
        lora_weights="镜头转换.safetensors",
        tip="Use realistic angles (15-90°). Larger angles may cause artifacts.",
        num_images=1,
    ),
    PromptRecipe(
        name="Rotate Camera (Fal)",
        category="Camera / Angle",
        template="{view_description}.{extra}",
        placeholders=["view_description", "extra"],
        placeholder_defaults={"view_description": "Front-right quarter view", "extra": ""},
        lora_repo="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        lora_weights="qwen-image-edit-2511-multiple-angles-lora.safetensors",
        tip="Fal's 2511-native multi-angle LoRA. Describe the target viewpoint.",
        num_images=1,
    ),

    # ── Lighting ───────────────────────────────────────────────────────
    PromptRecipe(
        name="Light Migration (two images)",
        category="Lighting",
        template="Refer to the color tone, remove the original lighting from Image 1, and relight Image 1 based on the lighting and color tone of Image 2.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="dx8152/Qwen-Edit-2509-Light-Migration",
        lora_weights="参考色调.safetensors",
        tip="Ref 1 = target scene, Ref 2 = lighting reference.",
        num_images=2,
    ),
    PromptRecipe(
        name="Any-Light (two images)",
        category="Lighting",
        template="Apply the lighting from image 2 to image 1.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="lilylilith/QIE-2511-MP-AnyLight",
        lora_weights="QIE-2511-AnyLight_.safetensors",
        tip="A 2511-native relighting LoRA. Works like Light-Migration but newer.",
        num_images=2,
    ),
    PromptRecipe(
        name="Studio De-Light",
        category="Lighting",
        template="Neutral uniform lighting Preserve identity and composition.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/QIE-2511-Studio-DeLight",
        lora_weights="QIE-2511-Studio-DeLight-5000.safetensors",
        tip="Removes dramatic lighting, producing flat studio-like illumination.",
        num_images=1,
    ),
    PromptRecipe(
        name="Cinematic Flat-Log",
        category="Lighting",
        template="Transform into a cinematic flat log.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/QIE-2511-Cinematic-FlatLog-Control",
        lora_weights="QIE-2511-Cinematic-FlatLog-Control-3200.safetensors",
        tip="Flat / log color grading, useful as a starting point for color work.",
        num_images=1,
    ),

    # ── Enhancement / Upscale ──────────────────────────────────────────
    PromptRecipe(
        name="Upscale to 4K",
        category="Enhancement / Upscale",
        template="Upscale this picture to 4K resolution.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="starsfriday/Qwen-Image-Edit-2511-Upscale2K",
        lora_weights="qwen_image_edit_2511_upscale.safetensors",
        tip="Set the output resolution to ≥2048 on whichever axis you need.",
        num_images=1,
    ),
    PromptRecipe(
        name="Unblur & Upscale",
        category="Enhancement / Upscale",
        template="Unblur and upscale.{detail}",
        placeholders=["detail"],
        placeholder_defaults={"detail": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale",
        lora_weights="Qwen-Image-Edit-Unblur-Upscale_15.safetensors",
        tip="Best for slightly blurry / low-res inputs.",
        num_images=1,
    ),

    # ── Portrait ───────────────────────────────────────────────────────
    PromptRecipe(
        name="Hyper-Realistic Portrait",
        category="Portrait",
        template="Transform into a hyper-realistic face portrait.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait",
        lora_weights="HRP_20.safetensors",
        tip="Focuses on extreme facial detail.",
        num_images=1,
    ),
    PromptRecipe(
        name="Ultra-Realistic Portrait",
        category="Portrait",
        template="ultra-realistic portrait.{extra}",
        placeholders=["extra"],
        placeholder_defaults={"extra": ""},
        lora_repo="prithivMLmods/Qwen-Image-Edit-2511-Ultra-Realistic-Portrait",
        lora_weights="URP_20.safetensors",
        tip="Similar to Hyper-Realistic but aimed at overall realism.",
        num_images=1,
    ),

    # ── Freeform (no LoRA) ─────────────────────────────────────────────
    PromptRecipe(
        name="Freeform Edit",
        category="Freeform",
        template="{prompt}",
        placeholders=["prompt"],
        placeholder_defaults={"prompt": "Describe the edit you want..."},
        tip="No LoRA attached — a blank canvas for any prompt.",
        num_images=1,
    ),
]


# ---------------------------------------------------------------------------
#  Custom recipe persistence
# ---------------------------------------------------------------------------

def load_custom_recipes() -> list[PromptRecipe]:
    """Load user-created recipes from disk."""
    if not CUSTOM_RECIPES_FILE.exists():
        return []
    try:
        data = json.loads(CUSTOM_RECIPES_FILE.read_text())
        recipes = []
        for entry in data:
            entry.pop("builtin", None)
            recipes.append(PromptRecipe(**entry, builtin=False))
        return recipes
    except Exception:
        logger.exception("Failed to load custom recipes")
        return []


def save_custom_recipes(recipes: list[PromptRecipe]) -> None:
    """Write user-created recipes to disk."""
    CUSTOM_RECIPES_FILE.parent.mkdir(parents=True, exist_ok=True)
    serializable = [asdict(r) for r in recipes if not r.builtin]
    CUSTOM_RECIPES_FILE.write_text(json.dumps(serializable, indent=2))
    logger.info("Saved %d custom recipes to %s", len(serializable), CUSTOM_RECIPES_FILE)

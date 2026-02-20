"""Resize utility with multiple fit modes (PIL-based).

Mirrors the logic of LTX-2's ``resize_with_fit_mode`` but operates on
PIL Images instead of torch Tensors, keeping the GUI dependency-free of
heavy ML frameworks.
"""

from __future__ import annotations

from PIL import Image


def resize_with_fit_mode(
    image: Image.Image,
    width: int,
    height: int,
    mode: str = "cover",
) -> Image.Image:
    """Resize *image* to (*width*, *height*) using the specified *mode*.

    Modes
    -----
    cover
        Scale so the image fully **covers** the target box, then centre-crop
        the excess.  No black bars, but parts of the image may be lost.
    contain
        Scale so the image fully **fits** within the target box, then pad the
        remaining area with black.  No cropping, but letterbox bars appear.
    stretch
        Resize to the exact target dimensions, ignoring aspect ratio.
    center
        Place the image centred in the target box without any scaling.
        Crop if larger; pad with black if smaller.
    """
    src_w, src_h = image.size

    if mode == "cover":
        return _cover(image, src_w, src_h, width, height)
    elif mode == "contain":
        return _contain(image, src_w, src_h, width, height)
    elif mode == "stretch":
        return image.resize((width, height), Image.LANCZOS)
    elif mode == "center":
        return _center(image, src_w, src_h, width, height)
    else:
        # Fallback to cover
        return _cover(image, src_w, src_h, width, height)


# ── private helpers ───────────────────────────────────────────────────────

def _cover(
    image: Image.Image, src_w: int, src_h: int, dst_w: int, dst_h: int,
) -> Image.Image:
    scale = max(dst_w / src_w, dst_h / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - dst_w) // 2
    top = (new_h - dst_h) // 2
    return image.crop((left, top, left + dst_w, top + dst_h))


def _contain(
    image: Image.Image, src_w: int, src_h: int, dst_w: int, dst_h: int,
) -> Image.Image:
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (dst_w, dst_h), (0, 0, 0))
    paste_x = (dst_w - new_w) // 2
    paste_y = (dst_h - new_h) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


def _center(
    image: Image.Image, src_w: int, src_h: int, dst_w: int, dst_h: int,
) -> Image.Image:
    canvas = Image.new("RGB", (dst_w, dst_h), (0, 0, 0))
    paste_x = (dst_w - src_w) // 2
    paste_y = (dst_h - src_h) // 2
    # Paste handles negative offsets (crops automatically)
    canvas.paste(image, (paste_x, paste_y))
    return canvas

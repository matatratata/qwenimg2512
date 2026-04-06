"""
Minimal render_tools shim for sg_core.

Only contains the functions actually needed by project.py at runtime.
The remaining imports (prepare_baking, bake_texture, unwrap) are provided
as no-op stubs so the module-level import in project.py succeeds.
"""
import bpy


def _get_camera_resolution(cam_obj, scene):
    """Return (res_x, res_y) for a camera.  Falls back to scene render
    resolution if no per-camera resolution is stored."""
    if "sg_res_x" in cam_obj and "sg_res_y" in cam_obj:
        return int(cam_obj["sg_res_x"]), int(cam_obj["sg_res_y"])
    return scene.render.resolution_x, scene.render.resolution_y


# ── Stubs for functions imported by project.py but not called ──────────
# These are only used by PBR baking paths (project_pbr_to_bsdf,
# _bake_ao_for_objects) which QwenBuild never invokes.

def prepare_baking(*args, **kwargs):
    raise NotImplementedError("prepare_baking is not available in sg_core shim")

def bake_texture(*args, **kwargs):
    raise NotImplementedError("bake_texture is not available in sg_core shim")

def unwrap(*args, **kwargs):
    raise NotImplementedError("unwrap is not available in sg_core shim")

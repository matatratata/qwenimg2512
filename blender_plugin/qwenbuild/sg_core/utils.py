"""
Minimal utils shim for sg_core.

Contains only the utility functions required by project.py.
All Blender Operator classes (AddHDRI, ApplyModifiers, etc.) are omitted.
"""
import bpy
import os
from datetime import datetime


def sg_modal_active(context):
    """Stub — always returns False in QwenBuild context."""
    return False


def get_last_material_index(obj):
    """
    Get the index of the last material of the object.
    The index is hidden inside default value of (the only) subtract node.
    If there are no subtract nodes, return -1.
    """
    highest_index = -1
    if obj.data.materials:
        for mat in obj.data.materials:
            if mat and mat.use_nodes:
                for node in mat.node_tree.nodes:
                    if node.type == 'MATH' and node.operation == 'SUBTRACT':
                        if node.inputs[0].default_value > highest_index:
                            highest_index = node.inputs[0].default_value
    return int(highest_index)


def get_generation_dirs(context):
    """
    Gets the directory structure for the current generation session.
    Creates a dictionary with paths to all required subdirectories.
    """
    try:
        base_dir = context.preferences.addons[__package__].preferences.output_dir
    except (KeyError, AttributeError):
        # Fallback: use a temp dir if the addon pref isn't available
        import tempfile
        base_dir = os.path.join(tempfile.gettempdir(), "qwenbuild_sg_core")

    scene_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
    if not scene_name:
        scene_name = context.scene.name
    timestamp = getattr(context.scene, 'output_timestamp', '')

    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        try:
            context.scene.output_timestamp = timestamp
        except AttributeError:
            pass

    revision_dir = os.path.join(base_dir, scene_name, timestamp)
    return {
        "revision": revision_dir,
        "controlnet_root": os.path.join(revision_dir, "controlnet"),
        "controlnet": {
            "depth": os.path.join(revision_dir, "controlnet", "depth"),
            "canny": os.path.join(revision_dir, "controlnet", "canny"),
            "normal": os.path.join(revision_dir, "controlnet", "normal"),
            "workbench": os.path.join(revision_dir, "controlnet", "workbench"),
            "viewport": os.path.join(revision_dir, "controlnet", "viewport"),
        },
        "generated": os.path.join(revision_dir, "generated"),
        "generated_baked": os.path.join(revision_dir, "generated_baked"),
        "baked": os.path.join(revision_dir, "baked"),
        "inpaint": {
            "render": os.path.join(revision_dir, "inpaint", "render"),
            "visibility": os.path.join(revision_dir, "inpaint", "visibility"),
        },
        "inpaint_root": os.path.join(revision_dir, "inpaint"),
        "uv_inpaint": {
            "visibility": os.path.join(revision_dir, "uv_inpaint", "uv_visibility"),
        },
        "uv_inpaint_root": os.path.join(revision_dir, "uv_inpaint"),
        "misc": os.path.join(revision_dir, "misc"),
        "pbr": os.path.join(revision_dir, "pbr"),
        "viewport_renders": os.path.join(revision_dir, "viewport_renders"),
    }


def ensure_dirs_exist(dirs_dict):
    """Ensure that all required directories exist."""
    os.makedirs(dirs_dict["revision"], exist_ok=True)
    os.makedirs(dirs_dict["generated"], exist_ok=True)
    os.makedirs(dirs_dict["generated_baked"], exist_ok=True)
    os.makedirs(dirs_dict["baked"], exist_ok=True)
    for key, path in dirs_dict["controlnet"].items():
        os.makedirs(path, exist_ok=True)
    for key, path in dirs_dict["inpaint"].items():
        os.makedirs(path, exist_ok=True)
    for key, path in dirs_dict["uv_inpaint"].items():
        os.makedirs(path, exist_ok=True)
    os.makedirs(dirs_dict["misc"], exist_ok=True)
    os.makedirs(dirs_dict["pbr"], exist_ok=True)
    os.makedirs(dirs_dict["viewport_renders"], exist_ok=True)


def get_file_path(context, file_type, subtype=None, filename=None,
                  camera_id=None, object_name=None, material_id=None,
                  legacy=False):
    """Generate the full file path for saving images."""
    dirs = get_generation_dirs(context)
    frame_suffix = "0001" if legacy or bpy.app.version < (5, 0, 0) else ""
    ensure_dirs_exist(dirs)

    if file_type == "controlnet" and subtype:
        base_dir = dirs["controlnet"][subtype]
        if not filename:
            if subtype == "depth":
                filename = f"depth_map{camera_id}{frame_suffix}" if camera_id is not None else "depth_map_grid"
            elif subtype == "canny":
                filename = f"canny{camera_id}{frame_suffix}" if camera_id is not None else "canny_grid"
            elif subtype == "normal":
                filename = f"normal_map{camera_id}{frame_suffix}" if camera_id is not None else "normal_grid"
            elif subtype == "workbench":
                filename = f"render{camera_id}{frame_suffix}" if camera_id is not None else "render_grid"
            elif subtype == "viewport":
                filename = f"viewport{camera_id}" if camera_id is not None else "viewport_grid"
        return os.path.join(base_dir, f"{filename}.png")

    elif file_type == "generated":
        base_dir = dirs["generated"]
        material_suffix = f"-{material_id}" if material_id is not None else ""
        return os.path.join(base_dir,
                            f"generated_image{camera_id}{material_suffix}-0001.png"
                            if camera_id is not None else "generated_image.png")

    elif file_type == "generated_baked":
        base_dir = dirs["generated_baked"]
        if object_name:
            material_suffix = f"{camera_id}-{material_id}" if material_id is not None else ""
            return os.path.join(base_dir, f"{object_name}_baked{material_suffix}.png")

    elif file_type == "baked":
        base_dir = dirs["baked"]
        if not filename:
            filename = f"{object_name}" if object_name else "baked_texture"
        return os.path.join(base_dir, f"{filename}.png")

    elif file_type == "inpaint" and subtype:
        base_dir = dirs["inpaint"][subtype]
        if subtype == "render":
            filename = f"ctx_render{camera_id}{frame_suffix}" if not filename else filename
        elif subtype == "visibility":
            filename = f"ctx_render{camera_id}_visibility{frame_suffix}" if not filename else filename
        return os.path.join(base_dir, f"{filename}.png")

    elif file_type == "uv_inpaint" and subtype:
        base_dir = dirs["uv_inpaint"][subtype]
        if subtype == "visibility":
            filename = f"{object_name}_baked_visibility" if not filename else filename
        return os.path.join(base_dir, f"{filename}.png")

    elif file_type == "pbr" and subtype:
        base_dir = dirs["pbr"]
        material_suffix = f"-{material_id}" if material_id is not None else ""
        if camera_id is not None:
            return os.path.join(base_dir, f"pbr_{subtype}_cam{camera_id}{material_suffix}.png")
        else:
            return os.path.join(base_dir, f"pbr_{subtype}{material_suffix}.png")

    return os.path.join(dirs["revision"], f"{filename or 'file'}.png")


def get_dir_path(context, file_type):
    """Get the directory path for a specific file type."""
    dirs = get_generation_dirs(context)
    ensure_dirs_exist(dirs)

    if file_type == "revision":
        return dirs["revision"]
    elif file_type == "controlnet":
        return dirs["controlnet"]
    elif file_type == "generated":
        return dirs["generated"]
    elif file_type == "generated_baked":
        return dirs["generated_baked"]
    elif file_type == "baked":
        return dirs["baked"]
    elif file_type == "inpaint":
        return dirs["inpaint"]
    elif file_type == "uv_inpaint":
        return dirs["uv_inpaint"]
    else:
        return dirs["misc"]


def remove_empty_dirs(context, dirs_obj=None):
    """Remove empty directories from the generation structure."""
    if dirs_obj is None:
        dirs_obj = get_generation_dirs(context)
    for key, value in dirs_obj.items():
        if isinstance(value, dict):
            remove_empty_dirs(context, dirs_obj=value)
        else:
            if os.path.exists(value) and not os.listdir(value):
                os.rmdir(value)


def get_compositor_node_tree(scene):
    """Get the compositor node tree, handling API differences."""
    if hasattr(scene, 'compositing_node_group'):
        if scene.compositing_node_group is None:
            new_tree = bpy.data.node_groups.new(name="Compositing", type='CompositorNodeTree')
            scene.compositing_node_group = new_tree
        return scene.compositing_node_group
    if hasattr(scene, 'node_tree'):
        return scene.node_tree
    return None


def configure_output_node_paths(node, directory, filename):
    """Configure output node paths using Blender 5.0+ semantics with 4.x fallback."""
    if hasattr(node.format, "media_type"):
        node.format.media_type = 'IMAGE'
    node.format.file_format = "PNG"
    node.format.color_depth = '8'
    if hasattr(node, "directory"):
        node.directory = directory
    else:
        node.base_path = directory
    if hasattr(node, "file_name"):
        node.file_name = ""
    slot = None
    if hasattr(node, "file_output_items"):
        items = node.file_output_items
        if items and len(items) > 0:
            slot = items[0]
            slot.name = filename
            if hasattr(slot, "path"):
                slot.path = filename
        else:
            slot = items.new(name=filename, socket_type='RGBA')
            if hasattr(slot, "path"):
                slot.path = filename
    elif hasattr(node, "file_slots"):
        slots = node.file_slots
        if slots and len(slots) > 0:
            slot = slots[0]
            if hasattr(slot, "path"):
                slot.path = filename
        else:
            slot = slots.new(filename)
    return slot


def get_eevee_engine_id():
    """Return the correct Eevee engine identifier for the current Blender version."""
    if bpy.app.version >= (5, 0, 0):
        return 'BLENDER_EEVEE'
    return 'BLENDER_EEVEE_NEXT'

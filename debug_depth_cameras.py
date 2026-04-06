"""
Debug script: Run in Blender's Script Editor.
Switches between all cameras and renders a depth map for each one.
Outputs go to /tmp/sg_debug_depth/

Check the resulting PNGs to see which cameras produce white depth maps.
"""
import bpy
import os

OUTPUT_DIR = "/tmp/sg_debug_depth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gather and sort cameras
cameras = sorted(
    [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA'],
    key=lambda c: c.name
)

print(f"\n{'='*60}")
print(f"DEBUG DEPTH: Found {len(cameras)} cameras")
for i, cam in enumerate(cameras):
    print(f"  [{i}] {cam.name}  loc={tuple(round(v,2) for v in cam.location)} rot=({round(cam.rotation_euler[0]*180/3.14159,1)}, {round(cam.rotation_euler[1]*180/3.14159,1)}, {round(cam.rotation_euler[2]*180/3.14159,1)})")
print(f"{'='*60}\n")

# Save original state
orig_camera = bpy.context.scene.camera
orig_engine = bpy.context.scene.render.engine
orig_vt = bpy.context.scene.view_settings.view_transform
orig_film = bpy.context.scene.render.film_transparent
orig_comp = bpy.context.scene.render.use_compositing
orig_fp = bpy.context.scene.render.filepath

view_layer = bpy.context.view_layer

for i, cam in enumerate(cameras):
    print(f"\n--- Camera [{i}] {cam.name} ---")

    # 1. Switch camera
    bpy.context.scene.camera = cam
    print(f"  scene.camera = {bpy.context.scene.camera.name}")

    # 2. Force depsgraph update
    bpy.context.view_layer.update()
    print(f"  view_layer.update() done")

    # 3. Print camera data
    cam_data = cam.data
    print(f"  type={cam_data.type}  lens={cam_data.lens}mm  "
          f"clip_start={cam_data.clip_start}  clip_end={cam_data.clip_end}")
    print(f"  sensor_width={cam_data.sensor_width}  sensor_height={cam_data.sensor_height}")

    # 4. Setup depth render (mirrors StableGen's export_depthmap)
    bpy.context.scene.frame_set(1)
    bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
    bpy.context.scene.display_settings.display_device = 'sRGB'
    bpy.context.scene.view_settings.view_transform = 'Raw'

    orig_pass_z = view_layer.use_pass_z
    view_layer.use_pass_z = True

    bpy.context.scene.render.use_compositing = True
    bpy.context.scene.use_nodes = True

    # Get compositor node tree (Blender 5.x uses compositing_node_group)
    if hasattr(bpy.context.scene, 'compositing_node_group'):
        if bpy.context.scene.compositing_node_group is None:
            new_tree = bpy.data.node_groups.new(name="Compositing", type='CompositorNodeTree')
            bpy.context.scene.compositing_node_group = new_tree
        node_tree = bpy.context.scene.compositing_node_group
    elif hasattr(bpy.context.scene, 'node_tree'):
        node_tree = bpy.context.scene.node_tree
    else:
        print(f"  ERROR: No compositor node tree API found!")
        continue

    nodes = node_tree.nodes
    links = node_tree.links

    bpy.context.scene.render.image_settings.file_format = 'PNG'

    # Clear nodes
    for node in list(nodes):
        nodes.remove(node)

    # Build compositor: RenderLayers → Normalize → Invert → OutputFile
    rl = nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    norm = nodes.new(type="CompositorNodeNormalize")
    norm.location = (200, 0)
    links.new(rl.outputs["Depth"], norm.inputs[0])

    inv = nodes.new(type="CompositorNodeInvert")
    inv.location = (400, 0)
    color_in = inv.inputs["Color"] if "Color" in inv.inputs else inv.inputs[1]
    links.new(norm.outputs[0], color_in)

    # Output file node — use Blender 5.x API with fallback
    out = nodes.new(type="CompositorNodeOutputFile")
    out.location = (600, 0)
    filename = f"depth_cam{i}"

    # Force single-image mode (Blender 5.x)
    if hasattr(out.format, "media_type"):
        out.format.media_type = 'IMAGE'
    out.format.file_format = "PNG"
    out.format.color_depth = '8'

    # Set directory
    if hasattr(out, "directory"):
        out.directory = OUTPUT_DIR
    elif hasattr(out, "base_path"):
        out.base_path = OUTPUT_DIR

    # Clear prefix
    if hasattr(out, "file_name"):
        out.file_name = ""

    # Configure slot
    if hasattr(out, "file_output_items"):
        items = out.file_output_items
        if items and len(items) > 0:
            items[0].name = filename
            if hasattr(items[0], "path"):
                items[0].path = filename
        else:
            slot = items.new(name=filename, socket_type='RGBA')
            if hasattr(slot, "path"):
                slot.path = filename
    elif hasattr(out, "file_slots"):
        slots = out.file_slots
        if slots and len(slots) > 0:
            slots[0].path = filename
        else:
            slots.new(filename)

    links.new(inv.outputs[0], out.inputs[0])

    # Output file node for RAW EXR
    out_raw = nodes.new(type="CompositorNodeOutputFile")
    out_raw.location = (600, -200)
    filename_raw = f"depth_cam{i}_raw"

    if hasattr(out_raw.format, "media_type"):
        out_raw.format.media_type = 'IMAGE'
    out_raw.format.file_format = "OPEN_EXR"
    out_raw.format.color_depth = '32'

    if hasattr(out_raw, "directory"):
        out_raw.directory = OUTPUT_DIR
    elif hasattr(out_raw, "base_path"):
        out_raw.base_path = OUTPUT_DIR
    if hasattr(out_raw, "file_name"):
        out_raw.file_name = ""

    if hasattr(out_raw, "file_output_items"):
        items = out_raw.file_output_items
        if items and len(items) > 0:
            items[0].name = filename_raw
            if hasattr(items[0], "path"):
                items[0].path = filename_raw
        else:
            slot = items.new(name=filename_raw, socket_type='FLOAT')
            if hasattr(slot, "path"):
                slot.path = filename_raw
    elif hasattr(out_raw, "file_slots"):
        slots = out_raw.file_slots
        if slots and len(slots) > 0:
            slots[0].path = filename_raw
        else:
            slots.new(filename_raw)

    links.new(rl.outputs["Depth"], out_raw.inputs[0])

    # 5. Render
    print(f"  Rendering depth map...")
    bpy.ops.render.render(write_still=True)

    # 6. Check output
    expected_name = f"depth_cam{i}"
    expected_raw = f"depth_cam{i}_raw"
    
    found_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(expected_name) and f.endswith(".png")]
    if found_files:
        for fn in found_files:
            fpath = os.path.join(OUTPUT_DIR, fn)
            fsize = os.path.getsize(fpath)
            print(f"  ✓ Output PNG: {fn}  ({fsize} bytes)")
    
    found_raw = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(expected_raw) and f.endswith(".exr")]
    if found_raw:
        for fn in found_raw:
            fpath = os.path.join(OUTPUT_DIR, fn)
            print(f"  ✓ Output RAW EXR: {fn}")
            
            # Analyze the EXR using Blender's Python Image API
            exr_img = bpy.data.images.load(fpath)
            exr_img.update()
            pixels = list(exr_img.pixels)  # RGBA flat array
            # Depth is stored in Red (and G,B, since it's a value node, but we'll use R)
            depth_vals = [pixels[idx] for idx in range(0, len(pixels), 4)]
            # Filter out true infinity (1e10 range)
            finite_depths = [d for d in depth_vals if d < 1e9]
            
            if len(finite_depths) > 0:
                print(f"      Min Depth: {min(finite_depths):.2f}m")
                print(f"      Max Depth: {max(finite_depths):.2f}m")
                print(f"      Avg Depth: {sum(finite_depths)/len(finite_depths):.2f}m")
                print(f"      Hit %:      {(len(finite_depths) / len(depth_vals)) * 100:.1f}% of pixels hit geometry")
            else:
                print(f"      WARNING: 100% of pixels are INFINITY! (Object not in frame)")
            
            bpy.data.images.remove(exr_img)

    # Restore pass
    view_layer.use_pass_z = orig_pass_z

# Restore original state
bpy.context.scene.camera = orig_camera
bpy.context.scene.render.engine = orig_engine
bpy.context.scene.view_settings.view_transform = orig_vt
bpy.context.scene.render.film_transparent = orig_film
bpy.context.scene.render.use_compositing = orig_comp
bpy.context.scene.render.filepath = orig_fp

print(f"\n{'='*60}")
print(f"DEBUG DEPTH COMPLETE — check {OUTPUT_DIR}/ for results")
print(f"{'='*60}\n")

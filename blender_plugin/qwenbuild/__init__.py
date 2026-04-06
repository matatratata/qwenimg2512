# SPDX-License-Identifier: GPL-3.0-or-later
"""QwenBuild – Standalone viewport multi-pass rendering & AO texture projection.

Self-contained Blender addon. No external dependencies required.
"""

import os
import math

import bpy  # type: ignore
import mathutils  # type: ignore
import numpy as np
from bpy.props import (  # type: ignore
    BoolProperty,
    IntProperty,
    StringProperty,
)

from . import camera_placement as cp


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Resolution helpers                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _get_camera_resolution(cam_obj, scene):
    """Return (res_x, res_y) for a camera.  Falls back to scene render
    resolution if no per-camera resolution is stored."""
    if "sg_res_x" in cam_obj and "sg_res_y" in cam_obj:
        return int(cam_obj["sg_res_x"]), int(cam_obj["sg_res_y"])
    return scene.render.resolution_x, scene.render.resolution_y


def adjust_camera_resolution_for_qwen(camera, scene):
    """Scale a camera's resolution to fit qwen-edit optimal ranges.

    Rules:
    - Minimum dimension: 960
    - Maximum dimension: 1664
    - Square aspect: 1344 × 1344
    - All values divisible by 64
    """
    ALIGN = 64
    MIN_DIM = 960
    MAX_DIM = 1664
    SQUARE = 1344

    res_x, res_y = _get_camera_resolution(camera, scene)

    if res_x == res_y:
        camera["sg_res_x"] = SQUARE
        camera["sg_res_y"] = SQUARE
        print(f"[QwenBuild]   {camera.name}: {res_x}x{res_y} → {SQUARE}x{SQUARE} (square)")
        return

    aspect = res_x / res_y

    if aspect >= 1.0:
        new_x = MAX_DIM
        new_y = int(round(new_x / aspect))
        if new_y < MIN_DIM:
            new_y = MIN_DIM
            new_x = int(round(new_y * aspect))
    else:
        new_y = MAX_DIM
        new_x = int(round(new_y * aspect))
        if new_x < MIN_DIM:
            new_x = MIN_DIM
            new_y = int(round(new_x / aspect))

    new_x = max(ALIGN, (new_x // ALIGN) * ALIGN)
    new_y = max(ALIGN, (new_y // ALIGN) * ALIGN)
    new_x = min(new_x, MAX_DIM)
    new_y = min(new_y, MAX_DIM)

    camera["sg_res_x"] = new_x
    camera["sg_res_y"] = new_y
    print(f"[QwenBuild]   {camera.name}: {res_x}x{res_y} → {new_x}x{new_y}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Viewport pass renderer                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def render_viewport_passes(context, camera, output_dir):
    """Capture Combined + AO passes from the viewport in Material Preview.
    Returns list of saved file paths.
    """
    cam_name = camera.name
    print(f"[QwenBuild] Rendering viewport passes for {cam_name}")
    os.makedirs(output_dir, exist_ok=True)

    context.scene.camera = camera
    bpy.context.view_layer.update()

    # ── Find a 3D viewport ──
    viewport_area = viewport_region = viewport_space = None
    viewport_region_3d = viewport_window = None

    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                viewport_area = area
                viewport_window = window
                for region in area.regions:
                    if region.type == 'WINDOW':
                        viewport_region = region
                        break
                viewport_space = area.spaces.active
                viewport_region_3d = viewport_space.region_3d if viewport_space else None
                break
        if viewport_area:
            break

    if not all([viewport_area, viewport_region, viewport_space,
                viewport_region_3d, viewport_window]):
        print("[QwenBuild] ERROR: no VIEW_3D area found.")
        return []

    # ── Save originals ──
    orig = {
        'view_perspective': viewport_region_3d.view_perspective,
        'shading_type': viewport_space.shading.type,
        'render_pass': viewport_space.shading.render_pass,
        'overlays': viewport_space.overlay.show_overlays,
        'gizmo': viewport_space.show_gizmo,
        'filepath': context.scene.render.filepath,
        'format': context.scene.render.image_settings.file_format,
        'res_x': context.scene.render.resolution_x,
        'res_y': context.scene.render.resolution_y,
    }

    # ── Apply per-camera resolution ──
    cam_rx, cam_ry = _get_camera_resolution(camera, context.scene)
    context.scene.render.resolution_x = cam_rx
    context.scene.render.resolution_y = cam_ry
    print(f"[QwenBuild]   Resolution: {cam_rx}x{cam_ry}")

    # ── Configure viewport ──
    viewport_region_3d.view_perspective = 'CAMERA'
    viewport_space.shading.type = 'MATERIAL'
    viewport_space.overlay.show_overlays = False
    viewport_space.show_gizmo = False
    context.scene.render.image_settings.file_format = 'PNG'

    override = {
        'window': viewport_window,
        'screen': viewport_window.screen,
        'area': viewport_area,
        'region': viewport_region,
        'scene': context.scene,
        'space_data': viewport_space,
        'region_data': viewport_region_3d,
    }

    saved_paths = []
    for pass_name in ('COMBINED', 'AO'):
        viewport_space.shading.render_pass = pass_name
        out_path = os.path.join(output_dir, f"{cam_name}_{pass_name.lower()}.png")
        context.scene.render.filepath = out_path
        with bpy.context.temp_override(**override):
            bpy.ops.render.opengl(write_still=True, view_context=True)
        saved_paths.append(out_path)
        print(f"[QwenBuild]   Saved {pass_name}: {out_path}")

    # ── Restore ──
    viewport_region_3d.view_perspective = orig['view_perspective']
    viewport_space.shading.type = orig['shading_type']
    viewport_space.shading.render_pass = orig['render_pass']
    viewport_space.overlay.show_overlays = orig['overlays']
    viewport_space.show_gizmo = orig['gizmo']
    context.scene.render.filepath = orig['filepath']
    context.scene.render.image_settings.file_format = orig['format']
    context.scene.render.resolution_x = orig['res_x']
    context.scene.render.resolution_y = orig['res_y']

    return saved_paths


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Operator 1:  Add Cameras                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝

class QWENBUILD_OT_add_cameras(bpy.types.Operator):
    """Place cameras using normal-weighted K-means with full BVH occlusion"""
    bl_idname = "qwenbuild.add_cameras"
    bl_label = "Add Cameras"
    bl_options = {'REGISTER', 'UNDO'}

    # Modal state
    _timer = None
    _occ_gen = None
    _occ_phase = False
    _occ_progress = 0.0
    _occ_state = None
    _cameras = []

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'MESH' for obj in context.scene.objects)

    def execute(self, context):
        scene = context.scene
        self._cameras = []

        # ── Delete existing cameras ──
        if scene.qb_delete_existing:
            to_remove = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
            for cam in to_remove:
                for col in list(cam.users_collection):
                    col.objects.unlink(cam)
                bpy.data.objects.remove(cam, do_unlink=True)
            for cam_data in list(bpy.data.cameras):
                if not cam_data.users:
                    bpy.data.cameras.remove(cam_data)
            scene.camera = None

        # ── Gather meshes ──
        target_meshes = cp.gather_target_meshes(context)
        if not target_meshes:
            self.report({'ERROR'}, "No mesh objects found")
            return {'CANCELLED'}

        total_faces = sum(len(o.data.polygons) for o in target_meshes)
        if total_faces == 0:
            self.report({'ERROR'}, "Target meshes have no faces")
            return {'CANCELLED'}

        # ── Get mesh data ──
        normals, areas, centers = cp.get_mesh_face_data(target_meshes)
        if scene.qb_exclude_bottom:
            normals, areas, centers = cp.filter_bottom_faces(
                normals, areas, centers, 1.5533)  # 89° in radians

        verts_world = cp.get_mesh_verts_world(target_meshes)
        mesh_center = verts_world.mean(axis=0)

        # ── Get or create camera settings ──
        temp_cam_data = None
        existing_cam = scene.camera
        if existing_cam and existing_cam.data:
            cam_settings = existing_cam.data
        else:
            temp_cam_data = bpy.data.cameras.new(name='_qb_temp_cam')
            cam_settings = temp_cam_data

        # ── Start BVH occlusion (full matrix, modal) ──
        depsgraph = context.evaluated_depsgraph_get()
        bvh_trees = cp.build_bvh_trees(target_meshes, depsgraph)
        self._occ_gen = cp.occ_filter_faces_generator(
            normals, areas, centers, bvh_trees)
        self._occ_phase = True
        self._occ_progress = 0.0
        self._occ_state = {
            'normals': normals,
            'areas': areas,
            'num_cameras': scene.qb_num_cameras,
            'verts_world': verts_world,
            'mesh_center': mesh_center,
            'cam_settings': cam_settings,
            'temp_cam_data': temp_cam_data,
        }

        self._timer = context.window_manager.event_timer_add(
            0.01, window=context.window)
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Computing occlusion matrix...")
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if not self._occ_phase:
            return {'PASS_THROUGH'}

        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._cleanup(context)
            self.report({'WARNING'}, "Occlusion computation cancelled")
            return {'CANCELLED'}

        if event.type == 'TIMER':
            try:
                progress = next(self._occ_gen)
                self._occ_progress = progress
                context.area.header_text_set(
                    f"QwenBuild: Computing occlusion… "
                    f"{progress * 100:.0f}%   (ESC to cancel)")
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
            except StopIteration as done:
                # Occlusion done — create cameras
                context.area.header_text_set(None)
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
                self._occ_gen = None
                self._occ_phase = False

                state = self._occ_state
                exterior = done.value
                normals = state['normals'][exterior]
                areas = state['areas'][exterior]
                n_removed = int((~exterior).sum())

                self.report({'INFO'},
                    f"Occlusion filter: removed {n_removed} interior faces, "
                    f"{len(normals)} remain")

                # K-means clustering
                k = min(state['num_cameras'], len(normals))
                if k > 0 and len(normals) > 0:
                    cluster_dirs = cp.kmeans_on_sphere(normals, areas, k)
                    directions = [cluster_dirs[i] for i in range(len(cluster_dirs))]
                else:
                    directions = []

                if not directions:
                    self.report({'WARNING'}, "No camera directions computed")
                    self._cleanup_state(context)
                    return {'CANCELLED'}

                # Sort directions spatially
                rv3d = context.region_data
                ref_dir = None
                if rv3d:
                    view_dir = rv3d.view_rotation @ mathutils.Vector((0, 0, -1))
                    ref_dir = np.array([-view_dir.x, -view_dir.y, -view_dir.z])
                directions = cp.sort_directions_spatially(directions, ref_dir)

                # Create cameras with per-camera aspect ratio
                self._create_cameras(
                    context, directions,
                    state['verts_world'], state['mesh_center'],
                    state['cam_settings'])

                # Clean up temp camera data
                if state.get('temp_cam_data'):
                    bpy.data.cameras.remove(state['temp_cam_data'])
                self._occ_state = None

                # Set first camera as active
                if self._cameras:
                    context.scene.camera = self._cameras[0]

                self.report({'INFO'},
                    f"Placed {len(self._cameras)} cameras (qwen-optimal res)")
                return {'FINISHED'}

            return {'RUNNING_MODAL'}

        return {'RUNNING_MODAL'}

    def _create_cameras(self, context, directions, verts_world, mesh_center,
                        cam_settings):
        """Create cameras with per-camera optimal aspect ratio and qwen
        resolution adjustment."""
        center_vec = mathutils.Vector(mesh_center.tolist())
        render = context.scene.render
        total_px = render.resolution_x * render.resolution_y
        center_for_aspect = verts_world.mean(axis=0)
        ALIGN = 64

        for i, d in enumerate(directions):
            d_np = np.array(d, dtype=float)
            d_unit = d_np / np.linalg.norm(d_np)

            # Pass 1: orthographic aspect → initial resolution → position
            aspect = cp.compute_per_camera_aspect(d_unit, verts_world, center_for_aspect)
            res_x, res_y = cp.resolution_from_aspect(aspect, total_px, align=ALIGN)
            fov_x, fov_y = cp.get_fov(cam_settings, res_x, res_y)
            dist, aim_off = cp.compute_silhouette_distance(
                verts_world, mesh_center, d_np, fov_x, fov_y)

            # Pass 2: refine with perspective aspect
            aim_point_np = mesh_center + aim_off
            cam_pos_np = aim_point_np + d_unit * dist
            aspect = cp.perspective_aspect(verts_world, cam_pos_np, d_np)
            res_x, res_y = cp.resolution_from_aspect(aspect, total_px, align=ALIGN)
            fov_x, fov_y = cp.get_fov(cam_settings, res_x, res_y)
            dist, aim_off = cp.compute_silhouette_distance(
                verts_world, mesh_center, d_np, fov_x, fov_y)

            dir_vec = mathutils.Vector(d_np.tolist()).normalized()
            aim_point = center_vec + mathutils.Vector(aim_off.tolist())
            pos = aim_point + dir_vec * dist

            cam_data = bpy.data.cameras.new(name=f'Camera_{i}')
            cam_obj = bpy.data.objects.new(f'Camera_{i}', cam_data)
            context.collection.objects.link(cam_obj)
            cam_obj.location = pos
            right, up_v, d_unit_cam = cp.camera_basis(d_np)
            cam_obj.rotation_euler = cp.rotation_from_basis(right, up_v, d_unit_cam)

            # Copy lens settings from reference
            cam_obj.data.type = cam_settings.type
            cam_obj.data.lens = cam_settings.lens
            cam_obj.data.sensor_width = cam_settings.sensor_width
            cam_obj.data.sensor_height = cam_settings.sensor_height
            cam_obj.data.clip_start = cam_settings.clip_start
            cam_obj.data.clip_end = cam_settings.clip_end

            # Store and adjust resolution
            cam_obj["sg_res_x"] = res_x
            cam_obj["sg_res_y"] = res_y
            adjust_camera_resolution_for_qwen(cam_obj, context.scene)
            res_x = int(cam_obj["sg_res_x"])
            res_y = int(cam_obj["sg_res_y"])

            # Display setup
            cam_obj.data.show_passepartout = True
            cam_obj.data.passepartout_alpha = 0.5

            self._cameras.append(cam_obj)

        # Set scene to square resolution for viewport display
        if self._cameras:
            max_side = max(
                max(int(c.get('sg_res_x', 0)), int(c.get('sg_res_y', 0)))
                for c in self._cameras if 'sg_res_x' in c
            )
            if max_side > 0:
                context.scene.render.resolution_x = max_side
                context.scene.render.resolution_y = max_side

    def _cleanup(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        self._occ_gen = None
        self._occ_phase = False
        context.area.header_text_set(None)
        if self._occ_state and self._occ_state.get('temp_cam_data'):
            bpy.data.cameras.remove(self._occ_state['temp_cam_data'])
        self._occ_state = None

    def cancel(self, context):
        self._cleanup(context)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Operator 1b: Adjust Resolutions (standalone)                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

class QWENBUILD_OT_adjust_resolutions(bpy.types.Operator):
    """Adjust all camera resolutions to qwen-edit optimal range (960-1664, ÷64)"""
    bl_idname = "qwenbuild.adjust_resolutions"
    bl_label = "Adjust Resolutions"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return any(obj.type == 'CAMERA' and obj.name.startswith('Camera_')
                   for obj in context.scene.objects)

    def execute(self, context):
        cameras = sorted(
            [obj for obj in context.scene.objects
             if obj.type == 'CAMERA' and obj.name.startswith('Camera_')],
            key=lambda c: c.name
        )
        for cam in cameras:
            adjust_camera_resolution_for_qwen(cam, context.scene)
        self.report({'INFO'}, f"Adjusted {len(cameras)} cameras to qwen-optimal res")
        return {'FINISHED'}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Operator 2:  Render Viewport Passes                                ║
# ╚══════════════════════════════════════════════════════════════════════╝

class QWENBUILD_OT_render_passes(bpy.types.Operator):
    """Render Combined + AO passes from all cameras"""
    bl_idname = "qwenbuild.render_passes"
    bl_label = "Render Passes"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        cameras = [obj for obj in context.scene.objects
                   if obj.type == 'CAMERA' and obj.name.startswith('Camera_')]
        return len(cameras) > 0 and context.scene.qb_output_dir != ""

    def execute(self, context):
        scene = context.scene
        output_dir = bpy.path.abspath(scene.qb_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        cameras = sorted(
            [obj for obj in scene.objects
             if obj.type == 'CAMERA' and obj.name.startswith('Camera_')],
            key=lambda c: c.name
        )

        all_paths = []
        export_data = {
            "source": "qwenbuild_blender",
            "cameras": []
        }

        for i, cam in enumerate(cameras):
            print(f"[QwenBuild] Camera {i+1}/{len(cameras)}: {cam.name}")
            paths = render_viewport_passes(context, cam, output_dir)
            all_paths.extend(paths)
            if len(paths) >= 2:
                export_data["cameras"].append({
                    "name": cam.name,
                    "combined": paths[0],
                    "ao": paths[1]
                })

        import json
        export_file = os.path.join(output_dir, "qwen_export.json")
        with open(export_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)

        self.report({'INFO'}, f"Rendered {len(all_paths)} passes to {output_dir}")
        return {'FINISHED'}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Operator 3:  Create Material from AO                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

class QWENBUILD_OT_create_material(bpy.types.Operator):
    """Create a material using AO passes as texture"""
    bl_idname = "qwenbuild.create_material"
    bl_label = "Create Material from AO"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        cameras = [obj for obj in context.scene.objects
                   if obj.type == 'CAMERA' and obj.name.startswith('Camera_')]
        has_meshes = any(obj.type == 'MESH' for obj in context.scene.objects)
        output_dir = bpy.path.abspath(context.scene.qb_output_dir)
        has_ao = False
        if output_dir and os.path.isdir(output_dir):
            has_ao = any(f.endswith('_ao.png') for f in os.listdir(output_dir))
        return len(cameras) > 0 and has_meshes and has_ao

    def execute(self, context):
        scene = context.scene
        output_dir = bpy.path.abspath(scene.qb_output_dir)

        cameras = sorted(
            [obj for obj in scene.objects
             if obj.type == 'CAMERA' and obj.name.startswith('Camera_')],
            key=lambda c: c.name
        )

        mesh_objs = [obj for obj in scene.objects if obj.type == 'MESH']
        if not mesh_objs:
            self.report({'ERROR'}, "No mesh objects found")
            return {'CANCELLED'}

        # Load AO images
        ao_images = []
        for i, cam in enumerate(cameras):
            ao_path = os.path.join(output_dir, f"{cam.name}_ao.png")
            if os.path.exists(ao_path):
                img_name = f"QB_AO_{cam.name}"
                if img_name in bpy.data.images:
                    bpy.data.images.remove(bpy.data.images[img_name])
                img = bpy.data.images.load(ao_path)
                img.name = img_name
                ao_images.append((i, cam, img))
                print(f"[QwenBuild] Loaded AO {ao_path}")
            else:
                print(f"[QwenBuild] WARNING: missing AO for {cam.name}: {ao_path}")

        if not ao_images:
            self.report({'ERROR'}, "No AO images found in output directory")
            return {'CANCELLED'}

        # Delete old materials to prevent buildup
        for m in list(bpy.data.materials):
            if m.name.startswith("QwenBuild_AO"):
                bpy.data.materials.remove(m)

        from .sg_core import project as sg_project
        from .sg_core import utils as sg_utils

        orig_get_file_path = sg_project.get_file_path
        orig_get_dir_path = sg_project.get_dir_path

        def mock_get_file_path(ctx, category, *args, **kwargs):
            if category == "generated":
                return ""  # Return empty path → placeholder fallback
            try:
                return orig_get_file_path(ctx, category, *args, **kwargs)
            except Exception:
                return ""

        def mock_get_dir_path(ctx, category, *args, **kwargs):
            if category == "inpaint":
                return ""  # disable edge feather loading
            try:
                return orig_get_dir_path(ctx, category, *args, **kwargs)
            except Exception:
                return ""

        sg_project.get_file_path = mock_get_file_path
        sg_utils.get_file_path = mock_get_file_path
        sg_project.get_dir_path = mock_get_dir_path
        sg_utils.get_dir_path = mock_get_dir_path

        mat_id = "QB_AO"
        try:
            sg_project.project_image(context, mesh_objs, mat_id=mat_id, stop_index=len(cameras)-1)
        except Exception as e:
            self.report({'ERROR'}, f"Failed to run projection: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            sg_project.get_file_path = orig_get_file_path
            sg_utils.get_file_path = orig_get_file_path
            sg_project.get_dir_path = orig_get_dir_path
            sg_utils.get_dir_path = orig_get_dir_path

        # Now link images into the generated tree
        for obj in mesh_objs:
            mat = obj.active_material
            if not mat or not mat.use_nodes: continue

            mat.name = "QwenBuild_AO"

            for node in mat.node_tree.nodes:
                if node.type == 'TEX_IMAGE' and node.label and node.label.endswith(f"-{mat_id}"):
                    try:
                        cam_idx = int(node.label.split('-')[0])
                        matching_ao = next((img for idx, c, img in ao_images if cameras.index(c) == cam_idx), None)
                        if matching_ao:
                            node.image = matching_ao
                    except ValueError:
                        pass

        self.report({'INFO'}, f"Created projection material from AO passes")
        return {'FINISHED'}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  UI Panel                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

class QWENBUILD_PT_main(bpy.types.Panel):
    """QwenBuild – viewport multi-pass pipeline"""
    bl_label = "QwenBuild"
    bl_idname = "VIEW3D_PT_qwenbuild"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "QwenBuild"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        cameras = [obj for obj in scene.objects
                   if obj.type == 'CAMERA' and obj.name.startswith('Camera_')]

        # ── Step 1: Cameras ──
        box = layout.box()
        row = box.row()
        row.label(text="Step 1: Cameras", icon="CAMERA_DATA")

        row = box.row()
        row.prop(scene, "qb_num_cameras", text="Number")
        row = box.row()
        row.prop(scene, "qb_exclude_bottom", text="Exclude Bottom Faces")
        row = box.row()
        row.prop(scene, "qb_delete_existing", text="Remove Existing Cameras")

        row = box.row()
        row.scale_y = 1.3
        row.operator("qwenbuild.add_cameras", icon="ADD")

        if cameras:
            info = box.row()
            info.label(text=f"{len(cameras)} cameras placed", icon="CHECKMARK")
            for cam in sorted(cameras, key=lambda c: c.name):
                if "sg_res_x" in cam:
                    r = box.row()
                    r.label(
                        text=f"  {cam.name}: {int(cam['sg_res_x'])}x{int(cam['sg_res_y'])}",
                        icon="IMAGE_DATA",
                    )
            row = box.row()
            row.operator("qwenbuild.adjust_resolutions", icon="MODIFIER")

        # ── Step 2: Render Passes ──
        box = layout.box()
        row = box.row()
        row.label(text="Step 2: Render Passes", icon="RENDER_STILL")

        row = box.row()
        row.prop(scene, "qb_output_dir", text="Output")

        row = box.row()
        row.scale_y = 1.3
        row.operator("qwenbuild.render_passes", icon="PLAY")
        row.enabled = len(cameras) > 0 and scene.qb_output_dir != ""

        output_dir = bpy.path.abspath(scene.qb_output_dir)
        if output_dir and os.path.isdir(output_dir):
            ao_files = [f for f in os.listdir(output_dir) if f.endswith('_ao.png')]
            combined_files = [f for f in os.listdir(output_dir) if f.endswith('_combined.png')]
            if ao_files:
                info = box.row()
                info.label(
                    text=f"{len(combined_files)} combined + {len(ao_files)} AO renders",
                    icon="CHECKMARK",
                )

        # ── Step 3: Create Material ──
        box = layout.box()
        row = box.row()
        row.label(text="Step 3: Material", icon="MATERIAL")

        row = box.row()
        row.scale_y = 1.3
        row.operator("qwenbuild.create_material", icon="NODE_MATERIAL")

        if "QwenBuild_AO" in bpy.data.materials:
            info = box.row()
            info.label(text="Material 'QwenBuild_AO' active", icon="CHECKMARK")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Registration                                                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

_classes = (
    QWENBUILD_OT_add_cameras,
    QWENBUILD_OT_adjust_resolutions,
    QWENBUILD_OT_render_passes,
    QWENBUILD_OT_create_material,
    QWENBUILD_PT_main,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.qb_num_cameras = IntProperty(
        name="Number of Cameras",
        description="Number of cameras to place around the object",
        default=7, min=1, max=20,
    )
    bpy.types.Scene.qb_exclude_bottom = BoolProperty(
        name="Exclude Bottom",
        description="Ignore downward-facing geometry when placing cameras",
        default=True,
    )
    bpy.types.Scene.qb_delete_existing = BoolProperty(
        name="Remove Existing",
        description="Delete all existing cameras before adding new ones",
        default=True,
    )
    bpy.types.Scene.qb_output_dir = StringProperty(
        name="Output Directory",
        description="Directory to save rendered viewport passes",
        subtype='DIR_PATH',
        default="//qwenbuild_renders/",
    )


def unregister():
    del bpy.types.Scene.qb_output_dir
    del bpy.types.Scene.qb_delete_existing
    del bpy.types.Scene.qb_exclude_bottom
    del bpy.types.Scene.qb_num_cameras

    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)

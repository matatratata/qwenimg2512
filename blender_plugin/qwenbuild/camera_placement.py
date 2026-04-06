# SPDX-License-Identifier: GPL-3.0-or-later
"""Camera placement – extracted from StableGen with minimal dependencies.

Provides normal-weighted K-means camera placement with BVH occlusion
filtering, per-camera aspect ratio, and silhouette-optimal distance.
"""

import math

import bpy  # type: ignore
import bmesh  # type: ignore
import mathutils  # type: ignore
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Mesh helpers
# ──────────────────────────────────────────────────────────────────────

def gather_target_meshes(context):
    """Return mesh objects to cover.  Uses selected meshes or all in scene."""
    selected = [o for o in context.selected_objects if o.type == 'MESH']
    return selected if selected else [o for o in context.scene.objects if o.type == 'MESH']


def get_mesh_face_data(objs):
    """Return world-space face (normals, areas, centers) as numpy arrays."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    all_normals, all_areas, all_centers = [], [], []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        rot = mat.to_3x3()
        scale_det = abs(rot.determinant())
        n = len(mesh.polygons)
        normals = np.empty((n, 3))
        areas = np.empty(n)
        centers = np.empty((n, 3))
        for idx, poly in enumerate(mesh.polygons):
            wn = rot @ poly.normal
            wn.normalize()
            normals[idx] = (wn.x, wn.y, wn.z)
            wc = mat @ poly.center
            centers[idx] = (wc.x, wc.y, wc.z)
            areas[idx] = poly.area * (scale_det ** 0.5)
        all_normals.append(normals)
        all_areas.append(areas)
        all_centers.append(centers)
    return np.vstack(all_normals), np.concatenate(all_areas), np.vstack(all_centers)


def get_mesh_verts_world(objs):
    """Return world-space vertex positions as (N, 3) numpy array."""
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    parts = []
    for obj in objs:
        mesh = obj.data
        mat = obj.matrix_world
        n = len(mesh.vertices)
        verts = np.empty((n, 3))
        for i, v in enumerate(mesh.vertices):
            wv = mat @ v.co
            verts[i] = (wv.x, wv.y, wv.z)
        parts.append(verts)
    return np.vstack(parts)


def filter_bottom_faces(normals, areas, centers, angle_rad):
    """Remove faces whose normals point more than *angle_rad* below horizontal."""
    threshold_z = -math.cos(angle_rad)
    mask = normals[:, 2] >= threshold_z
    return normals[mask], areas[mask], centers[mask]


# ──────────────────────────────────────────────────────────────────────
# Camera geometry
# ──────────────────────────────────────────────────────────────────────

def camera_basis(cam_dir_np):
    """Build orthonormal camera basis from centre-to-camera direction.
    Returns (right, up, d_unit)."""
    d = cam_dir_np / np.linalg.norm(cam_dir_np)
    forward = -d
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, world_up)) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up, d


def rotation_from_basis(right, up, d_unit):
    """Build Blender rotation (Euler) from camera basis vectors."""
    rot_mat = mathutils.Matrix((
        (right[0], up[0], d_unit[0]),
        (right[1], up[1], d_unit[1]),
        (right[2], up[2], d_unit[2]),
    ))
    return rot_mat.to_euler()


def get_fov(cam_settings, res_x, res_y):
    """Return (fov_x, fov_y) in radians."""
    fov_x = cam_settings.angle_x
    if res_y > res_x:
        fov_x = 2 * math.atan(math.tan(fov_x / 2) * res_x / res_y)
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * res_y / res_x)
    return fov_x, fov_y


def compute_silhouette_distance(verts_world, center_np, cam_dir_np,
                                fov_x, fov_y, margin=0.10):
    """Compute optimal distance so every mesh vertex fits in the camera frame.
    Returns (distance, aim_offset)."""
    right, up, d = camera_basis(cam_dir_np)

    rel = verts_world - center_np
    proj_r = rel @ right
    proj_u = rel @ up

    r_min, r_max = float(proj_r.min()), float(proj_r.max())
    u_min, u_max = float(proj_u.min()), float(proj_u.max())

    mid_r = (r_max + r_min) / 2.0
    mid_u = (u_max + u_min) / 2.0
    aim_offset = mid_r * right + mid_u * up

    eff_fov_x = fov_x * (1.0 - margin)
    eff_fov_y = fov_y * (1.0 - margin)
    tan_hx = math.tan(eff_fov_x / 2) if eff_fov_x > 0.02 else 1e-6
    tan_hy = math.tan(eff_fov_y / 2) if eff_fov_y > 0.02 else 1e-6

    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    pd = rel_aim @ d

    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd
    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    # ── Refine aim using perspective angular centre ──
    cam_pos = aim_point + d * dist
    rel_cam = verts_world - cam_pos
    depth = -(rel_cam @ d)
    depth = np.maximum(depth, 0.001)
    ang_r = np.arctan2(rel_cam @ right, depth)
    ang_u = np.arctan2(rel_cam @ up, depth)
    ang_mid_r = (float(ang_r.max()) + float(ang_r.min())) / 2.0
    ang_mid_u = (float(ang_u.max()) + float(ang_u.min())) / 2.0
    aim_offset = aim_offset + dist * math.tan(ang_mid_r) * right + dist * math.tan(ang_mid_u) * up

    # Recompute distance with refined aim
    aim_point = center_np + aim_offset
    rel_aim = verts_world - aim_point
    pr = rel_aim @ right
    pu = rel_aim @ up
    pd = rel_aim @ d
    min_dist_r = np.abs(pr) / tan_hx + pd
    min_dist_u = np.abs(pu) / tan_hy + pd
    dist = max(float(min_dist_r.max()), float(min_dist_u.max()), 0.5)

    return dist, aim_offset


# ──────────────────────────────────────────────────────────────────────
# Aspect ratio helpers
# ──────────────────────────────────────────────────────────────────────

def compute_per_camera_aspect(direction_np, verts_world, center):
    """Compute the silhouette aspect ratio (width / height)."""
    right, up, _d = camera_basis(direction_np)
    rel = verts_world - center
    proj_r = rel @ right
    proj_u = rel @ up
    w = max(float(proj_r.max() - proj_r.min()), 0.001)
    h = max(float(proj_u.max() - proj_u.min()), 0.001)
    return w / h


def perspective_aspect(verts_world, cam_pos_np, cam_dir_np):
    """Compute the perspective angular aspect ratio (width / height)."""
    right, up, d = camera_basis(cam_dir_np)
    forward = -d
    rel = verts_world - cam_pos_np
    depth = rel @ forward
    depth = np.maximum(depth, 0.001)
    pr = rel @ right
    pu = rel @ up
    angle_r = np.arctan2(pr, depth)
    angle_u = np.arctan2(pu, depth)
    angular_w = float(angle_r.max() - angle_r.min())
    angular_h = float(angle_u.max() - angle_u.min())
    if angular_h < 0.001:
        return 1.0
    return angular_w / angular_h


def resolution_from_aspect(aspect, total_px, align=64):
    """Compute (res_x, res_y) for a given aspect ratio, snapped to alignment."""
    new_x = math.sqrt(total_px * aspect)
    new_y = total_px / new_x
    new_x = max(align, int(round(new_x / align)) * align)
    new_y = max(align, int(round(new_y / align)) * align)
    return new_x, new_y


# ──────────────────────────────────────────────────────────────────────
# K-means on sphere
# ──────────────────────────────────────────────────────────────────────

def kmeans_on_sphere(directions, weights, k, max_iter=50):
    """Spherical K-means: cluster unit vectors weighted by area.
    Returns (k, 3) numpy array of cluster-centre unit vectors."""
    n_pts = len(directions)
    if n_pts == 0 or k == 0:
        return np.zeros((max(k, 1), 3))
    k = min(k, n_pts)
    rng = np.random.default_rng(42)
    probs = weights / weights.sum()
    indices = rng.choice(n_pts, size=k, replace=False, p=probs)
    centers = directions[indices].copy()
    for _ in range(max_iter):
        dots = directions @ centers.T
        labels = np.argmax(dots, axis=1)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                ws = (directions[mask] * weights[mask, np.newaxis]).sum(axis=0)
                nrm = np.linalg.norm(ws)
                new_centers[j] = ws / nrm if nrm > 0 else centers[j]
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers, atol=1e-6):
            break
        centers = new_centers
    return centers


# ──────────────────────────────────────────────────────────────────────
# Direction sorting
# ──────────────────────────────────────────────────────────────────────

def sort_directions_spatially(directions, ref_direction=None):
    """Sort direction vectors by azimuth so cameras progress smoothly."""
    if len(directions) <= 1:
        return directions
    angles = [math.atan2(float(d[1]), float(d[0])) for d in directions]
    paired = sorted(zip(angles, directions), key=lambda p: p[0])

    if ref_direction is not None:
        ref_v = mathutils.Vector(ref_direction).normalized()
        best_idx = 0
        best_dot = -2.0
        for idx, (_, d) in enumerate(paired):
            dot = mathutils.Vector(d).normalized().dot(ref_v)
            if dot > best_dot:
                best_dot = dot
                best_idx = idx
        paired = paired[best_idx:] + paired[:best_idx]

    return [d for _, d in paired]


# ──────────────────────────────────────────────────────────────────────
# BVH occlusion
# ──────────────────────────────────────────────────────────────────────

def fibonacci_sphere_points(n):
    """Generate *n* approximately evenly-spaced unit vectors on a sphere."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n):
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = 2 * math.pi * i / golden_ratio
        x = math.sin(theta) * math.cos(phi)
        y = math.sin(theta) * math.sin(phi)
        z = math.cos(theta)
        points.append((x, y, z))
    return points


def build_bvh_trees(objs, depsgraph):
    """Build a list of BVHTree objects (one per mesh) for raycasting."""
    from mathutils.bvhtree import BVHTree
    trees = []
    for obj in objs:
        bm = bmesh.new()
        bm.from_object(obj, depsgraph)
        bm.transform(obj.matrix_world)
        tree = BVHTree.FromBMesh(bm)
        bm.free()
        trees.append(tree)
    return trees


def _ray_occluded(bvh_trees, origin, direction, max_dist):
    """Return True if any BVH tree has a hit between origin and max_dist."""
    for tree in bvh_trees:
        hit_loc, _normal, _index, _dist = tree.ray_cast(
            mathutils.Vector(origin), mathutils.Vector(direction), max_dist)
        if hit_loc is not None:
            return True
    return False


def occ_filter_faces_generator(normals, areas, centers, bvh_trees,
                               n_candidates=200):
    """Generator: determine which faces are visible from at least one
    direction.  Yields progress [0,1].  Final result via StopIteration.value
    is a boolean exterior mask."""
    n_faces = len(normals)
    candidates = np.array(fibonacci_sphere_points(n_candidates))
    backface_vis = normals @ candidates.T > 0.26
    exterior = np.zeros(n_faces, dtype=bool)
    epsilon = 0.001
    BATCH = 5

    for j in range(n_candidates):
        cam_dir = candidates[j]
        for i in range(n_faces):
            if exterior[i]:
                continue
            if not backface_vis[i, j]:
                continue
            origin = centers[i] + normals[i] * epsilon
            if not _ray_occluded(bvh_trees, origin, cam_dir, 1e6):
                exterior[i] = True
        if (j + 1) % BATCH == 0 or j == n_candidates - 1:
            yield (j + 1) / n_candidates
            if exterior.all():
                break

    return exterior

"""Mesh curvature and tensor-field computation (LaplacianPLY-compatible)."""

from __future__ import annotations

import numpy as np


def load_mesh(path: str):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a triangle mesh PLY file.")
    mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    else:
        mesh.update_faces(mesh.nondegenerate_faces())
    # Drop triangles that reuse the same vertex (matches learnply.cpp).
    faces = mesh.faces
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    mesh.update_faces(keep)
    return mesh


def _one_ring_neighbors(faces: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Corner-based one-ring neighbors, matching LaplacianPLY corner->next->vertex."""
    neighbors: list[set[int]] = [set() for _ in range(n_vertices)]
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    return [np.fromiter(sorted(n), dtype=np.int64) for n in neighbors]


def _calc_vertex_local_frame(
    vi: np.ndarray, normal: np.ndarray, neighbor_indices: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Same idea as MeshProcessor::calcVertLocalframe in meshprocessor.cpp."""
    n = normal / (np.linalg.norm(normal) + 1e-12)
    min_proj = np.inf
    e1 = np.array([1.0, 0.0, 0.0])

    for idx in neighbor_indices:
        edge = vertices[idx] - vi
        proj = abs(float(np.dot(edge, n)))
        if proj < min_proj:
            min_proj = proj
            tangent = edge - np.dot(edge, n) * n
            norm = np.linalg.norm(tangent)
            if norm > 1e-12:
                e1 = tangent / norm

    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2) + 1e-12
    e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1) + 1e-12
    return e1, e2


def _cotangent_laplacian(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = len(vertices)
    lap = np.zeros((n, 3), dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)

    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            ea = vb - vc
            eb = va - vc
            cos_angle = np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-12)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            w = 0.5 / max(np.tan(np.arccos(cos_angle)), 1e-6)
            lap[a] += w * (vertices[b] - vertices[a])
            weights[a] += w

    mask = weights > 1e-12
    lap[mask] /= weights[mask, None]
    return lap


def _fit_curvature_tensor(
    vertices: np.ndarray,
    center_idx: int,
    normal: np.ndarray,
    neighbors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares tensor fit (MeshProcessor::calcCurvatureTensor)."""
    if len(neighbors) < 4:
        return (
            np.zeros(2),
            np.zeros(2),
            np.eye(2),
            np.zeros((2, 3)),
        )

    vi = vertices[center_idx]
    n = normal / (np.linalg.norm(normal) + 1e-12)
    e1, e2 = _calc_vertex_local_frame(vi, n, neighbors, vertices)

    rows = []
    heights = []
    for idx in neighbors:
        d = vertices[idx] - vi
        u = float(np.dot(d, e1))
        v = float(np.dot(d, e2))
        h = float(np.dot(d, n))
        rows.append((0.5 * u * u, u * v, 0.5 * v * v))
        heights.append(h)

    mat_a = np.array(rows, dtype=np.float64)
    vec_k = np.array(heights, dtype=np.float64)
    try:
        coeff, _, _, _ = np.linalg.lstsq(mat_a, vec_k, rcond=None)
    except np.linalg.LinAlgError:
        return np.zeros(2), np.zeros(2), np.eye(2), np.array([e1, e2])

    hessian = np.array([[coeff[0], coeff[1]], [coeff[1], coeff[2]]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(hessian)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    k_max, k_min = float(evals[0]), float(evals[1])
    d_max = e1 * evecs[0, 0] + e2 * evecs[1, 0]
    d_min = e1 * evecs[0, 1] + e2 * evecs[1, 1]
    d_max /= np.linalg.norm(d_max) + 1e-12
    d_min /= np.linalg.norm(d_min) + 1e-12
    return np.array([k_max, k_min]), np.array([d_max, d_min]), hessian, np.array([e1, e2])


def _tangent(v: np.ndarray, n: np.ndarray) -> np.ndarray | None:
    t = v - np.dot(v, n) * n
    norm = float(np.linalg.norm(t))
    if norm < 1e-12:
        return None
    return t / norm


def _neighbor_order_ccw(
    center_idx: int, vertices: np.ndarray, normals: np.ndarray, neighbors: list[np.ndarray]
) -> list[int]:
    vi = vertices[center_idx]
    n = normals[center_idx]
    ref = _tangent(np.array([1.0, 0.0, 0.0]), n)
    if ref is None:
        ref = _tangent(np.array([0.0, 1.0, 0.0]), n)
    if ref is None:
        return []
    perp = np.cross(n, ref)
    ordered: list[tuple[float, int]] = []
    for j in neighbors[center_idx]:
        edge = vertices[j] - vi
        edge = edge - np.dot(edge, n) * n
        if float(np.linalg.norm(edge)) < 1e-12:
            continue
        angle = float(np.arctan2(np.dot(edge, perp), np.dot(edge, ref)))
        ordered.append((angle, int(j)))
    ordered.sort(key=lambda item: item[0])
    return [j for _, j in ordered]


def _line_field_index(
    center_idx: int,
    vertices: np.ndarray,
    normals: np.ndarray,
    princ_dir: np.ndarray,
    neighbors: list[np.ndarray],
) -> float:
    n = normals[center_idx]
    order = _neighbor_order_ccw(center_idx, vertices, normals, neighbors)
    dirs: list[np.ndarray] = []
    for j in order:
        d = _tangent(princ_dir[j], n)
        if d is not None:
            dirs.append(d)
    if len(dirs) < 3:
        return 0.0

    total = 0.0
    for k in range(len(dirs)):
        a = dirs[k]
        b = dirs[(k + 1) % len(dirs)]
        sin_t = float(np.dot(np.cross(a, b), n))
        cos_t = float(np.dot(a, b))
        if cos_t < 0.0:
            sin_t = -sin_t
            cos_t = -cos_t
        total += float(np.arctan2(sin_t, cos_t))
    return total / np.pi


def _detect_tensor_singularities(
    vertices: np.ndarray,
    normals: np.ndarray,
    princ_dir_max: np.ndarray,
    neighbors: list[np.ndarray],
    max_markers: int = 48,
) -> list[dict]:
    scored: list[tuple[float, int, str]] = []
    for i in range(len(vertices)):
        index = _line_field_index(i, vertices, normals, princ_dir_max, neighbors)
        if index > 0.28:
            scored.append((index, i, "positive"))
        elif index < -0.28:
            scored.append((abs(index), i, "negative"))

    scored.sort(key=lambda item: item[0], reverse=True)
    singularities: list[dict] = []
    for _, idx, kind in scored[:max_markers]:
        singularities.append(
            {
                "position": vertices[idx].tolist(),
                "type": kind,
            }
        )
    return singularities


def compute_curvature_and_tensor(mesh, color_max: float = 20.0) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Face normals (MeshProcessor::calcFaceNormalsAndArea style).
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v2 - v0, v1 - v0)
    face_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(face_norms, 1e-12)

    # Vertex normals: sum of incident face normals (MeshProcessor::calcVertNormals).
    normals = np.zeros_like(vertices)
    for j in range(3):
        np.add.at(normals, faces[:, j], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-12)

    neighbors = _one_ring_neighbors(faces, len(vertices))
    max_princ = np.zeros(len(vertices))
    min_princ = np.zeros(len(vertices))
    princ_dir_max = np.zeros((len(vertices), 3))
    princ_dir_min = np.zeros((len(vertices), 3))

    for idx in range(len(vertices)):
        ks, dirs, _, _ = _fit_curvature_tensor(vertices, idx, normals[idx], neighbors[idx])
        max_princ[idx] = ks[0]
        min_princ[idx] = ks[1]
        princ_dir_max[idx] = dirs[0]
        princ_dir_min[idx] = dirs[1]

    # Mean curvature H = (κ₁ + κ₂) / 2 from the fitted shape operator.
    mean_curvature = 0.5 * (max_princ + min_princ)

    singularities = _detect_tensor_singularities(vertices, normals, princ_dir_max, neighbors)
    mesh_scale = float(max(mesh.extents.max(), 1e-6))

    return {
        "vertices": vertices.reshape(-1).tolist(),
        "faces": faces.reshape(-1).tolist(),
        "normals": normals.reshape(-1).tolist(),
        "mean_curvature": mean_curvature.tolist(),
        "max_princ_curvature": max_princ.tolist(),
        "min_princ_curvature": min_princ.tolist(),
        "princ_dir_max": princ_dir_max.reshape(-1).tolist(),
        "princ_dir_min": princ_dir_min.reshape(-1).tolist(),
        "tensor_singularities": singularities,
        "mesh_scale": mesh_scale,
        "curvature_range": [-float(color_max), float(color_max)],
        "color_max": float(color_max),
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
    }

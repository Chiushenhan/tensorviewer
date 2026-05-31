"""Flask server for Assignment4 tensor-field split viewer."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_compress import Compress

from processing.geometry import compute_curvature_and_tensor, load_mesh

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
CACHE_DIR = ROOT / "data" / "cache"


def _resolve_samples_dir() -> Path:
    for candidate in (ROOT / "model3d", ROOT.parent / "model3d"):
        if candidate.is_dir():
            return candidate
    return ROOT / "model3d"


SAMPLES_DIR = _resolve_samples_dir()

SAMPLE_LABELS = {
    "bunny1.ply": "Bunny 1",
    "dragon.ply": "Dragon",
    "feline.ply": "Feline",
    "happy.ply": "Happy Buddha",
    "sphere.ply": "Sphere",
    "torus.ply": "Torus",
}

DEFAULT_MESH = "bunny1.ply"
COLOR_MAX = 20.0
EXCLUDED_MESHES = {"icosahedron.ply"}

_mesh_cache: dict[tuple[str, float], dict] = {}


def _load_cached_payload(stem: str) -> dict | None:
    cache_file = CACHE_DIR / f"{stem}.json"
    if not cache_file.exists():
        return None
    return json.loads(cache_file.read_text(encoding="utf-8"))


def mesh_payload(path: Path, color_max: float = COLOR_MAX) -> dict:
    key = (str(path.resolve()), path.stat().st_mtime)
    cached = _mesh_cache.get(key)
    if cached is not None:
        return cached

    disk_cached = _load_cached_payload(path.stem)
    if disk_cached is not None:
        _mesh_cache[key] = disk_cached
        return disk_cached

    mesh = load_mesh(str(path))
    payload = compute_curvature_and_tensor(mesh, color_max=color_max)
    _mesh_cache[key] = payload
    return payload


def list_samples() -> list[dict]:
    if not SAMPLES_DIR.exists():
        return []
    items = []
    for path in sorted(SAMPLES_DIR.glob("*.ply")):
        if path.name in EXCLUDED_MESHES:
            continue
        items.append(
            {
                "id": path.name,
                "label": SAMPLE_LABELS.get(path.name, path.stem),
                "size_kb": path.stat().st_size // 1024,
            }
        )
    return items


app = Flask(__name__, static_folder=str(STATIC), static_url_path="")
Compress(app)


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/favicon.ico")
def favicon():
    return ("", 204)


@app.get("/api/health")
def health():
    samples = list_samples()
    return jsonify(
        {
            "ok": True,
            "samples_dir": str(SAMPLES_DIR),
            "sample_count": len(samples),
            "cache_dir": str(CACHE_DIR),
            "cache_count": len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0,
        }
    )


@app.get("/api/samples")
def samples_list():
    return jsonify(list_samples())


@app.get("/api/sample")
@app.get("/api/sample/<name>")
def sample_mesh(name: str | None = None):
    if name is None:
        name = DEFAULT_MESH
    safe_name = Path(name).name
    stem = Path(safe_name).stem

    cache_file = CACHE_DIR / f"{stem}.json"
    if cache_file.exists():
        return send_file(cache_file, mimetype="application/json")

    path = SAMPLES_DIR / safe_name
    if not path.exists():
        return jsonify({"error": f"Sample '{safe_name}' not found in model3d/"}), 404
    payload = mesh_payload(path)
    payload["filename"] = safe_name
    return jsonify(payload)


@app.post("/api/upload")
def upload_mesh():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"error": "No file uploaded."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".ply":
        return jsonify({"error": "Use ASCII/binary PLY files (same format as model3d/)."}), 400

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        mesh = load_mesh(tmp_path)
        payload = compute_curvature_and_tensor(mesh, color_max=COLOR_MAX)
        payload["filename"] = file.filename
        return jsonify(payload)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import os

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print("Assignment4 Tensor Field Viewer")
    print(f"Meshes: {SAMPLES_DIR}")
    print(f"Open http://127.0.0.1:{port}  (LAN: http://<your-ip>:{port})")
    app.run(host=host, port=port, debug=False, threaded=True)

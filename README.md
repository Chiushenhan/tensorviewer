# TensorViewer

From **3D curvature** (left) to **cross-field tensor LIC** (right) on the same mesh.

**Live site:** https://tensorviewer.onrender.com

App code lives in [`TensorViewer/`](TensorViewer/) — Render uses that folder as root directory.

## Run locally

```powershell
cd TensorViewer
py -m pip install -r requirements.txt
py app.py
```

Open http://127.0.0.1:5000

## Deploy

Connect this repo on [Render](https://render.com). It reads `render.yaml` at the repo root with `rootDir: TensorViewer`.

## Controls

- **Mesh dropdown** — bunny, dragon, torus, etc.
- **Curvature mode** — mean / max / min principal (left pane)
- **Drag center bar** — compare left vs right
- **Upload PLY** — load your own mesh

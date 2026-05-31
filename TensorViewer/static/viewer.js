import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { createCrossFieldLicMaterial, disposeCrossFieldLicMaterial } from "./crossfield-lic.js";

const canvas = document.getElementById("canvas");
const compareSlider = document.getElementById("compare-slider");
const fileInput = document.getElementById("file-input");
const sampleSelect = document.getElementById("sample-select");
const curvatureMode = document.getElementById("curvature-mode");
const statusEl = document.getElementById("status");
const statusText = statusEl?.querySelector(".status-text");

const BG_COLOR = 0xffffff;
const SURFACE_COLOR = 0xf3efe6;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(BG_COLOR, 1);

const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
camera.position.set(2.4, 1.6, 2.8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

const scene = new THREE.Scene();
scene.add(new THREE.AmbientLight(0xffffff, 0.72));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.55);
keyLight.position.set(2.5, 4.5, 3);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.28);
fillLight.position.set(-3, 1.5, -2);
scene.add(fillLight);

let curvatureMesh = null;
let tensorGroup = null;
let currentData = null;
let split = 0.5;
let colorScale = 1;

function jetColor(t) {
  const x = Math.max(0, Math.min(1, t));
  return new THREE.Color(
    Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 3))),
    Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 2))),
    Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 1)))
  );
}

function computeColorScale(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return 1;
  return Math.max(Math.abs(min), Math.abs(max), 1e-9);
}

function curvatureToColor(value, scale = colorScale) {
  const t = Math.max(0, Math.min(1, 0.5 + (value / scale) * 0.5));
  return jetColor(t);
}

function colorsForValues(values) {
  colorScale = computeColorScale(values);
  const colors = new Float32Array(values.length * 3);
  for (let i = 0; i < values.length; i += 1) {
    const c = curvatureToColor(values[i], colorScale);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  return colors;
}

function f32(values) {
  return values instanceof Float32Array ? values : new Float32Array(values);
}

function u32(values) {
  return values instanceof Uint32Array ? values : new Uint32Array(values);
}

function pickCurvatureValues(data) {
  switch (curvatureMode.value) {
    case "max": return data.max_princ_curvature;
    case "min": return data.min_princ_curvature;
    default: return data.mean_curvature;
  }
}

function buildCurvatureMesh(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(f32(data.vertices), 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(f32(data.normals), 3));
  geometry.setIndex(new THREE.BufferAttribute(u32(data.faces), 1));

  const values = pickCurvatureValues(data);
  geometry.setAttribute("color", new THREE.BufferAttribute(colorsForValues(values), 3));

  return new THREE.Mesh(
    geometry,
    new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.FrontSide,
      transparent: false,
      depthWrite: true,
      depthTest: true,
    })
  );
}

function buildSingularityMarkers(data) {
  const group = new THREE.Group();
  const radius = (data.mesh_scale ?? 1) * 0.0075;
  const geo = new THREE.SphereGeometry(radius, 12, 12);

  for (const sing of data.tensor_singularities || []) {
    const p = sing.position;
    const mat = new THREE.MeshBasicMaterial({
      color: sing.type === "positive" ? 0xffeb3b : 0x00e5ff,
    });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(p[0], p[1], p[2]);
    sphere.renderOrder = 2;
    group.add(sphere);
  }
  return group;
}

function buildCrossFieldLicMesh(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(f32(data.vertices), 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(f32(data.normals), 3));
  geometry.setAttribute("dirMax", new THREE.BufferAttribute(f32(data.princ_dir_max), 3));
  geometry.setAttribute("dirMin", new THREE.BufferAttribute(f32(data.princ_dir_min), 3));
  geometry.setIndex(new THREE.BufferAttribute(u32(data.faces), 1));

  const material = createCrossFieldLicMaterial(data.mesh_scale);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = 0;
  return mesh;
}

function buildTensorField(data) {
  const group = new THREE.Group();

  if (data.princ_dir_max && data.princ_dir_min) {
    try {
      group.add(buildCrossFieldLicMesh(data));
      group.add(buildSingularityMarkers(data));
      return group;
    } catch (err) {
      console.warn("Cross-field LIC unavailable, using shaded fallback.", err);
    }
  }

  const baseGeometry = new THREE.BufferGeometry();
  baseGeometry.setAttribute("position", new THREE.BufferAttribute(f32(data.vertices), 3));
  baseGeometry.setAttribute("normal", new THREE.BufferAttribute(f32(data.normals), 3));
  baseGeometry.setIndex(new THREE.BufferAttribute(u32(data.faces), 1));
  group.add(
    new THREE.Mesh(
      baseGeometry,
      new THREE.MeshLambertMaterial({ color: SURFACE_COLOR, side: THREE.FrontSide })
    )
  );
  return group;
}

function rebuildCurvatureColors() {
  if (!curvatureMesh || !currentData) return;
  const values = pickCurvatureValues(currentData);
  const colors = colorsForValues(values);
  curvatureMesh.geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  curvatureMesh.geometry.attributes.color.needsUpdate = true;
}

function disposeObject(obj) {
  if (!obj) return;
  obj.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else if (child.material.isShaderMaterial) disposeCrossFieldLicMaterial(child.material);
      else child.material.dispose();
    }
  });
}

function clearModel() {
  if (curvatureMesh) {
    scene.remove(curvatureMesh);
    disposeObject(curvatureMesh);
    curvatureMesh = null;
  }
  if (tensorGroup) {
    scene.remove(tensorGroup);
    disposeObject(tensorGroup);
    tensorGroup = null;
  }
}

function frameObject(object3d) {
  const box = new THREE.Box3().setFromObject(object3d);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const distance = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360)) * 1.35;
  camera.position.copy(center).add(new THREE.Vector3(distance * 0.8, distance * 0.55, distance));
  controls.target.copy(center);
  controls.update();
}

function setSplit(value) {
  split = Math.max(0.02, Math.min(0.98, value));
  compareSlider.style.left = `${split * 100}%`;
}

function splitFromClientX(clientX) {
  const rect = canvas.getBoundingClientRect();
  return (clientX - rect.left) / rect.width;
}

function setStatus(message, isError = false) {
  if (!statusEl || !statusText) return;
  if (!message) {
    statusEl.hidden = true;
    statusEl.style.display = "none";
    statusEl.classList.remove("error");
    return;
  }
  statusText.textContent = message;
  statusEl.classList.toggle("error", isError);
  statusEl.hidden = false;
  statusEl.style.display = "flex";
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function loadPayload(data) {
  if (!data?.vertices?.length || !data?.faces?.length) {
    throw new Error("Mesh data from server is empty or invalid.");
  }
  currentData = data;
  clearModel();

  await nextFrame();
  curvatureMesh = buildCurvatureMesh(data);
  scene.add(curvatureMesh);
  setSplit(0.5);
  frameObject(curvatureMesh);

  // Show curvature immediately — don't wait for tensor field / LIC shader.
  setStatus("");

  await nextFrame();
  try {
    tensorGroup = buildTensorField(data);
    scene.add(tensorGroup);
  } catch (err) {
    console.warn("Tensor field unavailable:", err);
  }
}

async function fetchJson(url, options, timeoutMs = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    return payload;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out. The server may be waking up — refresh and try again.");
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

async function loadSample(name) {
  setStatus(`Loading ${name}…`);
  try {
    const data = await fetchJson(`/api/sample/${encodeURIComponent(name)}`);
    setStatus("Building 3D view…");
    await loadPayload(data);
  } catch (err) {
    setStatus(err.message || "Failed to load mesh.", true);
    throw err;
  } finally {
    if (curvatureMesh) setStatus("");
  }
}

sampleSelect.addEventListener("change", async () => {
  if (!sampleSelect.value) return;
  try {
    await loadSample(sampleSelect.value);
  } catch (_) {
    /* status already shown */
  }
});

curvatureMode.addEventListener("change", rebuildCurvatureColors);

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const form = new FormData();
    form.append("file", file);
    setStatus("Processing upload…");
    await loadPayload(await fetchJson("/api/upload", { method: "POST", body: form }));
  } catch (err) {
    setStatus(err.message || "Upload failed.", true);
  }
});

window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", async (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files?.[0];
  if (!file) return;
  try {
    const form = new FormData();
    form.append("file", file);
    setStatus("Processing upload…");
    await loadPayload(await fetchJson("/api/upload", { method: "POST", body: form }));
  } catch (err) {
    setStatus(err.message || "Upload failed.", true);
  }
});

let draggingSplit = false;
const SPLIT_HIT_PX = 24;

function startSplitDrag(e) {
  draggingSplit = true;
  controls.enabled = false;
  canvas.classList.add("dragging-split");
  setSplit(splitFromClientX(e.clientX));
  e.preventDefault();
  e.stopPropagation();
}

function nearSplit(clientX) {
  const rect = canvas.getBoundingClientRect();
  const x = clientX - rect.left;
  return Math.abs(x - rect.width * split) <= SPLIT_HIT_PX;
}

compareSlider.addEventListener("mousedown", startSplitDrag);

canvas.addEventListener("mousedown", (e) => {
  if (e.button !== 0 || !nearSplit(e.clientX)) return;
  startSplitDrag(e);
});

window.addEventListener("mousemove", (e) => {
  if (draggingSplit) {
    setSplit(splitFromClientX(e.clientX));
    return;
  }
  canvas.classList.toggle("near-split", nearSplit(e.clientX));
});

window.addEventListener("mouseup", () => {
  if (draggingSplit) {
    draggingSplit = false;
    controls.enabled = true;
    canvas.classList.remove("dragging-split");
  }
});

compareSlider.addEventListener("touchstart", (e) => {
  if (e.touches.length !== 1) return;
  startSplitDrag(e.touches[0]);
  e.preventDefault();
}, { passive: false });

canvas.addEventListener("touchstart", (e) => {
  if (e.touches.length !== 1 || !nearSplit(e.touches[0].clientX)) return;
  startSplitDrag(e.touches[0]);
  e.preventDefault();
}, { passive: false });

window.addEventListener("touchmove", (e) => {
  if (!draggingSplit || e.touches.length !== 1) return;
  setSplit(splitFromClientX(e.touches[0].clientX));
  e.preventDefault();
}, { passive: false });

window.addEventListener("touchend", () => {
  if (draggingSplit) {
    draggingSplit = false;
    controls.enabled = true;
    canvas.classList.remove("dragging-split");
  }
});

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function render() {
  requestAnimationFrame(render);
  controls.update();

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, width, height);
  renderer.setClearColor(BG_COLOR, 1);
  renderer.clear(true, true, true);

  if (width === 0 || height === 0 || !curvatureMesh) {
    return;
  }

  if (!tensorGroup) {
    curvatureMesh.visible = true;
    renderer.render(scene, camera);
    return;
  }

  const splitX = Math.round(width * split);
  renderer.setScissorTest(true);

  curvatureMesh.visible = true;
  tensorGroup.visible = false;
  renderer.setScissor(0, 0, splitX, height);
  renderer.setViewport(0, 0, width, height);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);

  curvatureMesh.visible = false;
  tensorGroup.visible = true;
  renderer.setScissor(splitX, 0, width - splitX, height);
  renderer.setViewport(0, 0, width, height);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);

  renderer.setScissorTest(false);
  curvatureMesh.visible = true;
  tensorGroup.visible = true;
}
render();

async function populateSamples() {
  setStatus("Connecting to server…");
  try {
    const samples = await fetchJson("/api/samples");
    if (samples.length === 0) {
      setStatus("No sample meshes found on server.", true);
      return;
    }
    for (const sample of samples) {
      const option = document.createElement("option");
      option.value = sample.id;
      option.textContent = sample.label;
      sampleSelect.appendChild(option);
    }
    const preferred = samples.find((s) => s.id === "torus.ply")
      || samples.find((s) => s.id === "bunny1.ply")
      || samples[0];
    sampleSelect.value = preferred.id;
    await loadSample(preferred.id);
  } catch (err) {
    setStatus(err.message || "Could not reach server. Wait ~30s if the app is waking up, then refresh.", true);
  }
}

populateSamples().catch((err) => {
  setStatus(err.message || "Failed to start viewer.", true);
});

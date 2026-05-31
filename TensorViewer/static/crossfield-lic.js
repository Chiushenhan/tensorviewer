import * as THREE from "three";

const LIC_STEPS = 22;

const vertexShader = `
attribute vec3 dirMax;
attribute vec3 dirMin;

varying vec3 vPosObject;
varying vec3 vNormalObject;
varying vec3 vDirMaxObject;
varying vec3 vDirMinObject;

void main() {
  vPosObject = position;
  vNormalObject = normal;
  vDirMaxObject = dirMax;
  vDirMinObject = dirMin;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = `
precision highp float;

uniform sampler2D noiseTex;
uniform float noiseScale;
uniform float stepSize;
uniform vec3 lightDir;
uniform vec3 baseColor;

varying vec3 vPosObject;
varying vec3 vNormalObject;
varying vec3 vDirMaxObject;
varying vec3 vDirMinObject;

const int LIC_STEPS = ${LIC_STEPS};

float sampleNoise(vec3 p) {
  vec2 uv = p.xy * noiseScale + p.z * noiseScale * vec2(0.371, 0.619);
  return texture2D(noiseTex, fract(uv)).r;
}

vec3 tangentProject(vec3 v, vec3 n) {
  return v - n * dot(v, n);
}

vec3 hsl2rgb(float h, float s, float l) {
  float c = (1.0 - abs(2.0 * l - 1.0)) * s;
  float hp = h * 6.0;
  float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
  vec3 rgb;
  if (hp < 1.0) rgb = vec3(c, x, 0.0);
  else if (hp < 2.0) rgb = vec3(x, c, 0.0);
  else if (hp < 3.0) rgb = vec3(0.0, c, x);
  else if (hp < 4.0) rgb = vec3(0.0, x, c);
  else if (hp < 5.0) rgb = vec3(x, 0.0, c);
  else rgb = vec3(c, 0.0, x);
  float m = l - 0.5 * c;
  return rgb + vec3(m);
}

vec3 directionHue(vec3 n, vec3 d) {
  vec3 e = tangentProject(d, n);
  float len = length(e);
  if (len < 1e-5) return vec3(0.55);
  e /= len;
  vec3 ref = abs(n.y) < 0.85 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
  vec3 u = cross(n, ref);
  u = normalize(u);
  vec3 v = cross(n, u);
  float ang = atan(dot(e, v), dot(e, u));
  float hue = fract(ang / (2.0 * 3.14159265) + 0.5);
  return hsl2rgb(hue, 0.92, 0.46);
}

float crossFieldLic(vec3 pos, vec3 n, vec3 dMax, vec3 dMin) {
  vec3 e1 = tangentProject(dMax, n);
  vec3 e2 = tangentProject(dMin, n);
  float l1 = length(e1);
  float l2 = length(e2);
  if (l1 > 1e-5) e1 /= l1;
  if (l2 > 1e-5) e2 /= l2;

  float acc = 0.0;
  for (int i = -LIC_STEPS; i <= LIC_STEPS; i++) {
    float t = float(i) * stepSize;
    acc += sampleNoise(pos + e1 * t);
    acc += sampleNoise(pos + e2 * t);
  }
  return acc / float(4 * LIC_STEPS + 2);
}

void main() {
  vec3 n = normalize(vNormalObject);
  vec3 pos = vPosObject;

  float lic = crossFieldLic(pos, n, vDirMaxObject, vDirMinObject);
  lic = smoothstep(0.22, 0.78, lic);

  vec3 cMax = directionHue(n, vDirMaxObject);
  vec3 cMin = directionHue(n, vDirMinObject);
  vec3 fieldColor = mix(cMax, cMin, 0.5);

  float diff = 0.58 + 0.42 * max(dot(n, normalize(lightDir)), 0.0);
  vec3 shadedBase = baseColor * diff;
  vec3 licColor = fieldColor * (0.12 + 0.88 * lic);
  vec3 finalColor = mix(shadedBase, licColor, 0.84);

  gl_FragColor = vec4(finalColor, 1.0);
}
`;

let sharedNoiseTexture = null;

export function getNoiseTexture() {
  if (sharedNoiseTexture) return sharedNoiseTexture;
  const size = 256;
  const data = new Uint8Array(size * size * 4);
  for (let i = 0; i < size * size; i += 1) {
    const v = Math.floor(Math.random() * 256);
    const j = i * 4;
    data[j] = v;
    data[j + 1] = v;
    data[j + 2] = v;
    data[j + 3] = 255;
  }
  sharedNoiseTexture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  sharedNoiseTexture.wrapS = THREE.RepeatWrapping;
  sharedNoiseTexture.wrapT = THREE.RepeatWrapping;
  sharedNoiseTexture.minFilter = THREE.LinearFilter;
  sharedNoiseTexture.magFilter = THREE.LinearFilter;
  sharedNoiseTexture.needsUpdate = true;
  return sharedNoiseTexture;
}

export function createCrossFieldLicMaterial(meshScale) {
  const scale = Math.max(meshScale ?? 1, 1e-6);
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL1,
    uniforms: {
      noiseTex: { value: getNoiseTexture() },
      noiseScale: { value: 32.0 / scale },
      stepSize: { value: scale * 0.0026 },
      lightDir: { value: new THREE.Vector3(0.35, 0.95, 0.55).normalize() },
      baseColor: { value: new THREE.Color(0.953, 0.941, 0.902) },
    },
    vertexShader,
    fragmentShader,
    side: THREE.FrontSide,
    depthWrite: true,
    depthTest: true,
  });
}

export function disposeCrossFieldLicMaterial(material) {
  if (!material) return;
  material.dispose();
}

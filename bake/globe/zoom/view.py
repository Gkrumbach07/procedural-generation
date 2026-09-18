"""A zoom level (or a saved zoom window) as a lit 3-D mesh in one
self-contained HTML page: WebGL2, the 16-bit height texture fetched per
vertex, lakes and sea as flat water at their level, rivers from the
particles' discharge eased in (McDonald's stream-map rendering), ground
coloured by height and slope.  Drag to orbit, wheel to zoom, shift-drag to
pan; the vertical scale is true unless ``exag``."""
from __future__ import annotations

import base64
import io
import json
import math

import numpy as np
from PIL import Image
from scipy import ndimage


def _block(a: np.ndarray, k: int, how: str) -> np.ndarray:
    if k <= 1:
        return a
    n0, n1 = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:n0, :n1].reshape(n0 // k, k, n1 // k, k)
    return b.max(axis=(1, 3)) if how == "max" else b.mean(axis=(1, 3))


def _png(rgba: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba), "RGBA").save(buf, "PNG", optimize=False, compress_level=6)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_html(height, sediment, discharge, water_surface, mask, cell_m: float, title: str = "zoom", max_res: int = 1024, exag: float = 1.0,
               ocean=None, rain_cell: float | None = None, min_lake_cells: int = 4, min_lake_depth: float = 1.0, back: str | None = None,
               crop: bool = True) -> tuple[str, dict]:
    """The page for one square of arrays (metres; ``mask`` > 0 = drawn
    land, ``ocean`` = drawn as sea at 0 m).  Rivers: with ``rain_cell`` (the
    rain volume of one cell, the discharge unit) the stream map is eased in
    from 64 to 16,384 cells of upstream rain, the same at every level;
    without it from the 80th to the 99.5th percentile of the land's."""
    surf = (np.asarray(height, np.float64) + np.asarray(sediment, np.float64))
    mask = np.asarray(mask) > 0
    ocean = np.zeros(mask.shape, bool) if ocean is None else np.asarray(ocean, bool)
    drawn = mask | ocean
    sl = (slice(0, surf.shape[0]), slice(0, surf.shape[1]))
    if crop and drawn.any():
        idx = np.argwhere(drawn)
        (i0, j0), (i1, j1) = idx.min(0), idx.max(0) + 1
        m = max(4, (i1 - i0) // 40) if not ocean.any() else 0
        sl = (slice(max(i0 - m, 0), min(i1 + m, surf.shape[0])), slice(max(j0 - m, 0), min(j1 + m, surf.shape[1])))
    surf, mask, ocean = surf[sl], mask[sl], ocean[sl]
    ws = np.asarray(water_surface, np.float64)[sl]
    q = np.asarray(discharge, np.float64)[sl]
    sed = np.asarray(sediment, np.float64)[sl]
    k = max(1, math.ceil(max(surf.shape) / max_res))
    depth = np.where(mask & ~ocean, ws - surf, 0.0)
    lake = depth > min_lake_depth
    if min_lake_cells > 1 and lake.any():
        # a pool of a few cells is drawn as ground: the stream map carries the water there
        lab, n = ndimage.label(lake, structure=np.ones((3, 3), bool))
        size = np.bincount(lab.ravel())
        lake = lake & (size[lab] >= min_lake_cells)
    sea_depth = np.where(ocean, np.maximum(-surf, 0.0), 0.0)
    level = np.where(lake, ws, np.where(ocean, np.maximum(surf, 0.0), surf))   # water stands flat at its level
    level = _block(level, k, "mean")
    surf_b = _block(surf, k, "mean")
    depth_b = _block(np.where(lake, depth, 0.0), k, "max")
    sea_b = _block(np.where(ocean, np.maximum(sea_depth, 1.0), 0.0), k, "mean")
    q_b = _block(q, k, "max")
    sed_b = _block(sed, k, "mean")
    mask_b = _block(mask.astype(np.float64), k, "mean") > 0.5
    ocean_b = _block(ocean.astype(np.float64), k, "mean") > 0.5
    H, W = level.shape
    lo, hi = float(level.min()), float(level.max())
    hq = np.clip(np.round((level - lo) / max(hi - lo, 1e-6) * 65535), 0, 65535).astype(np.uint16)
    land = mask_b & ~ocean_b & (depth_b <= 0.5)
    lq = np.log1p(q_b / (rain_cell if rain_cell else 1.0)) if rain_cell else np.log1p(q_b)
    if rain_cell:
        qlo, qhi = math.log1p(64.0), math.log1p(16384.0)
    else:
        qlo, qhi = (np.percentile(lq[land], [80.0, 99.5]) if land.any() else (0.0, 1.0))
    qb = np.clip(np.round((lq - qlo) / max(qhi - qlo, 1e-6) * 255), 0, 255).astype(np.uint8)
    db = np.clip(np.round(depth_b / 50.0 * 255), 0, 255).astype(np.uint8)
    t0 = np.stack([(hq >> 8).astype(np.uint8), (hq & 255).astype(np.uint8), qb, db], -1)
    sb = np.clip(np.round(np.log1p(np.maximum(sed_b, 0)) / np.log1p(200.0) * 255), 0, 255).astype(np.uint8)
    ob = np.clip(np.round(np.where(ocean_b, np.maximum(sea_b, 1.0), 0.0) / 50.0 * 255), 0, 255).astype(np.uint8)
    t1 = np.stack([np.where(mask_b & ~ocean_b, 255, 0).astype(np.uint8), sb, ob, np.full_like(sb, 255)], -1)
    # default focus: the centroid of the highest tenth of the land, at its height
    if land.any():
        top = land & (surf_b >= np.percentile(surf_b[land], 90))
        ti, tj = np.argwhere(top).mean(axis=0)
        focus = [float(tj / max(W - 1, 1) - 0.5), float(ti / max(H - 1, 1) - 0.5), float(np.median(level[top]) - lo)]
    else:
        focus = [0.0, 0.0, 0.0]
    # ground colour by absolute elevation (green below 1.8 km, bare by 4.2 km,
    # snow from 5.1 km on gentle slopes): a window's own range painted a 600 m
    # lowland ridge as snow
    meta = {"W": W, "H": H, "cell_m": cell_m * k, "h0": lo, "h1": hi, "exag": exag, "focus": focus,
            "land_lo": 0.0, "land_hi": 6000.0,
            "title": title, "native_cell_m": cell_m, "step": k, "back": back}
    html = TEMPLATE.replace("__META__", json.dumps(meta)).replace("__T0__", _png(t0)).replace("__T1__", _png(t1))
    return html, meta


def screenshot(html_path, png_path, yaw=35.0, pitch=35.0, dist=1.2, tx=0.0, ty=0.0, exag=1.0, size=(1400, 900), trees=1) -> list[str]:
    """PNG of a page (Playwright, SwiftShader).  Returns page errors."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    w, h = size
    errors: list[str] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--allow-file-access-from-files"])
        page = browser.new_page(viewport={"width": int(w), "height": int(h)})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.goto(Path(html_path).resolve().as_uri() + f"#yaw={yaw}&pitch={pitch}&dist={dist}&tx={tx}&ty={ty}&exag={exag}&trees={trees}")
        page.wait_for_function("document.body.dataset.ready === '1'", timeout=300_000)
        page.wait_for_timeout(300)
        page.screenshot(path=str(png_path))
        browser.close()
    return errors


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>zoom window</title>
<style>html,body{margin:0;height:100%;background:#9fb3c8;overflow:hidden;font:12px system-ui}
#hud{position:fixed;left:10px;top:8px;color:#123;background:#fff8;padding:4px 8px;border-radius:4px}</style></head>
<body><canvas id="c"></canvas><div id="hud"></div>
<script>
const META = __META__;
const T0 = "data:image/png;base64,__T0__", T1 = "data:image/png;base64,__T1__";
const cv = document.getElementById("c"), gl = cv.getContext("webgl2", {antialias: true});
const P = new URLSearchParams(location.hash.slice(1));
const st = {yaw: +(P.get("yaw") ?? 35), pitch: +(P.get("pitch") ?? 35), dist: +(P.get("dist") ?? 1.2),
            tx: +(P.get("tx") ?? 0), ty: +(P.get("ty") ?? 0), exag: +(P.get("exag") ?? META.exag), sun: +(P.get("sun") ?? 135),
            trees: +(P.get("trees") ?? 1)};
// tx / ty offset the view from the focus (the highest tenth of the window), in window widths
const VS = `#version 300 es
precision highp float; precision highp int;
in vec2 aUV; uniform sampler2D uT0; uniform mat4 uMVP; uniform vec2 uRes; uniform vec3 uScale;
out vec2 vUV; out float vH;
float hAt(ivec2 p){ vec4 c = texelFetch(uT0, clamp(p, ivec2(0), ivec2(uRes) - 1), 0);
  return (floor(c.r*255.0+0.5)*256.0 + floor(c.g*255.0+0.5)) / 65535.0; }
void main(){ ivec2 p = ivec2(aUV * (uRes - 1.0) + 0.5); float h = hAt(p);
  vUV = (vec2(p) + 0.5) / uRes; vH = h;
  gl_Position = uMVP * vec4((aUV.x - 0.5) * uScale.x, h * uScale.z, (aUV.y - 0.5) * uScale.y, 1.0); }`;
const FS = `#version 300 es
precision highp float; out vec4 o; in vec2 vUV; in float vH;
uniform sampler2D uT0, uT1, uT0L; uniform vec2 uRes; uniform vec3 uScale; uniform vec3 uSun; uniform vec2 uLand; uniform float uH0, uH1;
uniform float uTrees, uBase;
float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
vec4 T(sampler2D t, vec2 uv){ return texture(t, uv); }
float H1(ivec2 p){ vec4 c = texelFetch(uT0, clamp(p, ivec2(0), ivec2(uRes) - 1), 0); return (floor(c.r*255.0+0.5)*256.0 + floor(c.g*255.0+0.5)) / 65535.0; }
float H(vec2 uv){ vec2 x = uv * uRes - 0.5; vec2 i = floor(x); vec2 f = x - i; ivec2 a = ivec2(i);   // bilinear on decoded heights
  return mix(mix(H1(a), H1(a + ivec2(1,0)), f.x), mix(H1(a + ivec2(0,1)), H1(a + ivec2(1,1)), f.x), f.y); }
void main(){
  vec4 a = T(uT0L, vUV), b = T(uT1, vUV);   // discharge and water depth interpolated: shores between cells
  if (b.r < 0.5 && b.b < 0.5) discard;   // outside the window
  vec2 e = 1.0 / uRes;
  float hx = (H(vUV + vec2(e.x, 0.0)) - H(vUV - vec2(e.x, 0.0))) * uScale.z / (2.0 * uScale.x / uRes.x);
  float hy = (H(vUV + vec2(0.0, e.y)) - H(vUV - vec2(0.0, e.y))) * uScale.z / (2.0 * uScale.y / uRes.y);
  vec3 n = normalize(vec3(-hx, 1.0, -hy));
  float slope = length(vec2(hx, hy));
  float hm = uH0 + vH * (uH1 - uH0) + uBase;
  float t = clamp((hm - uLand.x) / max(uLand.y - uLand.x, 1.0), 0.0, 1.0);
  vec3 grass = mix(vec3(0.30, 0.42, 0.20), vec3(0.45, 0.50, 0.28), smoothstep(0.0, 0.3, t));
  vec3 col = mix(grass, vec3(0.52, 0.47, 0.40), smoothstep(0.3, 0.7, t));
  col = mix(col, vec3(0.93, 0.94, 0.96), smoothstep(0.85, 0.95, t) * smoothstep(0.9, 0.4, slope));
  col = mix(col, vec3(0.45, 0.42, 0.38), smoothstep(0.45, 0.9, slope));          // bare rock on steep ground
  col = mix(col, vec3(0.62, 0.56, 0.44), b.g * 0.35 * smoothstep(0.3, 0.05, slope)); // alluvium on flats
  float depth = max(a.a * 50.0, b.b * 50.0);
  float qx = a.b; float w = qx * qx * (3.0 - 2.0 * qx);
  if (uTrees > 0.5) {
    // forest on gentle ground below the tree line and away from water: crowns
    // ~18 m apart where the pixel resolves them, their mean cover beyond
    vec2 c = vUV * uScale.xy / 18.0; vec2 ci = floor(c);
    vec2 fr = fract(c) - 0.5 - (vec2(hash(ci), hash(ci + 7.3)) - 0.5) * 0.5;
    float forest = smoothstep(0.7, 0.25, slope) * smoothstep(0.7, 0.45, t) * (1.0 - w) * (depth > 0.5 ? 0.0 : 1.0);
    float crown = smoothstep(0.45, 0.2, length(fr)) * step(hash(ci + 3.1), forest);
    float k = clamp(1.3 - fwidth(c.x), 0.0, 1.0);
    float cover = mix(0.6 * forest, crown, k);
    col = mix(col, mix(vec3(0.13, 0.27, 0.11), vec3(0.22, 0.36, 0.16), hash(ci + 1.7)) * (0.8 + 0.25 * k * (0.45 - length(fr))), cover * 0.85);
  }
  vec3 river = mix(vec3(0.33, 0.52, 0.62), vec3(0.16, 0.34, 0.52), w);
  col = mix(col, river, clamp(w * 1.1, 0.0, 0.95));
  float dif = max(dot(n, uSun), 0.0);
  vec3 lit = col * (0.38 + 0.75 * dif);
  if (depth > 0.5) {
    vec3 water = mix(vec3(0.22, 0.40, 0.50), vec3(0.08, 0.20, 0.34), clamp(depth / 30.0, 0.0, 1.0));
    lit = water * (0.75 + 0.35 * max(uSun.y, 0.0)) + vec3(0.9) * pow(max(dot(reflect(-uSun, vec3(0,1,0)), vec3(0,1,0)), 0.0), 40.0) * 0.1;
  }
  o = vec4(pow(lit, vec3(0.9)), 1.0);
}`;
function sh(t, s){ const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x);
  if (!gl.getShaderParameter(x, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(x)); return x; }
const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(pr); if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr)); gl.useProgram(pr);
const U = n => gl.getUniformLocation(pr, n);
const W = META.W, Hh = META.H;
const uv = new Float32Array(W * Hh * 2); let k = 0;
for (let i = 0; i < Hh; i++) for (let j = 0; j < W; j++) { uv[k++] = j / (W - 1); uv[k++] = i / (Hh - 1); }
const idx = new Uint32Array((W - 1) * (Hh - 1) * 6); k = 0;
for (let i = 0; i < Hh - 1; i++) for (let j = 0; j < W - 1; j++) { const a = i * W + j, b = a + 1, c = a + W, d = c + 1;
  idx[k++] = a; idx[k++] = c; idx[k++] = b; idx[k++] = b; idx[k++] = c; idx[k++] = d; }
const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
const loc = gl.getAttribLocation(pr, "aUV"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
function tex(unit, src, linear){ return new Promise(res => { const im = new Image(); im.onload = () => {
  const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, im);
  const f = linear ? gl.LINEAR : gl.NEAREST;
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  res(t); }; im.src = src; }); }
function mat4mul(a, b){ const o = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
  let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; o[i * 4 + j] = s; } return o; }
function persp(f, asp, n, fa){ const t = 1 / Math.tan(f / 2); return new Float32Array([t/asp,0,0,0, 0,t,0,0, 0,0,(fa+n)/(n-fa),-1, 0,0,2*fa*n/(n-fa),0]); }
function look(e, c, up){ const z = norm(sub(e, c)), x = norm(cross(up, z)), y = cross(z, x);
  return new Float32Array([x[0],y[0],z[0],0, x[1],y[1],z[1],0, x[2],y[2],z[2],0, -dot(x,e),-dot(y,e),-dot(z,e),1]); }
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]], dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]], norm=a=>{const l=Math.hypot(...a)||1;return a.map(v=>v/l)};
const SX = META.W * META.cell_m, SY = META.H * META.cell_m, SPAN = Math.max(SX, SY);
function draw(){
  cv.width = innerWidth * devicePixelRatio; cv.height = innerHeight * devicePixelRatio; gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0.62, 0.70, 0.78, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  const yaw = st.yaw * Math.PI / 180, pit = st.pitch * Math.PI / 180, d = st.dist * SPAN;
  const c = [(META.focus[0] * SX / SPAN + st.tx) * SPAN, META.focus[2] * st.exag, (META.focus[1] * SY / SPAN + st.ty) * SPAN];
  const eye = [c[0] + d * Math.cos(pit) * Math.sin(yaw), c[1] + d * Math.sin(pit), c[2] + d * Math.cos(pit) * Math.cos(yaw)];
  const mvp = mat4mul(persp(0.8, cv.width / cv.height, SPAN * 0.002, SPAN * 8), look(eye, c, [0, 1, 0]));
  gl.uniformMatrix4fv(U("uMVP"), false, mvp); gl.uniform2f(U("uRes"), META.W, META.H);
  gl.uniform3f(U("uScale"), SX, SY, (META.h1 - META.h0) * st.exag);
  const sa = st.sun * Math.PI / 180; gl.uniform3fv(U("uSun"), norm([Math.cos(sa), 0.9, Math.sin(sa)]));
  gl.uniform2f(U("uLand"), META.land_lo, META.land_hi); gl.uniform1f(U("uH0"), 0); gl.uniform1f(U("uH1"), META.h1 - META.h0); gl.uniform1f(U("uBase"), META.h0);
  gl.uniform1i(U("uT0"), 0); gl.uniform1i(U("uT1"), 1); gl.uniform1i(U("uT0L"), 2); gl.uniform1f(U("uTrees"), st.trees);
  gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_INT, 0);
  document.getElementById("hud").innerHTML = (META.back ? `<a href="${META.back}">← open in the viewer</a> (satellite, trees, water, time lapse) · ` : "") + `${META.title} · ${(SX/1000).toFixed(1)} x ${(SY/1000).toFixed(1)} km · ${META.native_cell_m.toFixed(0)} m cells (drawn at ${META.cell_m.toFixed(0)} m) · relief ${(META.h1-META.h0).toFixed(0)} m · exaggeration ${st.exag}x · T trees`;
}
let drag = null;
cv.onmousedown = e => drag = {x: e.clientX, y: e.clientY, pan: e.shiftKey};
onmouseup = () => drag = null;
onmousemove = e => { if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) { st.tx -= dx * 0.001 * st.dist; st.ty -= dy * 0.001 * st.dist; } else { st.yaw -= dx * 0.3; st.pitch = Math.max(5, Math.min(89, st.pitch + dy * 0.3)); }
  draw(); };
cv.onwheel = e => { st.dist *= Math.exp(e.deltaY * 0.001); draw(); e.preventDefault(); };
onresize = draw;
onkeydown = e => { if (e.key === "t" || e.key === "T") { st.trees = st.trees ? 0 : 1; draw(); } };
Promise.all([tex(0, T0, false), tex(1, T1, true), tex(2, T0, true)]).then(() => { gl.activeTexture(gl.TEXTURE0); draw(); document.body.dataset.ready = "1"; });
</script></body></html>"""


__all__ = ["build_html", "screenshot", "TEMPLATE"]

"""Export a baked world as a standalone Godot project you can fly around.

    python3 scripts/export_godot.py --world worlds/demo --out /tmp/globe
    godot --path /tmp/globe

Writes an equirectangular albedo and a 16-bit elevation map sampled off the
cube-sphere, plus a small Godot 4 project that draws them on a displaced
sphere with an orbit camera -- drag to spin, wheel to zoom, right-drag to
pan the light.

This replaces the in-engine streaming runtime that used to live in `game/`
and `gdextension/`. The point is to *look at* a world and to be a starting
point for a project that does more, not to be a game: there is no tile
streaming, no LOD, no collision and no player. A whole world is one texture,
which is why it is capped at a few thousand pixels and why the fine grid is
only worth exporting for a small body.

`--tiles` also copies the baked tile pyramid into the project, for a viewer
that outgrows a single texture.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from globe.cubesphere import get_grid          # noqa: E402
from globe.field import FaceField              # noqa: E402


def equirect(field: FaceField, w: int, h: int) -> np.ndarray:
    """Sample a FaceField onto a `w x h` equirectangular grid.

    Longitude runs -pi..pi across, latitude +pi/2..-pi/2 down, which is what
    Godot's SphereMesh UV expects, so the texture lands the right way up
    without a flip in the shader.
    """
    lon = (np.arange(w) + 0.5) / w * 2.0 * np.pi - np.pi
    lat = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    lo, la = np.meshgrid(lon, lat)
    p = np.stack([np.cos(la) * np.cos(lo), np.sin(la), np.cos(la) * np.sin(lo)], axis=-1)
    return field.sample_sphere(np.ascontiguousarray(p.reshape(-1, 3))).reshape(h, w)


def shade(z: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Relief shading on the equirect grid.

    The horizontal gradient is scaled by cos(latitude) because a degree of
    longitude is that much shorter away from the equator -- without it the
    poles look like cliffs.
    """
    h, w = z.shape
    lat = np.pi / 2.0 - (np.arange(h) + 0.5) / h * np.pi
    gy, gx = np.gradient(z)
    gx = gx / np.maximum(np.cos(lat)[:, None], 1e-3)
    slope = np.arctan(np.hypot(gx, gy) * strength)
    aspect = np.arctan2(-gx, gy)
    a, z0 = np.radians(45.0), np.radians(315.0)
    return np.clip(np.sin(a) * np.cos(slope) + np.cos(a) * np.sin(slope) * np.cos(z0 - aspect), 0.0, 1.0)


def albedo(z: np.ndarray) -> np.ndarray:
    """Land ramp plus a depth ramp for the ocean, relief-shaded.

    Land and sea are shaded at different strengths. An abyssal plain is
    nearly flat, so the exaggeration that makes mountains read turns its
    remaining wobble into false contour bands -- the sea floor gets a gentle
    shade and the land keeps the strong one.
    """
    land = z > 0
    rgb = np.zeros(z.shape + (3,), dtype=np.float64)
    if land.any():
        t = np.clip(z / max(float(np.percentile(z[land], 98)), 1e-6), 0.0, 1.0)
        lo = np.array([0.36, 0.47, 0.30])       # lowland
        mid = np.array([0.55, 0.51, 0.35])      # upland
        hi = np.array([0.88, 0.88, 0.86])       # bare rock / snow
        a = np.where((t < 0.5)[..., None], lo + (mid - lo) * (t / 0.5)[..., None],
                     mid + (hi - mid) * ((t - 0.5) / 0.5)[..., None])
        rgb[land] = a[land]
    sea = ~land
    if sea.any():
        d = np.clip(z / min(float(np.percentile(z[sea], 2)), -1.0), 0.0, 1.0)
        shallow = np.array([0.16, 0.35, 0.55])
        deep = np.array([0.04, 0.09, 0.22])
        rgb[sea] = (shallow + (deep - shallow) * d[..., None])[sea]
    sh = np.where(land, 0.45 + 0.75 * shade(z, 6.0), 0.75 + 0.35 * shade(z, 0.7))
    rgb *= sh[..., None]
    return np.clip(rgb, 0.0, 1.0)


PROJECT = '''\
config_version=5

[application]
config/name="{name}"
run/main_scene="res://Globe.tscn"
config/features=PackedStringArray("4.3", "Forward Plus")

[rendering]
textures/default_filters/anisotropic_filtering_level=3
'''

SCENE = '''\
[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://Globe.gd" id="1"]

[node name="Globe" type="Node3D"]
script = ExtResource("1")
'''

GLOBE_GD = '''\
# A baked world on a displaced sphere, with an orbit camera.
#
# Everything is built in code so the project is three files. The mesh is a
# high-subdivision SphereMesh displaced in the vertex shader by the exported
# elevation map, which keeps the relief geometric (it catches the light and
# shows a silhouette) rather than painted on.
extends Node3D

const RADIUS := 1.0
const MIN_DIST := 1.02      # just above the surface
const MAX_DIST := 6.0

var _dist := 3.0
var _yaw := 0.0
var _pitch := 0.0
var _cam: Camera3D
var _light: DirectionalLight3D
var _dragging := false
var _light_dragging := false

func _ready() -> void:
    var meta: Dictionary = {}
    var f := FileAccess.open("res://world.json", FileAccess.READ)
    if f:
        meta = JSON.parse_string(f.get_as_text())
        f.close()

    var albedo_tex: Texture2D = load("res://globe_albedo.png")
    var height_tex: Texture2D = load("res://globe_height.png")

    var mesh := SphereMesh.new()
    mesh.radius = RADIUS
    mesh.height = RADIUS * 2.0
    mesh.radial_segments = 512
    mesh.rings = 256

    var shader := Shader.new()
    shader.code = SHADER
    var mat := ShaderMaterial.new()
    mat.shader = shader
    mat.set_shader_parameter("albedo_tex", albedo_tex)
    mat.set_shader_parameter("height_tex", height_tex)
    # exaggerated, or an Earth-sized planet reads as a smooth ball: 9 km of
    # relief on 6371 km is a 0.14 % bump
    mat.set_shader_parameter("relief", float(meta.get("relief_scale", 0.02)))

    var mi := MeshInstance3D.new()
    mi.mesh = mesh
    mi.material_override = mat
    add_child(mi)

    _light = DirectionalLight3D.new()
    _light.rotation_degrees = Vector3(-35, 130, 0)
    _light.light_energy = 1.15
    add_child(_light)

    var env := Environment.new()
    env.background_mode = Environment.BG_COLOR
    env.background_color = Color(0.02, 0.02, 0.035)
    env.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
    env.ambient_light_color = Color(0.32, 0.35, 0.42)
    env.ambient_light_energy = 0.45
    var we := WorldEnvironment.new()
    we.environment = env
    add_child(we)

    _cam = Camera3D.new()
    _cam.near = 0.001
    _cam.far = 100.0
    add_child(_cam)
    _update_camera()

    var label := Label.new()
    label.text = "%s\\n%s\\ndrag orbit  ·  wheel zoom  ·  right-drag light" % [
        meta.get("name", "world"), meta.get("summary", "")]
    label.position = Vector2(12, 8)
    label.add_theme_color_override("font_color", Color(0.85, 0.87, 0.9))
    var ui := CanvasLayer.new()
    ui.add_child(label)
    add_child(ui)

func _update_camera() -> void:
    var b := Basis(Vector3.UP, _yaw) * Basis(Vector3.RIGHT, _pitch)
    _cam.position = b * Vector3(0, 0, _dist)
    _cam.look_at(Vector3.ZERO, Vector3.UP)

func _unhandled_input(e: InputEvent) -> void:
    if e is InputEventMouseButton:
        if e.button_index == MOUSE_BUTTON_LEFT:
            _dragging = e.pressed
        elif e.button_index == MOUSE_BUTTON_RIGHT:
            _light_dragging = e.pressed
        elif e.button_index == MOUSE_BUTTON_WHEEL_UP:
            _zoom(-0.1)
        elif e.button_index == MOUSE_BUTTON_WHEEL_DOWN:
            _zoom(0.1)
    elif e is InputEventMouseMotion:
        if _dragging:
            # scale with distance so the drag feels the same when zoomed in
            var k := 0.005 * clampf(_dist - RADIUS, 0.05, 3.0)
            _yaw -= e.relative.x * k
            _pitch = clampf(_pitch - e.relative.y * k, -1.5, 1.5)
            _update_camera()
        elif _light_dragging:
            _light.rotation_degrees.y += e.relative.x * 0.4
            _light.rotation_degrees.x = clampf(_light.rotation_degrees.x - e.relative.y * 0.3, -89, -2)
    elif e is InputEventKey and e.pressed and e.keycode == KEY_ESCAPE:
        get_tree().quit()

func _zoom(f: float) -> void:
    # proportional to height above the surface, so approach slows near it
    _dist = clampf(_dist + (_dist - RADIUS) * f * 2.0, MIN_DIST, MAX_DIST)
    _update_camera()

const SHADER := """
shader_type spatial;
render_mode cull_back, diffuse_burley;

uniform sampler2D albedo_tex : source_color, filter_linear;
uniform sampler2D height_tex : hint_default_black, filter_linear;
uniform float relief = 0.02;

varying vec3 v_normal_ws;

float elev(vec2 uv) { return texture(height_tex, uv).r; }

void vertex() {
    float e = elev(UV);
    VERTEX += normalize(VERTEX) * e * relief;
    // re-derive the normal from the height map: the displaced sphere's own
    // normal is still the sphere's, so without this the relief is invisible
    vec2 t = vec2(1.0 / 4096.0, 1.0 / 2048.0);
    float ex = elev(UV + vec2(t.x, 0.0)) - elev(UV - vec2(t.x, 0.0));
    float ey = elev(UV + vec2(0.0, t.y)) - elev(UV - vec2(0.0, t.y));
    vec3 n = normalize(NORMAL);
    vec3 tang = normalize(cross(vec3(0.0, 1.0, 0.0), n));
    vec3 bitan = cross(n, tang);
    NORMAL = normalize(n - (tang * ex + bitan * ey) * relief * 40.0);
}

void fragment() {
    ALBEDO = texture(albedo_tex, UV).rgb;
    ROUGHNESS = 0.92;
    SPECULAR = 0.12;
}
"""
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--width", type=int, default=4096, help="equirect texture width (height is half)")
    ap.add_argument("--level", choices=("auto", "fine", "coarse"), default="auto")
    ap.add_argument("--relief", type=float, default=0.0, help="0 = pick one that makes the relief visible")
    ap.add_argument("--tiles", action="store_true", help="also copy the baked tile pyramid")
    a = ap.parse_args()

    from PIL import Image

    meta = json.loads((a.world / "manifest.json").read_text())
    R = int(meta["params"]["world"]["R"])
    N_c = int(meta["params"]["world"]["N_c"])
    use_fine = ((a.world / "fine" / "height.f0.npy").exists() if a.level == "auto"
                else a.level == "fine")
    d, N = (("fine", N_c * R) if use_fine else ("coarse", N_c))
    arrs = np.stack([np.load(a.world / d / f"height.f{f}.npy") for f in range(6)]).astype(np.float32)
    grid = get_grid(N, int(meta["params"]["world"]["halo"]))
    field = FaceField.from_interior(grid, arrs, exchange=True)

    w = int(a.width)
    h = w // 2
    z = equirect(field, w, h).astype(np.float64)
    print(f"sampled {d} grid ({N}^2/face) -> {w}x{h} equirect; "
          f"relief {z.min():.0f}..{z.max():.0f} m, land {(z > 0).mean() * 100:.1f} %")

    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((albedo(z) * 255).astype(np.uint8)).save(a.out / "globe_albedo.png")
    # elevation normalised over the *whole* range, sea floor included, so the
    # displaced sphere keeps its ocean basins
    lo, hi = float(z.min()), float(z.max())
    e = (z - lo) / max(hi - lo, 1e-6)
    Image.fromarray((e * 65535).astype(np.uint16)).save(a.out / "globe_height.png")

    R_planet = float(meta.get("R_planet_m", N_c * meta["params"]["world"]["cell_size_m"] * 4 / (2 * np.pi)))
    true_scale = (hi - lo) / R_planet
    relief = a.relief if a.relief > 0 else max(0.015, min(0.06, true_scale * 25.0))
    (a.out / "project.godot").write_text(PROJECT.format(name=a.world.name))
    (a.out / "Globe.tscn").write_text(SCENE)
    (a.out / "Globe.gd").write_text(GLOBE_GD)
    (a.out / "world.json").write_text(json.dumps({
        "name": a.world.name,
        "summary": f"R {R_planet / 1000:.0f} km · {d} grid {N}²/face · relief {hi - lo:.0f} m",
        "relief_scale": relief,
        "true_relief_fraction": true_scale,
    }, indent=2))
    if a.tiles and (a.world / "tiles").exists():
        shutil.copytree(a.world / "tiles", a.out / "tiles", dirs_exist_ok=True)
        print("copied tile pyramid")
    print(f"wrote {a.out}  (relief x{relief / max(true_scale, 1e-9):.0f} exaggerated; "
          f"true relief is {true_scale * 100:.3f} % of radius)")
    print(f"  godot --path {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

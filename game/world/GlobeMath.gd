## Cube-sphere math in GDScript (mirror of bake/globe/cubesphere.py and the
## GDExtension CubeSphere class).  Used for the handful of per-frame calls
## the streamer needs; the extension is only required for fast tile decode.
class_name GlobeMath
extends RefCounted

const HALF_PI := PI * 0.5
const TWO_OVER_PI := 2.0 / PI
# faces 0..5 = +X, -X, +Y, -Y, +Z, -Z ; [right, up, normal] (OpenGL cubemap)
const BASES := [
	[Vector3(0, 0, -1), Vector3(0, -1, 0), Vector3(1, 0, 0)],
	[Vector3(0, 0, 1), Vector3(0, -1, 0), Vector3(-1, 0, 0)],
	[Vector3(1, 0, 0), Vector3(0, 0, 1), Vector3(0, 1, 0)],
	[Vector3(1, 0, 0), Vector3(0, 0, -1), Vector3(0, -1, 0)],
	[Vector3(1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, 1)],
	[Vector3(-1, 0, 0), Vector3(0, -1, 0), Vector3(0, 0, -1)],
]
# unfolded-net slots (row, col) used by the debug minimap; same as viz/unfold.py
const NET_SLOTS := {4: Vector2i(1, 1), 0: Vector2i(1, 2), 1: Vector2i(1, 0), 2: Vector2i(0, 1), 3: Vector2i(2, 1), 5: Vector2i(1, 3)}


static func planet_radius(n: int, cell_size_m: float) -> float:
	return n * cell_size_m * 4.0 / (2.0 * PI)


static func to_sphere(face: int, u: float, v: float) -> Vector3:
	var s := tan((u - 0.5) * HALF_PI)
	var t := tan((v - 0.5) * HALF_PI)
	var b: Array = BASES[face]
	return (b[2] + s * b[0] + t * b[1]).normalized()


static func face_of(p: Vector3) -> int:
	var ax := absf(p.x)
	var ay := absf(p.y)
	var az := absf(p.z)
	if ax >= ay and ax >= az:
		return 0 if p.x >= 0.0 else 1
	if ay >= az:
		return 2 if p.y >= 0.0 else 3
	return 4 if p.z >= 0.0 else 5


## Returns [face, u, v] with u, v in the closed interval [0, 1].
static func from_sphere(p: Vector3) -> Array:
	var face := face_of(p)
	var b: Array = BASES[face]
	var d: float = p.dot(b[2])
	var s: float = p.dot(b[0]) / d
	var t: float = p.dot(b[1]) / d
	return [face, atan(s) * TWO_OVER_PI + 0.5, atan(t) * TWO_OVER_PI + 0.5]


## Tile (x, y) at `lod` containing face-local (u, v); n_fine fine cells per face, T per tile.
static func tile_of(u: float, v: float, lod: int, n_fine: int, t_size: int) -> Vector2i:
	var tiles: int = maxi(1, (n_fine / t_size) >> lod)
	return Vector2i(clampi(int(floor(u * tiles)), 0, tiles - 1), clampi(int(floor(v * tiles)), 0, tiles - 1))


## Right-handed tangent frame at a unit vector: Basis(x = e1, y = anchor, z = e2)
static func tangent_frame(anchor: Vector3) -> Basis:
	var a := anchor.normalized()
	var ax := Vector3(1, 0, 0) if absf(a.x) < 0.9 else Vector3(0, 1, 0)
	var e1 := (ax - a * ax.dot(a)).normalized()
	var e2 := e1.cross(a)
	return Basis(e1, a, e2)


## Gnomonic local position (PLAN 12.2) of unit vector q at height h.
static func gnomonic_local(q: Vector3, h: float, frame: Basis, r_planet: float) -> Vector3:
	var d: float = maxf(q.dot(frame.y), 0.05)
	var g := q / d
	return Vector3(g.dot(frame.x) * r_planet, h, g.dot(frame.z) * r_planet)


static func arc_distance(a: Vector3, b: Vector3) -> float:
	return acos(clampf(a.dot(b), -1.0, 1.0))


## Rotate unit tangent `f` about axis `p` by angle (Rodrigues).
static func rotate_about(f: Vector3, p: Vector3, angle: float) -> Vector3:
	return f.rotated(p.normalized(), angle)

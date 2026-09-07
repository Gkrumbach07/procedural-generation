## Streams baked tiles around the player and renders them as flat local
## terrain in an anchor-based gnomonic frame (PLAN 12.2 / 12.3).
class_name WorldRoot
extends Node3D

signal anchor_changed(anchor: Vector3)
signal tile_ready(key: String)
signal streaming_settled

@export var world_dir: String = ""          ## empty -> ProjectSettings "globe/world_dir"
@export var view_distance_m: float = 8000.0
@export var lod_factor: float = 2.5         ## refine a tile while player distance < lod_factor * tile edge
@export var reanchor_distance_tiles: float = 1.0
@export var loads_per_frame: int = 3
@export var physics_radius_m: float = 500.0
@export var stream_interval: float = 0.15
@export var sync_loads: bool = false        ## decode on the main thread (tests / debugging)
@export var render_water: bool = true       ## lake meshes (water.png)

var manifest := {}
var index := {}
var N_c := 0
var N_fine := 0
var T := 0
var max_lod := 0
var R_planet := 1.0
var anchor := Vector3(0, 0, 1)
var frame := Basis()
var tiles := {}          ## key -> TerrainTile
var loading := {}        ## key -> true while a load task is running
var desired := {}        ## key -> [lod, face, x, y]
var source := TileSource.new()
var terrain_shader: Shader = preload("res://world/terrain.gdshader")
var water_shader: Shader = preload("res://world/water.gdshader")
var ocean_shader: Shader = preload("res://world/ocean.gdshader")
var grid_mesh: ArrayMesh
var water_mesh: ArrayMesh
var ocean: MeshInstance3D
var player: Node3D
var drainage: Object = null   ## DrainageGraph (extension) when available
var _since_stream := 0.0
var _pending_results: Array = []
var _mutex := Mutex.new()
var loaded_ok := false


func _ready() -> void:
	if world_dir == "":
		world_dir = ProjectSettings.get_setting("globe/world_dir", "res://data/worlds/demo")
	world_dir = ProjectSettings.globalize_path(world_dir) if world_dir.begins_with("res://") or world_dir.begins_with("user://") else world_dir
	player = get_node_or_null("Player")
	if not load_world():
		push_error("WorldRoot: no baked world at %s (run bake.py and copy/symlink it into game/data/worlds)" % world_dir)
		return
	if player and player.has_method("place_on_sphere") and not player.get("placed"):
		player.place_on_sphere(GlobeMath.to_sphere(4, 0.5, 0.5))
	set_anchor(player.sphere_pos if player else GlobeMath.to_sphere(4, 0.5, 0.5))
	_make_ocean()
	stream_now()


func load_world() -> bool:
	manifest = TileSource.read_json(world_dir + "/manifest.json")
	if manifest.is_empty():
		return false
	index = TileSource.read_json(world_dir + "/tiles/index.json")
	N_c = int(manifest.get("N_c", 0))
	N_fine = int(manifest.get("N_fine", 0))
	T = int(manifest.get("T", 0))
	max_lod = int(manifest.get("max_lod", 0))
	R_planet = float(manifest.get("R_planet", GlobeMath.planet_radius(N_c, float(manifest.get("cell_size_m", 50.0)))))
	if N_fine <= 0 or T <= 0:
		return false
	grid_mesh = TileSource.build_grid_mesh(T + 1, true)
	water_mesh = TileSource.build_grid_mesh(T + 1, false)
	RenderingServer.global_shader_parameter_set("globe_r_planet", R_planet)
	if ClassDB.class_exists("DrainageGraph"):
		drainage = ClassDB.instantiate("DrainageGraph")
		if not drainage.call("load", world_dir):
			drainage = null
	loaded_ok = true
	print("WorldRoot: %s  N_c=%d N_fine=%d T=%d LODs=0..%d R=%.0f m  extension=%s" % [world_dir, N_c, N_fine, T, max_lod, R_planet, str(source.has_extension())])
	return true


# ---------------------------------------------------------------- geometry
func tiles_per_face(lod: int) -> int:
	return maxi(1, (N_fine / T) >> lod)


## Arc length of a tile edge in metres (face edge = quarter circumference).
func tile_edge_m(lod: int) -> float:
	return (PI * 0.5 * R_planet) / float(tiles_per_face(lod))


func tile_key(lod: int, face: int, x: int, y: int) -> String:
	return "L%d/f%d/%d_%d" % [lod, face, x, y]


func tile_center(face: int, lod: int, x: int, y: int) -> Vector3:
	var n := float(tiles_per_face(lod))
	return GlobeMath.to_sphere(face, (x + 0.5) / n, (y + 0.5) / n)


## Local (anchor-frame) position of vertex (i, j) of a tile at height h.
func tile_local_pos(face: int, lod: int, x: int, y: int, i: float, j: float, h: float) -> Vector3:
	var scale := float(1 << lod)
	var u := (float(x * T) + i) * scale / float(N_fine)
	var v := (float(y * T) + j) * scale / float(N_fine)
	return GlobeMath.gnomonic_local(GlobeMath.to_sphere(face, u, v), h, frame, R_planet)


func to_local_pos(p: Vector3, h: float) -> Vector3:
	return GlobeMath.gnomonic_local(p, h, frame, R_planet)


## Local-frame direction of a unit tangent vector f at sphere point p.
func local_dir(p: Vector3, f: Vector3) -> Vector3:
	var eps := 1e-4
	var a := to_local_pos(p, 0.0)
	var b := to_local_pos((p + f * eps).normalized(), 0.0)
	var d := b - a
	d.y = 0.0
	return d.normalized()


func set_anchor(a: Vector3) -> void:
	anchor = a.normalized()
	frame = GlobeMath.tangent_frame(anchor)
	RenderingServer.global_shader_parameter_set("globe_e1", frame.x)
	RenderingServer.global_shader_parameter_set("globe_n", frame.y)
	RenderingServer.global_shader_parameter_set("globe_e2", frame.z)
	for t in tiles.values():
		if t.body:
			t.update_collision(self, true)
	anchor_changed.emit(anchor)


func _maybe_reanchor() -> void:
	if not player:
		return
	var d := GlobeMath.arc_distance(player.sphere_pos, anchor) * R_planet
	if d > reanchor_distance_tiles * tile_edge_m(0):
		set_anchor(player.sphere_pos)


# ---------------------------------------------------------------- streaming
func _process(dt: float) -> void:
	if not loaded_ok:
		return
	_drain_results()
	_since_stream += dt
	if _since_stream >= stream_interval:
		_since_stream = 0.0
		_maybe_reanchor()
		_update_desired()
		_apply_desired()
	if ocean and player:
		var p := to_local_pos(player.sphere_pos, 0.0)
		ocean.position = Vector3(p.x, 0.0, p.z)


## Synchronous streaming pass (loads everything desired before returning).
func stream_now() -> void:
	_maybe_reanchor()
	_update_desired()
	var was_sync := sync_loads
	sync_loads = true
	_apply_desired(true)
	sync_loads = was_sync
	_drain_results()
	_apply_desired(true)  # second pass releases tiles now covered by finer ones
	streaming_settled.emit()


func _player_pos() -> Vector3:
	return player.sphere_pos if player else anchor


func _update_desired() -> void:
	desired.clear()
	var p := _player_pos()
	for f in range(6):
		_descend(f, max_lod, 0, 0, p)


func _descend(face: int, lod: int, x: int, y: int, p: Vector3) -> void:
	var edge := tile_edge_m(lod)
	var dist := GlobeMath.arc_distance(tile_center(face, lod, x, y), p) * R_planet - edge * 0.75
	if dist > view_distance_m:
		return
	if lod > 0:
		# hysteresis: keep refined children if they are already loaded
		var thresh := lod_factor * edge * 0.5
		var children_loaded := 0
		for dx in range(2):
			for dy in range(2):
				if tiles.has(tile_key(lod - 1, face, 2 * x + dx, 2 * y + dy)):
					children_loaded += 1
		if children_loaded == 4:
			thresh *= 1.2
		if dist < thresh:
			for dx in range(2):
				for dy in range(2):
					_descend(face, lod - 1, 2 * x + dx, 2 * y + dy, p)
			return
	desired[tile_key(lod, face, x, y)] = [lod, face, x, y]


func _apply_desired(all_at_once: bool = false) -> void:
	var p := _player_pos()
	# queue missing tiles, nearest first
	var missing: Array = []
	for key in desired:
		if not tiles.has(key) and not loading.has(key):
			var d: Array = desired[key]
			missing.append([GlobeMath.arc_distance(tile_center(d[1], d[0], d[2], d[3]), p), key])
	missing.sort()
	var budget := missing.size() if all_at_once else loads_per_frame
	for k in range(mini(budget, missing.size())):
		_request(missing[k][1])
	# release tiles that are no longer desired and fully covered by loaded desired tiles
	for key in tiles.keys():
		if desired.has(key):
			continue
		var t: TerrainTile = tiles[key]
		if _covered(t):
			tiles.erase(key)
			t.queue_free()
	_update_collisions()


## True when every desired tile overlapping `t` is loaded (so `t` can go).
func _covered(t: TerrainTile) -> bool:
	var a0 := t.tx * T * (1 << t.lod)
	var a1 := (t.tx + 1) * T * (1 << t.lod)
	var b0 := t.ty * T * (1 << t.lod)
	var b1 := (t.ty + 1) * T * (1 << t.lod)
	for key in desired:
		var d: Array = desired[key]
		if d[1] != t.face:
			continue
		var s: int = T * (1 << d[0])
		var c0: int = d[2] * s
		var c1: int = (d[2] + 1) * s
		var e0: int = d[3] * s
		var e1: int = (d[3] + 1) * s
		if c0 < a1 and c1 > a0 and e0 < b1 and e1 > b0:
			if not tiles.has(key):
				return false
	return true


func _request(key: String) -> void:
	var d: Array = desired[key]
	loading[key] = true
	if sync_loads:
		_on_loaded(source.load_tile(world_dir, d[0], d[1], d[2], d[3]), key)
	else:
		WorkerThreadPool.add_task(_load_task.bind(key, d))


func _load_task(key: String, d: Array) -> void:
	var data := source.load_tile(world_dir, d[0], d[1], d[2], d[3])
	_mutex.lock()
	_pending_results.append([key, data])
	_mutex.unlock()


func _drain_results() -> void:
	_mutex.lock()
	var results := _pending_results
	_pending_results = []
	_mutex.unlock()
	for r in results:
		_on_loaded(r[1], r[0])


func _on_loaded(data: Dictionary, key: String) -> void:
	loading.erase(key)
	if not data.get("ok", false):
		push_warning("tile %s failed: %s" % [key, str(data.get("error", "?"))])
		return
	if not desired.has(key) and not sync_loads:
		return  # no longer wanted
	if tiles.has(key):
		return
	var t := TerrainTile.new()
	t.setup(data, self, terrain_shader, water_shader, grid_mesh, water_mesh)
	tiles[key] = t
	add_child(t)
	tile_ready.emit(key)


func _update_collisions() -> void:
	var p := _player_pos()
	for t in tiles.values():
		var want: bool = t.lod == 0 and GlobeMath.arc_distance(tile_center(t.face, 0, t.tx, t.ty), p) * R_planet < physics_radius_m + tile_edge_m(0)
		if want and not t.body:
			t.update_collision(self, true)
		elif not want and t.body:
			t.update_collision(self, false)


# ---------------------------------------------------------------- queries
## Finest loaded tile containing sphere point p -> [tile, fi, fj] or [].
func tile_at(p: Vector3) -> Array:
	var fuv := GlobeMath.from_sphere(p)
	var face: int = fuv[0]
	for lod in range(0, max_lod + 1):
		var xy := GlobeMath.tile_of(fuv[1], fuv[2], lod, N_fine, T)
		var key := tile_key(lod, face, xy.x, xy.y)
		if tiles.has(key):
			var t: TerrainTile = tiles[key]
			var n := float(tiles_per_face(lod))
			var fi: float = (fuv[1] * n - xy.x) * T
			var fj: float = (fuv[2] * n - xy.y) * T
			return [t, fi, fj]
	return []


func sample_height(p: Vector3) -> float:
	var r := tile_at(p)
	if r.is_empty():
		return 0.0
	return r[0].sample_height(r[1], r[2])


func sample_water(p: Vector3) -> float:
	var r := tile_at(p)
	if r.is_empty():
		return 0.0
	return r[0].water_surface_at(r[1], r[2])


func basin_at(p: Vector3) -> int:
	if drainage:
		var fuv := GlobeMath.from_sphere(p)
		return drainage.call("basin_of", fuv[0], fuv[1], fuv[2])
	var r := tile_at(p)
	if r.is_empty():
		return -1
	var t: TerrainTile = r[0]
	var basins: Array = t.meta.get("basins", [])
	return basins.size()  # placeholder count without the extension


func _make_ocean() -> void:
	var pm := PlaneMesh.new()
	pm.size = Vector2(view_distance_m * 2.4, view_distance_m * 2.4)
	ocean = MeshInstance3D.new()
	ocean.mesh = pm
	var m := ShaderMaterial.new()
	m.shader = ocean_shader
	ocean.material_override = m
	add_child(ocean)

## Headless streaming test (PLAN 15): stream a ring of tiles around a point on
## a cube edge and assert nothing is missing.
##   godot --headless --path game -s tests/test_stream.gd -- --world=/abs/path/to/world
extends SceneTree

var failures := 0


func check(cond: bool, msg: String) -> void:
	if not cond:
		failures += 1
		printerr("FAIL: " + msg)


func _init() -> void:
	# nodes added during _init become ready on the first frame; run then
	process_frame.connect(_run, CONNECT_ONE_SHOT)


func _run() -> void:
	var world_dir := ""
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--world="):
			world_dir = a.substr(8)
	if world_dir == "":
		world_dir = ProjectSettings.globalize_path("res://data/worlds/demo")
	var scene: PackedScene = load("res://world/WorldRoot.tscn")
	var world: WorldRoot = scene.instantiate()
	world.world_dir = world_dir
	world.sync_loads = true
	world.view_distance_m = 1e9  # keep everything at some LOD
	var player: GlobePlayer = world.get_node("Player")
	# a point on the edge between +Z (face 4) and +X (face 0), slightly inside +Z
	var p := GlobeMath.to_sphere(4, 0.999, 0.5)
	player.place_on_sphere(p)
	root.add_child(world)
	await process_frame
	check(world.loaded_ok, "world loaded from " + world_dir)
	if not world.loaded_ok:
		quit(1)
		return
	world.stream_now()
	print("tiles loaded: %d (desired %d), extension=%s" % [world.tiles.size(), world.desired.size(), str(world.source.has_extension())])
	check(world.tiles.size() == world.desired.size(), "all desired tiles loaded")
	# ring of sample points around the player (half a LOD-0 tile away) must hit loaded tiles at LOD 0
	var half := world.tile_edge_m(0) * 0.5 / world.R_planet
	var frame := GlobeMath.tangent_frame(p)
	var faces := {}
	for k in range(8):
		var ang := k * PI / 4.0
		var q := (p + (frame.x * cos(ang) + frame.z * sin(ang)) * half).normalized()
		var r := world.tile_at(q)
		check(not r.is_empty(), "ring point %d has a tile" % k)
		if not r.is_empty():
			check(r[0].lod == 0, "ring point %d is at LOD 0 (got %d)" % [k, r[0].lod])
			faces[r[0].face] = true
			var h: float = world.sample_height(q)
			check(is_finite(h), "height finite at ring point %d" % k)
	check(faces.has(4) and faces.has(0), "ring spans both faces of the edge (faces %s)" % str(faces.keys()))
	# shared-column invariant across the face edge: last column of face 4's edge tile
	# equals first column of face 0's edge tile (vertex samples coincide by construction)
	var n0 := world.tiles_per_face(0)
	var t4: TerrainTile = world.tiles.get(world.tile_key(0, 4, n0 - 1, n0 / 2))
	var t0: TerrainTile = world.tiles.get(world.tile_key(0, 0, 0, n0 / 2))
	check(t4 != null and t0 != null, "edge tiles on both faces loaded")
	if t4 and t0:
		var maxd := 0.0
		for j in range(t4.size):
			maxd = maxf(maxd, absf(t4.height[j * t4.size + (t4.size - 1)] - t0.height[j * t0.size + 0]))
		var span: float = maxf(t4.meta.get("height_max", 1.0) - t4.meta.get("height_min", 0.0), 1.0)
		print("cross-face shared column max diff: %.3f m (tile span %.1f m)" % [maxd, span])
		check(maxd < span * 0.03 + 0.5, "cross-face shared column agrees")
	# walk around the planet in a straight line and come back
	var start := player.sphere_pos
	var steps := 400
	var step_len := 2.0 * PI / steps
	for s in range(steps):
		var np := (player.sphere_pos + player.forward * step_len).normalized()
		player.forward = (player.forward - np * np.dot(player.forward)).normalized()
		player.sphere_pos = np
		if s % 25 == 0:
			world.stream_now()
			check(world.tiles.size() == world.desired.size(), "tiles complete during walk step %d" % s)
			check(not world.tile_at(player.sphere_pos).is_empty(), "player stands on a tile at step %d" % s)
	check(GlobeMath.arc_distance(player.sphere_pos, start) < 0.02, "returned to start after a full circle (dist %.4f rad)" % GlobeMath.arc_distance(player.sphere_pos, start))
	# gnomonic sanity: anchor maps to origin
	var o := world.to_local_pos(world.anchor, 0.0)
	check(o.length() < 1e-3, "anchor at local origin")
	# collision: HeightMapShape3D bodies near the player must agree with sample_height
	player.place_on_sphere(p)
	world.set_anchor(p)
	world.stream_now()
	await physics_frame
	await physics_frame
	var space := world.get_world_3d().direct_space_state
	var n_bodies := 0
	for t in world.tiles.values():
		if t.body:
			n_bodies += 1
	check(n_bodies > 0, "collision bodies exist near the player (%d)" % n_bodies)
	var worst := 0.0
	var hits := 0
	for k in range(12):
		var ang := k * PI / 6.0
		var q := (p + (frame.x * cos(ang) + frame.z * sin(ang)) * (120.0 / world.R_planet)).normalized()
		var h := world.sample_height(q)
		var lp := world.to_local_pos(q, h)
		var query := PhysicsRayQueryParameters3D.create(lp + Vector3(0, 2000, 0), lp - Vector3(0, 2000, 0))
		var hit := space.intersect_ray(query)
		if hit.is_empty():
			continue
		hits += 1
		worst = maxf(worst, absf(hit.position.y - h))
	print("collision: %d bodies, %d/12 ray hits, worst |ray - sample_height| = %.2f m" % [n_bodies, hits, worst])
	check(hits >= 10, "rays hit the collision heightmap")
	check(worst < 8.0, "collision surface matches rendered heights (worst %.2f m)" % worst)
	print("test_stream: %d failure(s)" % failures)
	quit(1 if failures > 0 else 0)

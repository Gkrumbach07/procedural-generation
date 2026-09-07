extends SceneTree
# Probe: ray hits vs sample_height at tile corners / centre to diagnose HeightMapShape3D placement.
func _init() -> void:
	process_frame.connect(_run, CONNECT_ONE_SHOT)

func _run() -> void:
	var world_dir := ""
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--world="):
			world_dir = a.substr(8)
	var world: WorldRoot = load("res://world/WorldRoot.tscn").instantiate()
	world.world_dir = world_dir
	world.sync_loads = true
	var player: GlobePlayer = world.get_node("Player")
	var p := GlobeMath.to_sphere(4, 0.6, 0.6)
	player.place_on_sphere(p)
	root.add_child(world)
	await process_frame
	world.stream_now()
	await physics_frame
	await physics_frame
	var space := world.get_world_3d().direct_space_state
	var r := world.tile_at(p)
	var t: TerrainTile = r[0]
	print("tile %s body=%s" % [t.key, str(t.body != null)])
	var n := t.size
	for pt in [[0, 0], [n - 1, 0], [0, n - 1], [n - 1, n - 1], [n / 2, n / 2], [10, 40], [40, 10]]:
		var i: int = pt[0]
		var j: int = pt[1]
		var h: float = t.height[j * n + i]
		var lp: Vector3 = world.tile_local_pos(t.face, t.lod, t.tx, t.ty, float(i), float(j), h)
		var q := PhysicsRayQueryParameters3D.create(lp + Vector3(0, 3000, 0), lp - Vector3(0, 3000, 0))
		var hit := space.intersect_ray(q)
		var hy: float = hit.get("position", Vector3(0, NAN, 0)).y
		# candidate mirrored values
		var h_mi: float = t.height[j * n + (n - 1 - i)]
		var h_mj: float = t.height[(n - 1 - j) * n + i]
		var h_t: float = t.height[i * n + j]
		print("(%2d,%2d) h=%8.1f hit=%8.1f | mirror_i=%8.1f mirror_j=%8.1f transpose=%8.1f  xz=(%.0f, %.0f)" % [i, j, h, hy, h_mi, h_mj, h_t, lp.x, lp.z])
	quit(0)

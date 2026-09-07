## Render a few frames of the world from a given spot and save a screenshot.
## --along=u|-u|v|-v sets the heading along a face axis (e.g. to look across an edge).
##   xvfb-run godot --rendering-driver opengl3 --path game -s tests/screenshot.gd -- --world=/abs/world --out=/abs/shot.png [--face=4 --u=0.999 --v=0.5 --alt=300 --pitch=-0.5 --debug=0]
extends SceneTree

var world: WorldRoot
var out_path := "screenshot.png"
var frames := 0


func _arg(name: String, default: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--" + name + "="):
			return a.substr(name.length() + 3)
	return default


func _init() -> void:
	process_frame.connect(_start, CONNECT_ONE_SHOT)


func _start() -> void:
	var world_dir := _arg("world", ProjectSettings.globalize_path("res://data/worlds/demo"))
	out_path = _arg("out", "screenshot.png")
	var scene: PackedScene = load("res://world/WorldRoot.tscn")
	world = scene.instantiate()
	world.world_dir = world_dir
	world.sync_loads = true
	world.render_water = _arg("water", "1") != "0"
	var player: GlobePlayer = world.get_node("Player")
	var face := int(_arg("face", "4"))
	var u := float(_arg("u", "0.999"))
	var v := float(_arg("v", "0.5"))
	var heading := Vector3.ZERO
	var along := _arg("along", "")
	if along != "":
		var jac := GlobeMath.jacobian(face, u, v)
		heading = jac.x if along == "u" else (-jac.x if along == "-u" else (jac.y if along == "v" else -jac.y))
	player.place_on_sphere(GlobeMath.to_sphere(face, u, v), heading)
	player.altitude = float(_arg("alt", "300"))
	player.pitch = float(_arg("pitch", "-0.5"))
	root.add_child(world)
	await process_frame
	RenderingServer.global_shader_parameter_set("globe_debug_mode", int(_arg("debug", "0")))
	world.stream_now()
	player._update_camera()
	print("screenshot: %d tiles, camera at %s" % [world.tiles.size(), str(player.global_position)])
	process_frame.connect(_tick)


func _tick() -> void:
	frames += 1
	if frames == 8:
		var img := root.get_viewport().get_texture().get_image()
		img.save_png(out_path)
		print("saved " + out_path + " " + str(img.get_size()))
		quit(0)

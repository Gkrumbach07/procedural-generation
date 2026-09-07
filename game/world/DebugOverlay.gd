## F3 cycles debug modes: 0 off, 1 basin tint, 2 discharge, 3 tile bounds.
## Shows a minimap of the unfolded cube net with the player, anchor and loaded
## tiles, and text stats (PLAN 12.4).
class_name DebugOverlay
extends CanvasLayer

var world: WorldRoot
var mode := 0
var panel: Control
var label: Label
const CELL := 70


func _ready() -> void:
	world = get_parent() as WorldRoot
	panel = Control.new()
	panel.set_anchors_preset(Control.PRESET_FULL_RECT)
	panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.draw.connect(_draw_panel)
	add_child(panel)
	label = Label.new()
	label.position = Vector2(12, 12)
	label.add_theme_color_override("font_color", Color(1, 1, 1))
	label.add_theme_color_override("font_shadow_color", Color(0, 0, 0))
	add_child(label)
	visible = false


func _unhandled_input(event: InputEvent) -> void:
	if event.is_action_pressed("toggle_debug"):
		mode = (mode + 1) % 4
		RenderingServer.global_shader_parameter_set("globe_debug_mode", mode)
		visible = mode > 0


func _process(_dt: float) -> void:
	if not visible or world == null or not world.loaded_ok:
		return
	panel.queue_redraw()
	var p: Vector3 = world.player.sphere_pos if world.player else world.anchor
	var fuv := GlobeMath.from_sphere(p)
	var basin := world.basin_at(p)
	var reach := ""
	if world.drainage:
		var r: Dictionary = world.drainage.call("nearest_reach", p, 6)
		if not r.is_empty():
			reach = "reach %d order %d (%.1f cells)" % [r.get("edge_id", -1), r.get("order", 0), r.get("distance_cells", 0.0)]
	label.text = "mode %d (F3)  fps %d\nface %d u %.4f v %.4f  h %.1f m\nanchor dist %.0f m  tiles %d loading %d desired %d\nbasin %d  %s\n%s" % [
		mode, Engine.get_frames_per_second(), fuv[0], fuv[1], fuv[2], world.sample_height(p),
		GlobeMath.arc_distance(p, world.anchor) * world.R_planet, world.tiles.size(), world.loading.size(), world.desired.size(),
		basin, reach, "fly" if (world.player and world.player.fly) else "walk"]


func _draw_panel() -> void:
	if world == null or not world.loaded_ok:
		return
	var origin := Vector2(panel.size.x - 4 * CELL - 16, 16)
	# net background
	for f in range(6):
		var slot: Vector2i = GlobeMath.NET_SLOTS[f]
		var r := Rect2(origin + Vector2(slot.y * CELL, slot.x * CELL), Vector2(CELL, CELL))
		panel.draw_rect(r, Color(0.1, 0.1, 0.15, 0.8))
		panel.draw_rect(r, Color(0.7, 0.7, 0.7), false, 1.0)
	# loaded tiles
	for t in world.tiles.values():
		var slot: Vector2i = GlobeMath.NET_SLOTS[t.face]
		var n := float(world.tiles_per_face(t.lod))
		var r := Rect2(origin + Vector2(slot.y * CELL + t.tx / n * CELL, slot.x * CELL + t.ty / n * CELL), Vector2(CELL / n, CELL / n))
		var c := Color.from_hsv(fmod(t.lod * 0.17, 1.0), 0.6, 0.9, 0.5)
		panel.draw_rect(r, c)
	# player + anchor
	for pair in [[world.anchor, Color(1, 0.8, 0.2)], [world.player.sphere_pos if world.player else world.anchor, Color(1, 0.2, 0.2)]]:
		var fuv := GlobeMath.from_sphere(pair[0])
		var slot: Vector2i = GlobeMath.NET_SLOTS[fuv[0]]
		var pt := origin + Vector2(slot.y * CELL + fuv[1] * CELL, slot.x * CELL + fuv[2] * CELL)
		panel.draw_circle(pt, 3.0, pair[1])

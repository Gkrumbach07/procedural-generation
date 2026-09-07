## Draws drainage-graph reaches near the player as 3-D line strips (debug
## overlay, PLAN 12.4).  Needs the DrainageGraph extension; silent otherwise.
class_name FlowLines
extends MeshInstance3D

var world: WorldRoot
var _mesh := ImmediateMesh.new()
var _timer := 0.0
var enabled := false


func _ready() -> void:
	world = get_parent() as WorldRoot
	mesh = _mesh
	var m := StandardMaterial3D.new()
	m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	m.vertex_color_use_as_albedo = true
	m.no_depth_test = false
	material_override = m
	custom_aabb = AABB(Vector3(-1e6, -1e5, -1e6), Vector3(2e6, 2e5, 2e6))


func _process(dt: float) -> void:
	if world == null or not world.loaded_ok or world.drainage == null:
		return
	visible = enabled
	if not enabled:
		return
	_timer += dt
	if _timer > 1.0:
		_timer = 0.0
		rebuild()


func rebuild() -> void:
	_mesh.clear_surfaces()
	var p: Vector3 = world.player.sphere_pos if world.player else world.anchor
	var n_edges: int = world.drainage.call("edge_count")
	if n_edges == 0:
		return
	var max_arc: float = world.view_distance_m / world.R_planet
	var any := false
	for e in range(n_edges):
		var ed: Dictionary = world.drainage.call("edge", e)
		var cells: Array = ed.get("cells", [])
		if cells.size() < 2:
			continue
		var c0: Vector3 = world.drainage.call("cell_center", cells[0])
		if GlobeMath.arc_distance(c0, p) > max_arc:
			continue
		var order: int = ed.get("order", 1)
		var col := Color.from_hsv(0.6 - 0.1 * order, 0.9, 1.0)
		if not any:
			_mesh.surface_begin(Mesh.PRIMITIVE_LINES)
			any = true
		var prev := Vector3.ZERO
		for k in range(cells.size()):
			var q: Vector3 = world.drainage.call("cell_center", cells[k])
			var h: float = maxf(world.sample_height(q), world.sample_water(q)) + 3.0
			var lp := world.to_local_pos(q, h)
			if k > 0:
				_mesh.surface_set_color(col)
				_mesh.surface_add_vertex(prev)
				_mesh.surface_set_color(col)
				_mesh.surface_add_vertex(lp)
			prev = lp
	if any:
		_mesh.surface_end()

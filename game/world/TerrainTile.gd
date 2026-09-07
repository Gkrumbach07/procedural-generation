## One streamed tile: terrain mesh + optional lake mesh + optional collision.
class_name TerrainTile
extends Node3D

var key: String
var lod: int
var face: int
var tx: int
var ty: int
var size: int
var height: PackedFloat32Array
var water: PackedFloat32Array
var meta: Dictionary
var has_water := false
var mesh_instance: MeshInstance3D
var water_instance: MeshInstance3D
var body: StaticBody3D
var height_tex: ImageTexture


func setup(data: Dictionary, world: Node, terrain_shader: Shader, water_shader: Shader, grid_mesh: ArrayMesh, water_mesh: ArrayMesh) -> void:
	lod = data["lod"]
	face = data["face"]
	tx = data["x"]
	ty = data["y"]
	size = data["size"]
	height = data["height"]
	water = data["water"]
	meta = data["meta"]
	key = "L%d/f%d/%d_%d" % [lod, face, tx, ty]
	name = key.replace("/", "_")
	var himg := Image.create_from_data(size, size, false, Image.FORMAT_RF, height.to_byte_array())
	height_tex = ImageTexture.create_from_image(himg)
	var layers_img: Image = data["layers"]
	var flow_img: Image = data["flow"]
	var layers_tex := ImageTexture.create_from_image(layers_img) if layers_img else null
	var flow_tex := ImageTexture.create_from_image(flow_img) if flow_img else null
	var mat := ShaderMaterial.new()
	mat.shader = terrain_shader
	mat.set_shader_parameter("height_tex", height_tex)
	if layers_tex:
		mat.set_shader_parameter("layers_tex", layers_tex)
	if flow_tex:
		mat.set_shader_parameter("flow_tex", flow_tex)
	mat.set_shader_parameter("face", face)
	mat.set_shader_parameter("tile_x", tx)
	mat.set_shader_parameter("tile_y", ty)
	mat.set_shader_parameter("lod", lod)
	mat.set_shader_parameter("n_fine", world.N_fine)
	mat.set_shader_parameter("tile_size", world.T)
	mat.set_shader_parameter("skirt_drop", maxf(20.0, world.tile_edge_m(lod) * 0.05) * world.skirt_drop_scale)
	mat.set_shader_parameter("tile_seed", float((face * 7919 + tx * 131 + ty) % 1000))
	mesh_instance = MeshInstance3D.new()
	mesh_instance.mesh = grid_mesh
	mesh_instance.material_override = mat
	# the shader places vertices in world space directly; disable frustum culling
	# on the flat proxy AABB by giving it a huge custom AABB
	mesh_instance.custom_aabb = AABB(Vector3(-1e6, -1e5, -1e6), Vector3(2e6, 2e5, 2e6))
	add_child(mesh_instance)
	has_water = bool(meta.get("has_water", false))
	if has_water and water.size() == height.size() and world.render_water:
		var wimg := Image.create_from_data(size, size, false, Image.FORMAT_RF, water.to_byte_array())
		var wmat := ShaderMaterial.new()
		wmat.shader = water_shader
		wmat.set_shader_parameter("height_tex", height_tex)
		wmat.set_shader_parameter("water_tex", ImageTexture.create_from_image(wimg))
		for k in ["face", "tile_x", "tile_y", "lod", "n_fine", "tile_size"]:
			wmat.set_shader_parameter(k, mat.get_shader_parameter(k))
		water_instance = MeshInstance3D.new()
		water_instance.mesh = water_mesh
		water_instance.material_override = wmat
		water_instance.custom_aabb = mesh_instance.custom_aabb
		add_child(water_instance)


## Height at fractional vertex coordinates (fi, fj) in 0..size-1 (bilinear).
func sample_height(fi: float, fj: float) -> float:
	var i0 := clampi(int(floor(fi)), 0, size - 2)
	var j0 := clampi(int(floor(fj)), 0, size - 2)
	var wi := clampf(fi - i0, 0.0, 1.0)
	var wj := clampf(fj - j0, 0.0, 1.0)
	var h00 := height[j0 * size + i0]
	var h10 := height[j0 * size + i0 + 1]
	var h01 := height[(j0 + 1) * size + i0]
	var h11 := height[(j0 + 1) * size + i0 + 1]
	return (h00 * (1 - wi) + h10 * wi) * (1 - wj) + (h01 * (1 - wi) + h11 * wi) * wj


func water_surface_at(fi: float, fj: float) -> float:
	if not has_water or water.size() != height.size():
		return 0.0
	var i0 := clampi(int(round(fi)), 0, size - 1)
	var j0 := clampi(int(round(fj)), 0, size - 1)
	return water[j0 * size + i0]


## (Re)build the collision shape in the current anchor frame (PLAN 12.3:
## HeightMapShape3D only for LOD-0 tiles near the player).  The gnomonic
## tile is very slightly non-square; we place the shape with the tile's
## projected axes (error ~ (d / R)^2, negligible within physics_radius).
func update_collision(world: Node, enabled: bool) -> void:
	if not enabled:
		if body:
			body.queue_free()
			body = null
		return
	if body:
		body.queue_free()
	body = StaticBody3D.new()
	var shape := HeightMapShape3D.new()
	shape.map_width = size
	shape.map_depth = size
	# HeightMapShape3D data is indexed [z * width + x]; our height is [j * size + i]
	shape.map_data = height
	var cs := CollisionShape3D.new()
	cs.shape = shape
	var p00: Vector3 = world.tile_local_pos(face, lod, tx, ty, 0.0, 0.0, 0.0)
	var p10: Vector3 = world.tile_local_pos(face, lod, tx, ty, float(size - 1), 0.0, 0.0)
	var p01: Vector3 = world.tile_local_pos(face, lod, tx, ty, 0.0, float(size - 1), 0.0)
	var ax := (p10 - p00) / float(size - 1)
	var az := (p01 - p00) / float(size - 1)
	var basis := Basis(ax, Vector3(0, 1, 0), az)
	# HeightMapShape3D is centred: shift by half extents in local shape space
	var center := p00 + ax * float(size - 1) * 0.5 + az * float(size - 1) * 0.5
	cs.transform = Transform3D(basis, center)
	body.add_child(cs)
	add_child(body)

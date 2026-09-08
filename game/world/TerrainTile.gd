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
var _shape: ConcavePolygonShape3D          ## rebuilt in place on re-anchor
var _dirs: PackedVector3Array              ## unit-sphere direction per vertex (GDScript collision path)

## CubeSphere.build_collision_faces (GDExtension) when available.
static var _has_cubesphere: bool = ClassDB.class_exists("CubeSphere")


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
	# debug mode 1 tints by *global* basin id: upload the tile's local index
	# (flow.G) -> id map from meta["basins"] (255 / unlisted stays -1)
	var bids := PackedInt32Array()
	bids.resize(256)
	bids.fill(-1)
	var blist: Array = meta.get("basins", [])
	for bi in range(mini(blist.size(), 255)):
		bids[bi] = int(blist[bi])
	mat.set_shader_parameter("basin_ids", bids)
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


## (Re)build the collision shape in the current anchor frame (only LOD-0
## tiles near the player, PLAN 12.3).  PLAN suggests HeightMapShape3D, but a
## gnomonically projected tile is a (slightly) skewed quadrilateral and
## physics shapes cannot be skewed, so we build an exact triangle mesh from
## the same projected vertices the shader uses (ConcavePolygonShape3D;
## 2*T^2 triangles).  Re-anchoring rebuilds every body in one frame
## (collision must never lag the anchor), so this stays on the fast path:
## the faces come from the GDExtension when it is loaded, the body and its
## shape are reused in place, and the GDScript fallback caches the
## anchor-independent part of the projection.
func update_collision(world: Node, enabled: bool) -> void:
	if not enabled:
		if body:
			body.queue_free()
			body = null
			_shape = null
		return
	var faces := _collision_faces(world)
	if faces.is_empty():
		return
	if body == null:
		_shape = ConcavePolygonShape3D.new()
		_shape.backface_collision = true
		_shape.set_faces(faces)
		var cs := CollisionShape3D.new()
		cs.shape = _shape
		body = StaticBody3D.new()
		body.add_child(cs)
		add_child(body)
	else:
		_shape.set_faces(faces)


## Triangle soup of this tile's vertices in the current anchor frame.
func _collision_faces(world: Node) -> PackedVector3Array:
	if _has_cubesphere:
		return ClassDB.class_call_static("CubeSphere", "build_collision_faces",
			face, lod, tx, ty, world.T, world.N_fine, world.R_planet, world.frame, height)
	# Fallback: the unit-sphere direction of a vertex does not depend on the
	# anchor, so cache it and redo only the gnomonic step per rebuild.
	if _dirs.size() != size * size:
		_dirs.resize(size * size)
		var scale := float(1 << lod)
		for j in range(size):
			var v := (float(ty * world.T) + j) * scale / float(world.N_fine)
			for i in range(size):
				var u := (float(tx * world.T) + i) * scale / float(world.N_fine)
				_dirs[j * size + i] = GlobeMath.to_sphere(face, u, v)
	var fr: Basis = world.frame
	var rp: float = world.R_planet
	var pts := PackedVector3Array()
	pts.resize(size * size)
	for k in range(size * size):
		var q: Vector3 = _dirs[k]
		var g := q / maxf(q.dot(fr.y), 0.05)
		pts[k] = Vector3(g.dot(fr.x) * rp, height[k], g.dot(fr.z) * rp)
	var faces := PackedVector3Array()
	faces.resize((size - 1) * (size - 1) * 6)
	var n := 0
	for j in range(size - 1):
		for i in range(size - 1):
			var v00 := pts[j * size + i]
			var v10 := pts[j * size + i + 1]
			var v01 := pts[(j + 1) * size + i]
			var v11 := pts[(j + 1) * size + i + 1]
			faces[n] = v00; faces[n + 1] = v01; faces[n + 2] = v10
			faces[n + 3] = v10; faces[n + 4] = v01; faces[n + 5] = v11
			n += 6
	return faces

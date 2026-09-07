## Loads tiles.  Uses the GDExtension TileLoader when it is available (fast
## C++ decode) and falls back to a pure-GDScript 16-bit PNG decoder otherwise
## (slow but complete, so the project runs without compiling the extension).
class_name TileSource
extends RefCounted

var _ext: Object = null


func _init() -> void:
	if ClassDB.class_exists("TileLoader"):
		_ext = ClassDB.instantiate("TileLoader")


func has_extension() -> bool:
	return _ext != null


static func tile_dir(world_dir: String, lod: int, face: int, x: int, y: int) -> String:
	return "%s/tiles/L%d/f%d/%d_%d" % [world_dir, lod, face, x, y]


static func tile_exists(world_dir: String, lod: int, face: int, x: int, y: int) -> bool:
	return FileAccess.file_exists(tile_dir(world_dir, lod, face, x, y) + "/meta.json")


static func read_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var v: Variant = JSON.parse_string(FileAccess.get_file_as_string(path))
	return v if v is Dictionary else {}


## Returns {ok, size, height: PackedFloat32Array [j*size+i], water, layers: Image, flow: Image, meta}
func load_tile(world_dir: String, lod: int, face: int, x: int, y: int) -> Dictionary:
	if _ext != null:
		return _ext.call("load", world_dir, lod, face, x, y)
	return _load_gd(world_dir, lod, face, x, y)


func _load_gd(world_dir: String, lod: int, face: int, x: int, y: int) -> Dictionary:
	var dir := tile_dir(world_dir, lod, face, x, y)
	var meta := read_json(dir + "/meta.json")
	if meta.is_empty():
		return {"ok": false, "error": "missing meta.json in " + dir}
	var height := decode_png16(FileAccess.get_file_as_bytes(dir + "/height.png"), meta.get("height_min", 0.0), meta.get("height_max", 1.0), false)
	if height.is_empty():
		return {"ok": false, "error": "bad height.png in " + dir}
	var water := decode_png16(FileAccess.get_file_as_bytes(dir + "/water.png"), meta.get("water_min", 0.0), meta.get("water_max", 1.0), true)
	var layers := Image.new()
	layers.load_png_from_buffer(FileAccess.get_file_as_bytes(dir + "/layers.png"))
	var flow := Image.new()
	flow.load_png_from_buffer(FileAccess.get_file_as_bytes(dir + "/flow.png"))
	var size: int = int(meta.get("size", int(round(sqrt(float(height.size()))))))
	return {"ok": true, "lod": lod, "face": face, "x": x, "y": y, "size": size, "height": height, "water": water,
		"layers": layers, "flow": flow, "meta": meta, "height_min": meta.get("height_min", 0.0), "height_max": meta.get("height_max", 1.0)}


## Pure-GDScript decoder for 16-bit grayscale non-interlaced PNG.
static func decode_png16(bytes: PackedByteArray, lo: float, hi: float, zero_is_none: bool) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	if bytes.size() < 8 or bytes[0] != 137 or bytes[1] != 80 or bytes[2] != 78 or bytes[3] != 71:
		return out
	var pos := 8
	var width := 0
	var height := 0
	var bit_depth := 0
	var color_type := -1
	var idat := PackedByteArray()
	while pos + 8 <= bytes.size():
		var length := _be32(bytes, pos)
		var type := bytes.slice(pos + 4, pos + 8).get_string_from_ascii()
		var data_start := pos + 8
		if type == "IHDR":
			width = _be32(bytes, data_start)
			height = _be32(bytes, data_start + 4)
			bit_depth = bytes[data_start + 8]
			color_type = bytes[data_start + 9]
			if bytes[data_start + 12] != 0:
				push_error("interlaced PNG unsupported")
				return out
		elif type == "IDAT":
			idat.append_array(bytes.slice(data_start, data_start + length))
		elif type == "IEND":
			break
		pos = data_start + length + 4
	if bit_depth != 16 or color_type != 0:
		push_error("decode_png16: expected 16-bit grayscale")
		return out
	var stride := width * 2
	var raw_size := height * (stride + 1)
	var raw := idat.decompress(raw_size, FileAccess.COMPRESSION_DEFLATE)
	if raw.size() != raw_size:
		push_error("decode_png16: inflate failed")
		return out
	out.resize(width * height)
	var prev := PackedByteArray()
	prev.resize(stride)
	var cur := PackedByteArray()
	cur.resize(stride)
	var span := hi - lo
	for y in range(height):
		var base := y * (stride + 1)
		var filter := raw[base]
		for x in range(stride):
			var a: int = cur[x - 2] if x >= 2 else 0
			var b: int = prev[x]
			var c: int = prev[x - 2] if x >= 2 else 0
			var v: int = raw[base + 1 + x]
			match filter:
				1:
					v += a
				2:
					v += b
				3:
					v += (a + b) >> 1
				4:
					var p := a + b - c
					var pa := absi(p - a)
					var pb := absi(p - b)
					var pc := absi(p - c)
					v += a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
			cur[x] = v & 255
		var row := y * width
		for i in range(width):
			var s: int = (cur[2 * i] << 8) | cur[2 * i + 1]
			if zero_is_none:
				out[row + i] = 0.0 if s == 0 else (float(s - 1) / 65534.0) * span + lo
			else:
				out[row + i] = (float(s) / 65535.0) * span + lo
		var tmp := prev
		prev = cur
		cur = tmp
	return out


static func _be32(b: PackedByteArray, at: int) -> int:
	return (b[at] << 24) | (b[at + 1] << 16) | (b[at + 2] << 8) | b[at + 3]


## Flat grid mesh (size x size vertices at (i, 0, j)) with an optional skirt
## ring (y = -1, clamped grid coords) so the terrain shader can drop it.
static func build_grid_mesh(size: int, skirt: bool) -> ArrayMesh:
	if ClassDB.class_exists("TileLoader"):
		var ext: Object = ClassDB.instantiate("TileLoader")
		var arrays: Array = ext.call("build_mesh_arrays", size, skirt)
		var m := ArrayMesh.new()
		m.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		return m
	var ext_n := size + 2 if skirt else size
	var verts := PackedVector3Array()
	var uvs := PackedVector2Array()
	verts.resize(ext_n * ext_n)
	uvs.resize(ext_n * ext_n)
	for a in range(ext_n):
		for b in range(ext_n):
			var i := a - 1 if skirt else a
			var j := b - 1 if skirt else b
			var is_skirt := skirt and (i < 0 or j < 0 or i >= size or j >= size)
			var ci := clampi(i, 0, size - 1)
			var cj := clampi(j, 0, size - 1)
			verts[a * ext_n + b] = Vector3(ci, -1.0 if is_skirt else 0.0, cj)
			uvs[a * ext_n + b] = Vector2(float(ci) / (size - 1), float(cj) / (size - 1))
	var idx := PackedInt32Array()
	idx.resize((ext_n - 1) * (ext_n - 1) * 6)
	var k := 0
	for a in range(ext_n - 1):
		for b in range(ext_n - 1):
			var v00 := a * ext_n + b
			var v10 := (a + 1) * ext_n + b
			var v01 := a * ext_n + b + 1
			var v11 := (a + 1) * ext_n + b + 1
			idx[k] = v00; idx[k + 1] = v01; idx[k + 2] = v10
			idx[k + 3] = v10; idx[k + 4] = v01; idx[k + 5] = v11
			k += 6
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = idx
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return mesh

## Player on the sphere: unit position + parallel-transported heading.
## Fly mode (F to toggle) or walk mode (follows the terrain height).
class_name GlobePlayer
extends Node3D

@export var walk_speed: float = 8.0
@export var fly_speed: float = 120.0
@export var sprint_mult: float = 5.0
@export var mouse_sensitivity: float = 0.0025
@export var eye_height: float = 1.8

var sphere_pos := Vector3(0, 0, 1)
var forward := Vector3(1, 0, 0)
var altitude := 150.0      ## above terrain in fly mode
var pitch := -0.35
var fly := true
var placed := false
var world: WorldRoot
var camera: Camera3D
var _mouse_captured := false


func _ready() -> void:
	world = get_parent() as WorldRoot
	camera = get_node_or_null("Camera3D")
	if camera == null:
		camera = Camera3D.new()
		camera.far = 60000.0
		camera.near = 0.5
		add_child(camera)
		camera.current = true


func place_on_sphere(p: Vector3, heading: Vector3 = Vector3.ZERO) -> void:
	sphere_pos = p.normalized()
	if heading == Vector3.ZERO:
		heading = Vector3(0, 0, 1) if absf(sphere_pos.z) < 0.9 else Vector3(1, 0, 0)
	forward = (heading - sphere_pos * sphere_pos.dot(heading)).normalized()
	placed = true


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and _mouse_captured:
		forward = GlobeMath.rotate_about(forward, sphere_pos, -event.relative.x * mouse_sensitivity)
		pitch = clampf(pitch - event.relative.y * mouse_sensitivity, -1.5, 1.5)
	if event.is_action_pressed("toggle_mouse"):
		_mouse_captured = not _mouse_captured
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED if _mouse_captured else Input.MOUSE_MODE_VISIBLE
	if event.is_action_pressed("toggle_fly"):
		fly = not fly


func _physics_process(dt: float) -> void:
	if world == null or not world.loaded_ok:
		return
	var right := forward.cross(sphere_pos)
	var ax := Input.get_action_strength("move_forward") - Input.get_action_strength("move_back")
	var ay := Input.get_action_strength("move_right") - Input.get_action_strength("move_left")
	var speed := (fly_speed if fly else walk_speed) * (sprint_mult if Input.is_action_pressed("sprint") else 1.0)
	var move := (forward * ax + right * ay) * speed * dt / world.R_planet
	if move.length_squared() > 0.0:
		var np := (sphere_pos + move).normalized()
		forward = (forward - np * np.dot(forward)).normalized()
		sphere_pos = np
	if fly:
		altitude = maxf(2.0, altitude + (Input.get_action_strength("move_up") - Input.get_action_strength("move_down")) * speed * dt)
	_update_camera()


func _update_camera() -> void:
	var h: float = maxf(world.sample_height(sphere_pos), world.sample_water(sphere_pos))
	h = maxf(h, 0.0)
	var eye: float = h + (altitude if fly else eye_height)
	var pos := world.to_local_pos(sphere_pos, eye)
	var dir := world.local_dir(sphere_pos, forward)
	var look := dir.rotated(dir.cross(Vector3.UP).normalized(), pitch)
	global_transform = Transform3D(Basis.looking_at(look, Vector3.UP), pos)

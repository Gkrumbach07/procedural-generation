// Lake water surface per tile: vertex y = water surface where water.png > 0,
// otherwise pushed under the terrain.  Same projection as terrain.gdshader.
shader_type spatial;
render_mode cull_disabled, depth_draw_opaque, blend_mix;

uniform sampler2D height_tex : filter_linear, repeat_disable, hint_default_black;
uniform sampler2D water_tex : filter_linear, repeat_disable, hint_default_black;
uniform int face = 4;
uniform int tile_x = 0;
uniform int tile_y = 0;
uniform int lod = 0;
uniform int n_fine = 256;
uniform int tile_size = 64;
uniform vec4 water_color : source_color = vec4(0.12, 0.35, 0.55, 0.75);

global uniform vec3 globe_e1;
global uniform vec3 globe_n;
global uniform vec3 globe_e2;
global uniform float globe_r_planet;

varying float v_depth;
const float HALF_PI = 1.5707963267948966;

void face_basis(int f, out vec3 r, out vec3 up, out vec3 n) {
	if (f == 0) { r = vec3(0.0, 0.0, -1.0); up = vec3(0.0, -1.0, 0.0); n = vec3(1.0, 0.0, 0.0); }
	else if (f == 1) { r = vec3(0.0, 0.0, 1.0); up = vec3(0.0, -1.0, 0.0); n = vec3(-1.0, 0.0, 0.0); }
	else if (f == 2) { r = vec3(1.0, 0.0, 0.0); up = vec3(0.0, 0.0, 1.0); n = vec3(0.0, 1.0, 0.0); }
	else if (f == 3) { r = vec3(1.0, 0.0, 0.0); up = vec3(0.0, 0.0, -1.0); n = vec3(0.0, -1.0, 0.0); }
	else if (f == 4) { r = vec3(1.0, 0.0, 0.0); up = vec3(0.0, -1.0, 0.0); n = vec3(0.0, 0.0, 1.0); }
	else { r = vec3(-1.0, 0.0, 0.0); up = vec3(0.0, -1.0, 0.0); n = vec3(0.0, 0.0, -1.0); }
}

vec3 to_sphere(int f, float u, float v) {
	vec3 r, up, n;
	face_basis(f, r, up, n);
	float s = tan((u - 0.5) * HALF_PI);
	float t = tan((v - 0.5) * HALF_PI);
	return normalize(n + s * r + t * up);
}

vec3 local_pos(float i, float j, float h) {
	float scale = float(1 << lod);
	float u = (float(tile_x * tile_size) + i) * scale / float(n_fine);
	float v = (float(tile_y * tile_size) + j) * scale / float(n_fine);
	vec3 q = to_sphere(face, u, v);
	float d = max(dot(q, globe_n), 0.05);
	vec3 g = q / d;
	return vec3(dot(g, globe_e1) * globe_r_planet, h, dot(g, globe_e2) * globe_r_planet);
}

void vertex() {
	vec2 uv = (VERTEX.xz + 0.5) / float(tile_size + 1);
	float h = texture(height_tex, uv).r;
	float w = texture(water_tex, uv).r;
	bool has = w > 0.0 && w > h;
	v_depth = has ? w - h : 0.0;
	VERTEX = local_pos(VERTEX.x, VERTEX.z, has ? w : h - 20.0);
	NORMAL = vec3(0.0, 1.0, 0.0);
}

void fragment() {
	ALBEDO = water_color.rgb;
	ALPHA = clamp(water_color.a * (0.5 + 0.5 * smoothstep(0.0, 3.0, v_depth)), 0.0, 1.0);
	ROUGHNESS = 0.15;
	METALLIC = 0.1;
}

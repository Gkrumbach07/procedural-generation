// Terrain tile: vertex (i, j) grid -> face (u, v) -> unit sphere (EAC) ->
// gnomonic projection into the anchor's tangent frame (PLAN 12.2/12.3).
shader_type spatial;
render_mode cull_back, depth_draw_opaque, specular_disabled;

uniform sampler2D height_tex : filter_linear, repeat_disable, hint_default_black;
uniform sampler2D layers_tex : filter_nearest, repeat_disable, hint_default_white;
uniform sampler2D flow_tex : filter_nearest, repeat_disable, hint_default_black;
uniform int face = 4;
uniform int tile_x = 0;
uniform int tile_y = 0;
uniform int lod = 0;
uniform int n_fine = 256;
uniform int tile_size = 64;
uniform float skirt_drop = 40.0;
uniform float sea_level = 0.0;
uniform float tile_seed = 0.0;

global uniform vec3 globe_e1;
global uniform vec3 globe_n;
global uniform vec3 globe_e2;
global uniform float globe_r_planet;
global uniform int globe_debug_mode;

varying vec2 v_uv;
varying float v_h;
varying float v_slope;
varying float v_skirt;

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

float sample_h(float i, float j) {
	vec2 uv = (vec2(i, j) + 0.5) / float(tile_size + 1);
	return texture(height_tex, uv).r;
}

void vertex() {
	float i = VERTEX.x;
	float j = VERTEX.z;
	bool skirt = VERTEX.y < -0.5;
	float h = sample_h(i, j);
	float ts = float(tile_size);
	float i0 = max(i - 1.0, 0.0), i1 = min(i + 1.0, ts);
	float j0 = max(j - 1.0, 0.0), j1 = min(j + 1.0, ts);
	vec3 pi0 = local_pos(i0, j, sample_h(i0, j));
	vec3 pi1 = local_pos(i1, j, sample_h(i1, j));
	vec3 pj0 = local_pos(i, j0, sample_h(i, j0));
	vec3 pj1 = local_pos(i, j1, sample_h(i, j1));
	vec3 nrm = normalize(cross(pj1 - pj0, pi1 - pi0));
	if (nrm.y < 0.0) nrm = -nrm;
	VERTEX = local_pos(i, j, skirt ? h - skirt_drop : h);
	NORMAL = nrm;
	v_uv = (vec2(i, j) + 0.5) / float(tile_size + 1);
	v_h = h;
	v_slope = 1.0 - nrm.y;
	v_skirt = skirt ? 1.0 : 0.0;
}

vec3 biome_color(int code) {
	// codes and colours mirror bake/globe/derive/biomes.py BIOMES (docs/DEVELOPING.md)
	if (code == 0) return vec3(0.08, 0.20, 0.43);    // ocean
	if (code == 1) return vec3(0.94, 0.96, 1.00);    // ice
	if (code == 2) return vec3(0.67, 0.75, 0.63);    // tundra
	if (code == 3) return vec3(0.16, 0.35, 0.24);    // boreal forest
	if (code == 4) return vec3(0.67, 0.75, 0.35);    // temperate grassland
	if (code == 5) return vec3(0.24, 0.51, 0.24);    // temperate forest
	if (code == 6) return vec3(0.08, 0.39, 0.27);    // temperate rainforest
	if (code == 7) return vec3(0.90, 0.82, 0.55);    // desert
	if (code == 8) return vec3(0.63, 0.59, 0.31);    // shrubland
	if (code == 9) return vec3(0.78, 0.71, 0.31);    // savanna
	if (code == 10) return vec3(0.31, 0.59, 0.20);   // tropical seasonal forest
	if (code == 11) return vec3(0.04, 0.43, 0.16);   // tropical rainforest
	if (code == 12) return vec3(0.59, 0.55, 0.51);   // alpine
	if (code == 13) return vec3(0.43, 0.39, 0.37);   // cliff
	if (code == 14) return vec3(0.20, 0.55, 0.35);   // riparian
	if (code == 15) return vec3(0.35, 0.55, 0.47);   // wetland
	if (code == 16) return vec3(0.27, 0.51, 0.90);   // lake
	return vec3(0.5, 0.5, 0.5);
}

vec3 hash_color(float x) {
	return 0.35 + 0.6 * fract(vec3(x * 0.1031, x * 0.1030 + 0.37, x * 0.0973 + 0.71) * 17.0);
}

void fragment() {
	vec4 layers = texture(layers_tex, v_uv);
	vec3 flow = texture(flow_tex, v_uv).rgb;
	int biome = int(layers.b * 255.0 + 0.5);
	vec3 col = biome_color(biome);
	float veg = layers.a;
	col = mix(col * 0.9 + vec3(0.08, 0.06, 0.04), col, veg);
	float rock = smoothstep(0.35, 0.6, v_slope);
	col = mix(col, vec3(0.42, 0.40, 0.38), rock);
	float sed = layers.r;
	col = mix(col, col * vec3(1.05, 1.0, 0.9), sed * 0.5);
	if (v_h < sea_level) col = mix(col, vec3(0.30, 0.32, 0.28), 0.6);
	if (flow.b > 0.5) col = mix(col, vec3(0.15, 0.35, 0.7), 0.7);
	if (globe_debug_mode == 1) {
		col = mix(col, hash_color(flow.g * 255.0 + tile_seed), 0.65);
	} else if (globe_debug_mode == 2) {
		col = mix(col, vec3(0.1, 0.3, 1.0), clamp(flow.r * 1.5, 0.0, 1.0));
	} else if (globe_debug_mode == 4) {
		float t = clamp(v_h / 500.0, -1.0, 1.0);
		col = t >= 0.0 ? vec3(t, 1.0 - t, 0.0) : vec3(0.0, 0.0, -t);
	} else if (globe_debug_mode == 3) {
		col = mix(col, hash_color(float(tile_x * 131 + tile_y * 17 + lod * 977 + face * 3001)), 0.5);
		if (v_uv.x < 0.004 || v_uv.y < 0.004 || v_uv.x > 0.996 || v_uv.y > 0.996) col = vec3(1.0, 0.2, 0.2);
		if (v_skirt > 0.5) col = vec3(1.0, 0.0, 1.0);
	}
	ALBEDO = col;
	ROUGHNESS = 0.95;
	METALLIC = 0.0;
}

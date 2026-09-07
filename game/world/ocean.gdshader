// Sea-level plane in the anchor's local frame (y = 0 is sea level everywhere
// in gnomonic local coordinates, up to the (d/R)^2 distortion).
shader_type spatial;
render_mode cull_disabled, blend_mix, depth_draw_opaque;
uniform vec4 color : source_color = vec4(0.08, 0.28, 0.50, 0.85);
void fragment() {
	ALBEDO = color.rgb;
	ALPHA = color.a;
	ROUGHNESS = 0.1;
	METALLIC = 0.2;
}

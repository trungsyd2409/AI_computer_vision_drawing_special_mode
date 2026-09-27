#version 330
// Particles and hand joints. Positions arrive in canvas pixels (800 x 600,
// y down) and are converted to clip space here.
uniform vec2 u_canvas;
uniform float u_px_scale;   // framebuffer pixels per canvas pixel

in vec2 in_pos;
in float in_size;
in float in_alpha;
in float in_hue;

out float v_alpha;
out float v_hue;

void main() {
    vec2 ndc = vec2(in_pos.x / u_canvas.x * 2.0 - 1.0, 1.0 - in_pos.y / u_canvas.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    // The sprite is drawn larger than the particle so it has room for a halo.
    gl_PointSize = max(in_size * u_px_scale * 3.5, 1.0);
    v_alpha = in_alpha;
    v_hue = in_hue;
}

#version 330
// Hand skeleton lines, canvas pixel coordinates in, clip space out.
uniform vec2 u_canvas;
in vec2 in_pos;
in vec4 in_color;
out vec4 v_color;

void main() {
    vec2 ndc = vec2(in_pos.x / u_canvas.x * 2.0 - 1.0, 1.0 - in_pos.y / u_canvas.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_color = in_color;
}

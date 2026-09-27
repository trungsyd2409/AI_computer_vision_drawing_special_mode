#version 330
// One oversized triangle that covers the whole viewport. v_uv has its
// origin in the top-left corner, matching NumPy/OpenCV image rows.
in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = vec2(in_pos.x * 0.5 + 0.5, 0.5 - in_pos.y * 0.5);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}

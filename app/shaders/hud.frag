#version 330
// Draws the text overlay rendered on the CPU with Pillow.
uniform sampler2D u_hud;
in vec2 v_uv;
out vec4 frag;

void main() {
    frag = texture(u_hud, v_uv);
}

#version 330
in vec4 v_color;
out vec4 frag;

void main() {
    frag = vec4(v_color.rgb * v_color.a, 1.0);
}

#version 330
// Soft round sprite: a hot core plus a faint halo, blended additively.
in float v_alpha;
in float v_hue;
out vec4 frag;

vec3 palette(float t) {
    // cyan -> violet -> magenta, the same ramp as the fabric glow
    vec3 cyan    = vec3(0.30, 0.92, 1.00);
    vec3 violet  = vec3(0.58, 0.42, 1.00);
    vec3 magenta = vec3(1.00, 0.36, 0.78);
    return t < 0.5 ? mix(cyan, violet, t * 2.0) : mix(violet, magenta, t * 2.0 - 1.0);
}

void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    float core = exp(-d * d * 22.0);
    float halo = exp(-d * d * 4.0) * 0.35;
    vec3 col = palette(v_hue);
    col = mix(col, vec3(1.0), core * 0.55);
    frag = vec4(col * (core + halo) * v_alpha, 1.0);
}

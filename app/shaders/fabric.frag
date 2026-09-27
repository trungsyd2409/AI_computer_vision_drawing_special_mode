#version 330
// Main composite: webcam image + neon grid, both bent by the fabric height
// field and by the vortices, then lit as if the sheet had real depth.
//
// Coordinate spaces
//   v_uv : 0..1 across the canvas, origin top-left
//   P    : "aspect space" = (uv.x * aspect, uv.y), so distances are the
//          same horizontally and vertically (1.0 = canvas height)

#define MAX_VORTICES 6
#define TAU 6.28318530718

uniform sampler2D u_cam;
uniform sampler2D u_height;

uniform float u_aspect;
uniform float u_canvas_h;       // canvas height in px, converts px <-> P
uniform float u_time;
uniform vec2  u_cam_scale;      // crop applied to the camera image
uniform bool  u_has_cam;

uniform float u_dim;
uniform float u_brightness;
uniform float u_contrast;
uniform float u_saturation;
uniform float u_grid;

uniform int   u_vort_count;
uniform vec4  u_vort[MAX_VORTICES];        // xy centre (P), radius (P), signed strength
uniform float u_vort_phase[MAX_VORTICES];  // accumulated rotation (radians)

uniform vec4 u_pen;      // xy (P), pressed, visible
uniform vec4 u_clutch;   // xy (P), closeness 0..1, visible
uniform float u_clutch_on;

in vec2 v_uv;
out vec4 frag;

const vec3 CYAN    = vec3(0.30, 0.92, 1.00);
const vec3 VIOLET  = vec3(0.58, 0.42, 1.00);
const vec3 MAGENTA = vec3(1.00, 0.36, 0.78);

vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    return t < 0.5 ? mix(CYAN, VIOLET, t * 2.0) : mix(VIOLET, MAGENTA, t * 2.0 - 1.0);
}

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Total depth at a point: the simulated grooves plus a funnel for each vortex.
float depth_at(vec2 P) {
    float h = texture(u_height, vec2(P.x / u_aspect, P.y)).r;
    for (int i = 0; i < u_vort_count; ++i) {
        vec2 d = P - u_vort[i].xy;
        float R = u_vort[i].z;
        float s = abs(u_vort[i].w);
        h -= s * 1.15 * exp(-dot(d, d) / (0.30 * R * R));
    }
    return h;
}

vec2 rotate(vec2 v, float a) {
    float c = cos(a), s = sin(a);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// Thin anti-aliased grid lines; returns 1 on a line, 0 between lines.
float grid_lines(vec2 g, float width) {
    vec2 w = fwidth(g) * width;
    vec2 d = abs(fract(g - 0.5) - 0.5) / max(w, vec2(1e-5));
    return 1.0 - clamp(min(d.x, d.y), 0.0, 1.0);
}

vec3 camera_colour(vec2 Q) {
    vec2 uv = vec2(Q.x / u_aspect, Q.y);
    if (!u_has_cam) {
        // Deep navy backdrop when no camera is available.
        return mix(vec3(0.035, 0.045, 0.085), vec3(0.015, 0.02, 0.04), uv.y);
    }
    vec2 cuv = 0.5 + (uv - 0.5) * u_cam_scale;
    vec3 c = texture(u_cam, cuv).rgb;
    c = (c - 0.5) * u_contrast + 0.5 + u_brightness;
    float luma = dot(c, vec3(0.299, 0.587, 0.114));
    c = mix(vec3(luma), c, u_saturation);
    c *= vec3(0.90, 0.97, 1.08);            // cool, slightly cinematic grade
    return max(c, 0.0) * u_dim;
}

void main() {
    vec2 P = vec2(v_uv.x * u_aspect, v_uv.y);
    float px = 1.0 / u_canvas_h;               // one canvas pixel in P units

    // --- Surface shape -----------------------------------------------------
    float e = 2.0 * px;
    float h  = depth_at(P);
    vec2 grad = vec2(depth_at(P + vec2(e, 0.0)) - depth_at(P - vec2(e, 0.0)),
                     depth_at(P + vec2(0.0, e)) - depth_at(P - vec2(0.0, e))) / (2.0 * e);

    // Looking into a pit, everything inside seems to slide towards its
    // centre. Sampling "uphill" along the gradient reproduces that pull.
    vec2 Q = P + grad * 0.0014;

    // Vortices twist the sampling position around their centres. The twist
    // grows with strength and fades with distance.
    for (int i = 0; i < u_vort_count; ++i) {
        vec2 c = u_vort[i].xy;
        float R = u_vort[i].z;
        float s = u_vort[i].w;
        vec2 d = Q - c;
        float fall = exp(-dot(d, d) / (0.55 * R * R));
        Q = c + rotate(d, -s * 2.4 * fall);
    }

    // --- Base image with depth shading -------------------------------------
    vec3 col = camera_colour(Q);

    vec3 n = normalize(vec3(-grad * 0.055, 1.0));
    vec3 L = normalize(vec3(-0.55, -0.75, 0.9));    // light from the top-left
    float diffuse = max(dot(n, L), 0.0) / L.z;       // 1.0 on flat ground
    col *= mix(1.0, diffuse, 0.65);
    col *= mix(0.16, 1.0, exp(h * 1.6));            // deeper = darker

    float depth01 = clamp(-h / 1.3, 0.0, 1.0);

    // --- Neon spacetime grid -------------------------------------------------
    vec2 g = Q * 17.0;
    float minor = grid_lines(g, 1.0);
    float major = grid_lines(g / 4.0, 1.4);
    float halo  = grid_lines(g / 4.0, 5.0);
    float lift = 1.0 + depth01 * 4.0;                // lines glow inside dents
    vec3 gcol = ramp(depth01 * 1.1);
    float gridA = (minor * 0.10 + major * 0.15 + halo * 0.035) * lift;
    col += gcol * gridA * u_grid;

    // Iso-depth rings, like a topographic map, only inside dents.
    float iso = abs(fract(h * 5.0 + 0.5) - 0.5) / max(fwidth(h * 5.0), 1e-4);
    float isoLine = (1.0 - clamp(iso, 0.0, 1.0)) * smoothstep(0.04, 0.25, depth01);
    col += ramp(depth01) * isoLine * 0.22 * u_grid;

    // Rim light on slopes that face the light.
    vec3 refl = reflect(-L, n);
    float spec = pow(max(refl.z, 0.0), 40.0) * smoothstep(0.02, 0.2, length(grad) * 0.055);
    col += mix(CYAN, vec3(1.0), 0.4) * spec * 0.55;

    // --- Vortex glow ---------------------------------------------------------
    for (int i = 0; i < u_vort_count; ++i) {
        vec2 d = P - u_vort[i].xy;
        float R = u_vort[i].z;
        float s = u_vort[i].w;
        float a = abs(s);
        float r = length(d);
        float th = atan(d.y, d.x);
        float dir = s >= 0.0 ? 1.0 : -1.0;

        // Logarithmic spiral arms that trail behind the rotation.
        float arms = 3.0;
        float wave = arms * (th - u_vort_phase[i]) + dir * 4.2 * log(max(r, 1e-4) / R);
        float arm = pow(0.5 + 0.5 * cos(wave), 5.0);
        float mask = smoothstep(0.10 * R, 0.30 * R, r) * exp(-r * r / (0.9 * R * R));
        col += ramp(1.0 - r / R) * arm * mask * a * 0.75;

        // Dark core with a thin bright ring around it.
        float coreR = 0.13 * R;
        col *= mix(1.0, 0.08, a * (1.0 - smoothstep(coreR * 0.5, coreR * 1.3, r)));
        float ring = exp(-pow((r - coreR) / (0.035 * R), 2.0));
        col += mix(VIOLET, vec3(1.0), 0.5) * ring * a * 0.9;
    }

    // --- Pen reticle -----------------------------------------------------------
    if (u_pen.w > 0.5) {
        float dpx = length(P - u_pen.xy) / px;
        float pressed = u_pen.z;
        float radius = mix(12.0, 7.0, pressed);
        float ring = 1.0 - smoothstep(0.6, 1.8, abs(dpx - radius));
        float dot_ = 1.0 - smoothstep(1.2, 2.4, dpx);
        vec3 pc = mix(CYAN, mix(MAGENTA, vec3(1.0), 0.35), pressed);
        col += pc * (ring * 0.9 + dot_ * 0.8);
        col += pc * exp(-dpx / 14.0) * 0.45 * pressed;
    }

    // --- Left-hand clutch indicator (fills as thumb and pinky close) --------
    if (u_clutch.w > 0.5) {
        vec2 d = P - u_clutch.xy;
        float dpx = length(d) / px;
        float ang = fract(atan(d.x, -d.y) / TAU + 1.0);   // 0 at 12 o'clock, clockwise
        float ring = 1.0 - smoothstep(0.7, 1.9, abs(dpx - 17.0));
        float filled = step(ang, u_clutch.z);
        vec3 cc = mix(VIOLET, MAGENTA, u_clutch_on);
        col += cc * ring * (0.18 + 0.85 * filled);
        col += cc * (1.0 - smoothstep(9.0, 11.0, dpx)) * 0.35 * u_clutch_on;
    }

    // --- Finishing -----------------------------------------------------------
    vec2 vc = v_uv - 0.5;
    col *= 1.0 - 0.45 * pow(length(vc * vec2(1.0, 0.9)) * 1.35, 2.4);
    col += (hash(v_uv * 800.0 + fract(u_time) * 17.0) - 0.5) * 0.018;
    frag = vec4(col, 1.0);
}

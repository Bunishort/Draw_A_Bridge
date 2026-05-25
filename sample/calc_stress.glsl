#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rgba32f, binding = 2) uniform image2D img_stress_old;
layout(rg32f, binding = 5) uniform image2D img_masks; // Remplace le SSBO

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;
uniform float elas_lambda_2mu; uniform float elas_2mu;

vec2 get_ut(ivec2 p) {
    vec4 data = imageLoad(img_pos_vel, p);
    return explicit_b * data.xy + G0 * data.zw;
}

struct Masks {
    bool isddx1, isddx2, isddy1, isddy2;
    bool x_front, y_front, x_front_sb, y_front_sb;
    bool solid_not_uimp;
};

Masks get_masks(ivec2 p) {
    // Lecture du canal R (solid) sur le voisinage 2x2
    bool s00 = imageLoad(img_masks, p).r > 0.5;
    bool s10 = imageLoad(img_masks, p + ivec2(1, 0)).r > 0.5;
    bool s01 = imageLoad(img_masks, p + ivec2(0, 1)).r > 0.5;
    bool s11 = imageLoad(img_masks, p + ivec2(1, 1)).r > 0.5;

    Masks m;
    // L'équivalent de tes ddx**2 == 2 (les deux pixels adjacents sont solides)
    m.isddx1 = s00 && s01;
    m.isddx2 = s10 && s11;
    m.isddy1 = s00 && s10;
    m.isddy2 = s01 && s11;

    // L'équivalent des frontières (XOR = un pixel solide, l'autre vide)
    m.x_front = (s00 != s01) || (s10 != s11);
    m.y_front = (s00 != s10) || (s01 != s11);
    bool corner = m.x_front && m.y_front;

    // Frontières sans les coins
    m.x_front_sb = m.x_front && !corner;
    m.y_front_sb = m.y_front && !corner;

    // Canal G (solid_not_uimp)
    m.solid_not_uimp = imageLoad(img_masks, p).g > 0.5;
    return m;
}

struct Strains { float exx, eyy, exy, eyx; };

Strains calc_def(ivec2 p) {
    Strains s; s.exx = 0; s.eyy = 0; s.exy = 0; s.eyx = 0;
    Masks m = get_masks(p); // Calcul local des masques pour ce point

    if (m.isddx1) {
        vec2 u_i1_j = get_ut(p + ivec2(0, 1)); vec2 u_i_j = get_ut(p);
        s.exx += (u_i1_j.x - u_i_j.x); s.eyx += (u_i1_j.y - u_i_j.y);
    }
    if (m.isddx2) {
        vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i_j1 = get_ut(p + ivec2(1, 0));
        s.exx += (u_i1_j1.x - u_i_j1.x); s.eyx += (u_i1_j1.y - u_i_j1.y);
    }
    if (m.isddy1) {
        vec2 u_i_j1 = get_ut(p + ivec2(1, 0)); vec2 u_i_j = get_ut(p);
        s.eyy += (u_i_j1.y - u_i_j.y); s.exy += (u_i_j1.x - u_i_j.x);
    }
    if (m.isddy2) {
        vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i1_j = get_ut(p + ivec2(0, 1));
        s.exy += (u_i1_j1.x - u_i1_j.x); s.eyy += (u_i1_j1.y - u_i1_j.y);
    }

    s.exx /= (2.0 * lm); s.eyy /= (2.0 * lm); s.exy /= (4.0 * lm); s.eyx /= (4.0 * lm);

    if (m.y_front) { s.exx *= 2.0; s.eyx *= 2.0; }
    if (m.x_front) { s.eyy *= 2.0; s.exy *= 2.0; }
    if (m.x_front_sb) { s.exx = coef * s.eyy; s.eyx = -s.exy; }
    if (m.y_front_sb) { s.eyy = coef * s.exx; s.exy = -s.eyx; }
    return s;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.y < height - 2 && p.x < width - 2) {
        Masks m = get_masks(p);

        float duxdx2 = 0.0, duydx2 = 0.0, duxdy2 = 0.0, duydy2 = 0.0;
        if (m.isddx2) {
            vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i_j1 = get_ut(p + ivec2(1, 0));
            duxdx2 = (u_i1_j1.x - u_i_j1.x) / (2.0 * lm);
            duydx2 = (u_i1_j1.y - u_i_j1.y) / (4.0 * lm);
        }
        if (m.isddy2) {
            vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i1_j = get_ut(p + ivec2(0, 1));
            duxdy2 = (u_i1_j1.x - u_i1_j.x) / (4.0 * lm);
            duydy2 = (u_i1_j1.y - u_i1_j.y) / (2.0 * lm);
        }

        Strains s00 = calc_def(p);
        Strains s01 = calc_def(p + ivec2(1, 0));
        Strains s10 = calc_def(p + ivec2(0, 1));

        vec4 s_old_val = imageLoad(img_stress_old, p);
        vec4 s_now;
        s_now.x = s_old_val.x * visco_fact_2;
        s_now.y = s_old_val.y * visco_fact_2;
        s_now.z = s_old_val.z * visco_fact_2;
        s_now.w = s_old_val.w * visco_fact_2;

        if (m.isddx2) {
            float sxx = ((s00.exx + s01.exx + 2.0*elas_lambda_ratio*(s00.eyy + s01.eyy))/4.0 + duxdx2) * elas_lambda_2mu;
            float sxy_x = ((2.0*(s00.exy + s01.exy) + s00.eyx + s01.eyx)/4.0 + duydx2) * elas_2mu;
            s_now.x += sxx * visco_fact_1;
            s_now.y += sxy_x * visco_fact_1;
        }

        if (m.isddy2) {
            float syy = ((s00.eyy + s10.eyy + 2.0*elas_lambda_ratio*(s00.exx + s10.exx))/4.0 + duydy2) * elas_lambda_2mu;
            float sxy_y = ((s00.exy + s10.exy + 2.0*(s00.eyx + s10.eyx))/4.0 + duxdy2) * elas_2mu;
            s_now.z += syy * visco_fact_1;
            s_now.w += sxy_y * visco_fact_1;
        }

        imageStore(img_stress_old, p, s_now);
    }
}
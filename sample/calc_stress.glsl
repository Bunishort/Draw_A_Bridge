#version 430
layout (local_size_x = 16, local_size_y = 16) in;

// On utilise image2D pour l'accès direct en lecture/écriture
layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rgba32f, binding = 2) uniform image2D img_stress_old;
layout(rgba32f, binding = 7) uniform image2D img_masks_flt;
// Pour les masques entiers, on peut utiliser des buffers ou d'autres textures
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;

// Helper pour récupérer uxt/uyt à partir de la texture pos_vel
vec2 get_ut(ivec2 p) {
    vec4 data = imageLoad(img_pos_vel, p);
    // data.x = ux, data.y = uy, data.z = vx, data.w = vy
    return explicit_b * data.xy + G0 * data.zw;
}

struct Strains { float exx, eyy, exy, eyx; };

Strains calc_def(ivec2 p) {
    Strains s; s.exx = 0; s.eyy = 0; s.exy = 0; s.eyx = 0;
    int id = p.y * width + p.x;
    int off = width * height;

    if (m_int[id] > 0)           { vec2 u_p10 = get_ut(p+ivec2(1,0)); vec2 u_p00 = get_ut(p); s.exx += (u_p10.x - u_p00.x); s.eyx += (u_p10.y - u_p00.y); }
    if (m_int[id + off] > 0)     { vec2 u_p11 = get_ut(p+ivec2(1,1)); vec2 u_p01 = get_ut(p+ivec2(0,1)); s.exx += (u_p11.x - u_p01.x); s.eyx += (u_p11.y - u_p01.y); }
    if (m_int[id + 2*off] > 0)   { vec2 u_p01 = get_ut(p+ivec2(0,1)); vec2 u_p00 = get_ut(p); s.eyy += (u_p01.y - u_p00.y); s.exy += (u_p01.x - u_p00.x); }
    if (m_int[id + 3*off] > 0)   { vec2 u_p11 = get_ut(p+ivec2(1,1)); vec2 u_p10 = get_ut(p+ivec2(1,0)); s.exy += (u_p11.x - u_p10.x); s.eyy += (u_p11.y - u_p10.y); }

    s.exx /= (2.0 * lm); s.eyy /= (2.0 * lm); s.exy /= (4.0 * lm); s.eyx /= (4.0 * lm);

    if (m_int[id + 4*off] > 0) { s.exx *= 2.0; s.eyx *= 2.0; }
    if (m_int[id + 5*off] > 0) { s.eyy *= 2.0; s.exy *= 2.0; }
    if (m_int[id + 6*off] > 0) { s.exx = coef * s.eyy; s.eyx = -s.exy; }
    if (m_int[id + 7*off] > 0) { s.eyy = coef * s.exx; s.exy = -s.eyx; }
    return s;
}

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.y < height - 2 && p.x < width - 2) {
        int id = p.y * width + p.x;
        int off = width * height;

        float duxdx2 = 0.0, duydx2 = 0.0, duxdy2 = 0.0, duydy2 = 0.0;
        if (m_int[id + off] > 0) {
            vec2 u11 = get_ut(p+ivec2(1,1)); vec2 u01 = get_ut(p+ivec2(0,1));
            duxdx2 = (u11.x - u01.x) / (2.0 * lm);
            duydx2 = (u11.y - u01.y) / (4.0 * lm);
        }
        if (m_int[id + 3*off] > 0) {
            vec2 u11 = get_ut(p+ivec2(1,1)); vec2 u10 = get_ut(p+ivec2(1,0));
            duxdy2 = (u11.x - u10.x) / (4.0 * lm);
            duydy2 = (u11.y - u10.y) / (2.0 * lm);
        }

        Strains s00 = calc_def(p);
        Strains s01 = calc_def(p + ivec2(0, 1));
        Strains s10 = calc_def(p + ivec2(1, 0));

        vec4 m_flt = imageLoad(img_masks_flt, p);
        float sxx = ((s00.exx + s01.exx + 2.0*elas_lambda_ratio*(s00.eyy + s01.eyy))/4.0 + duxdx2) * m_flt.x;
        float syy = ((s00.eyy + s10.eyy + 2.0*elas_lambda_ratio*(s00.exx + s10.exx))/4.0 + duydy2) * m_flt.y;
        float sxy_x = ((2.0*(s00.exy + s01.exy) + s00.eyx + s01.eyx)/4.0 + duydx2) * m_flt.z;
        float sxy_y = ((s00.exy + s10.exy + 2.0*(s00.eyx + s10.eyx))/4.0 + duxdy2) * m_flt.w;

        vec4 s_old = imageLoad(img_stress_old, p);
        vec4 s_now;
        s_now.x = sxx * visco_fact_1 + s_old.x * visco_fact_2;
        s_now.y = sxy_x * visco_fact_1 + s_old.y * visco_fact_2;
        s_now.z = syy * visco_fact_1 + s_old.z * visco_fact_2;
        s_now.w = sxy_y * visco_fact_1 + s_old.w * visco_fact_2;

        imageStore(img_stress_old, p, s_now);
    }
}
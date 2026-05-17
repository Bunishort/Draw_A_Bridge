#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rgba32f, binding = 2) uniform image2D img_stress_old;
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;
uniform float elas_lambda_2mu; uniform float elas_2mu;


vec2 get_ut(ivec2 p) {
    vec4 data = imageLoad(img_pos_vel, p);
    return explicit_b * data.xy + G0 * data.zw;
}

struct Strains { float exx, eyy, exy, eyx; };

Strains calc_def(ivec2 p) {
    Strains s; s.exx = 0; s.eyy = 0; s.exy = 0; s.eyx = 0;
    int id = p.y * width + p.x;
    int off = width * height;

    // Rappel mapping : i = p.y, j = p.x
    // idx(i+1, j)   -> p + ivec2(0, 1)
    // idx(i, j+1)   -> p + ivec2(1, 0)
    // idx(i+1, j+1) -> p + ivec2(1, 1)

    if (m_int[id] > 0) {
        vec2 u_i1_j = get_ut(p + ivec2(0, 1)); vec2 u_i_j = get_ut(p);
        s.exx += (u_i1_j.x - u_i_j.x); s.eyx += (u_i1_j.y - u_i_j.y);
    }
    if (m_int[id + off] > 0) {
        vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i_j1 = get_ut(p + ivec2(1, 0));
        s.exx += (u_i1_j1.x - u_i_j1.x); s.eyx += (u_i1_j1.y - u_i_j1.y);
    }
    if (m_int[id + 2*off] > 0) {
        vec2 u_i_j1 = get_ut(p + ivec2(1, 0)); vec2 u_i_j = get_ut(p);
        s.eyy += (u_i_j1.y - u_i_j.y); s.exy += (u_i_j1.x - u_i_j.x);
    }
    if (m_int[id + 3*off] > 0) {
        vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i1_j = get_ut(p + ivec2(0, 1));
        s.exy += (u_i1_j1.x - u_i1_j.x); s.eyy += (u_i1_j1.y - u_i1_j.y);
    }

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
            vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i_j1 = get_ut(p + ivec2(1, 0));
            duxdx2 = (u_i1_j1.x - u_i_j1.x) / (2.0 * lm);
            duydx2 = (u_i1_j1.y - u_i_j1.y) / (4.0 * lm);
        }
        if (m_int[id + 3*off] > 0) {
            vec2 u_i1_j1 = get_ut(p + ivec2(1, 1)); vec2 u_i1_j = get_ut(p + ivec2(0, 1));
            duxdy2 = (u_i1_j1.x - u_i1_j.x) / (4.0 * lm);
            duydy2 = (u_i1_j1.y - u_i1_j.y) / (2.0 * lm);
        }

        Strains s00 = calc_def(p);
        Strains s01 = calc_def(p + ivec2(1, 0)); // j+1 correspond à idx(i, j+1)
        Strains s10 = calc_def(p + ivec2(0, 1)); // i+1 correspond à idx(i+1, j)

        vec4 s_old_val = imageLoad(img_stress_old, p);
        vec4 s_now;
        s_now.x = s_old_val.x * visco_fact_2;
        s_now.y = s_old_val.y * visco_fact_2;
        s_now.z = s_old_val.z * visco_fact_2;
        s_now.w = s_old_val.w * visco_fact_2;

        // isstress x edge
        if (m_int[id + 9*off] > 0) {
            float sxx = ((s00.exx + s01.exx + 2.0*elas_lambda_ratio*(s00.eyy + s01.eyy))/4.0 + duxdx2) * elas_lambda_2mu;
            float sxy_x = ((2.0*(s00.exy + s01.exy) + s00.eyx + s01.eyx)/4.0 + duydx2) * elas_2mu;
            s_now.x += sxx * visco_fact_1;
            s_now.y += sxy_x * visco_fact_1;
        }
        //isstress y edge
        if (m_int[id + 10*off] > 0) {
            float syy = ((s00.eyy + s10.eyy + 2.0*elas_lambda_ratio*(s00.exx + s10.exx))/4.0 + duydy2) * elas_lambda_2mu;
            float sxy_y = ((s00.exy + s10.exy + 2.0*(s00.eyx + s10.eyx))/4.0 + duxdy2) * elas_2mu;
            s_now.z += syy * visco_fact_1;
            s_now.w += sxy_y * visco_fact_1;
        }


        imageStore(img_stress_old, p, s_now);
    }
}
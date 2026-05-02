#version 430

layout (local_size_x = 16, local_size_y = 16) in;

// --- BUFFERS D'ENTRÉE/SORTIE ---
layout(std430, binding = 0) buffer b_pos { float ux[]; float uy[]; };
layout(std430, binding = 1) buffer b_vel { float vx[]; float vy[]; };
layout(std430, binding = 2) buffer b_stress_old { float sxx_old[]; float sxy_x_old[]; float syy_old[]; float sxy_y_old[]; };
layout(std430, binding = 3) buffer b_force { float fx_imp[]; float fy_imp[]; };
layout(std430, binding = 4) buffer b_ext { float bx[]; float by[]; };

// --- BUFFERS DE MASQUES ET CONSTANTES ---
// Note: Les booléens sont passés en int (0 ou 1)
layout(std430, binding = 5) buffer b_masks {
    int isddx1[]; int isddx2[]; int isddy1[]; int isddy2[];
    int y_front_def[]; int x_front_def[]; int x_front_s[]; int y_front_s[];
    int solid_not_uimp[];
    float isstress_x_l2m[]; float isstress_y_l2m[]; float isstress_x_2m[]; float isstress_y_2m[];
};

// Buffers temporaires pour stocker les contraintes de l'étape actuelle (nécessaires pour la divergence)
layout(std430, binding = 6) buffer b_stress_curr { float sxx_c[]; float sxy_xc[]; float syy_c[]; float sxy_yc[]; };

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;
uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

int idx(int i, int j) { return i * width + j; }

// --- LOGIQUE CALC_STRESS (Step 1) ---
void compute_stress() {
    int j = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    if (i >= height - 2 || j >= width - 2) return;

    int id = idx(i, j);

    // Fonction locale pour calculer les déformations
    // On simule uxt = explicit_b * ux + G0 * vx
    auto get_uxt = [&](int r, int c) { return explicit_b * ux[idx(r,c)] + G0 * vx[idx(r,c)]; };
    auto get_uyt = [&](int r, int c) { return explicit_b * uy[idx(r,c)] + G0 * vy[idx(r,c)]; };

    auto calc_def = [&](int r, int c) {
        float exx = 0, eyy = 0, exy = 0, eyx = 0;
        int rid = idx(r, c);
        if (isddx1[rid] > 0) { exx += (get_uxt(r+1, c) - get_uxt(r, c)); eyx += (get_uyt(r+1, c) - get_uyt(r, c)); }
        if (isddx2[rid] > 0) { exx += (get_uxt(r+1, c+1) - get_uxt(r, c+1)); eyx += (get_uyt(r+1, c+1) - get_uyt(r, c+1)); }
        if (isddy1[rid] > 0) { eyy += (get_uyt(r, c+1) - get_uyt(r, c)); exy += (get_uxt(r, c+1) - get_uxt(r, c)); }
        if (isddy2[rid] > 0) { exy += (get_uxt(r+1, c+1) - get_uxt(r+1, c)); eyy += (get_uyt(r+1, c+1) - get_uyt(r+1, c)); }

        exx /= (2.*lm); eyy /= (2.*lm); exy /= (4.*lm); eyx /= (4.*lm);
        if (y_front_def[rid] > 0) { exx *= 2.; eyx *= 2.; }
        if (x_front_def[rid] > 0) { eyy *= 2.; exy *= 2.; }
        if (x_front_s[rid] > 0) { exx = coef * eyy; eyx = -exy; }
        if (y_front_s[rid] > 0) { eyy = coef * exx; exy = -eyx; }
        return vec4(exx, eyy, exy, eyx);
    };

    vec4 s00 = calc_def(i, j);
    vec4 s01 = calc_def(i, j+1);
    vec4 s10 = calc_def(i+1, j);

    float duxdx2 = (isddx2[id] > 0) ? (get_uxt(i+1, j+1) - get_uxt(i, j+1))/(2.*lm) : 0;
    float duydx2 = (isddx2[id] > 0) ? (get_uyt(i+1, j+1) - get_uyt(i, j+1))/(4.*lm) : 0;
    float duxdy2 = (isddy2[id] > 0) ? (get_uxt(i+1, j+1) - get_uxt(i+1, j))/(4.*lm) : 0;
    float duydy2 = (isddy2[id] > 0) ? (get_uyt(i+1, j+1) - get_uyt(i+1, j))/(2.*lm) : 0;

    // Calcul final Stress et mise à jour Visco
    sxx_c[id] = ((s00.x + s01.x + 2.0*elas_lambda_ratio*(s00.y + s01.y))/4.0 + duxdx2) * isstress_x_l2m[id];
    sxy_xc[id] = ((2.0*(s00.z + s01.z) + s00.w + s01.w)/4.0 + duydx2) * isstress_x_2m[id];
    syy_c[id] = ((s00.y + s10.y + 2.0*elas_lambda_ratio*(s00.x + s10.x))/4.0 + duydy2) * isstress_y_l2m[id];
    sxy_yc[id] = ((s00.z + s10.z + 2.0*(s00.w + s10.w))/4.0 + duxdy2) * isstress_y_2m[id];

    // Update old stress
    sxx_old[id] = sxx_c[id] * visco_fact_1 + sxx_old[id] * visco_fact_2;
    sxy_x_old[id] = sxy_xc[id] * visco_fact_1 + sxy_x_old[id] * visco_fact_2;
    syy_old[id] = syy_c[id] * visco_fact_1 + syy_old[id] * visco_fact_2;
    sxy_y_old[id] = sxy_yc[id] * visco_fact_1 + sxy_y_old[id] * visco_fact_2;
}

// --- LOGIQUE EXPLICIT_STEP (Step 2) ---
void compute_physics() {
    int j = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    if (i < 1 || j < 1 || i >= height || j >= width) return;

    int id = idx(i, j);

    float div_x = -sxx_c[idx(i-1, j-1)] + sxx_c[idx(i, j-1)] - sxy_yc[idx(i-1, j-1)] + sxy_yc[idx(i-1, j)];
    float div_y = -syy_c[idx(i-1, j-1)] + syy_c[idx(i-1, j)] - sxy_xc[idx(i-1, j-1)] + sxy_xc[idx(i, j-1)];

    float m = float(solid_not_uimp[id]) / lm;
    float dvx = (div_x * m - bx[id]) * dt_by_vol_mass;
    float dvy = (div_y * m - by[id]) * dt_by_vol_mass;

    vx[id] += dvx;
    vy[id] += dvy;
    fx_imp[id] -= damping_eff * dvx;
    fy_imp[id] -= damping_eff * dvy;
    ux[id] += vx[id] * dt;
    uy[id] += vy[id] * dt;
}
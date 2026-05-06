#version 430

layout (local_size_x = 16, local_size_y = 16) in;

// --- MAPPING DES BUFFERS (Strictement identique à votre stack Python) ---
layout(std430, binding = 0) buffer b_pos { float pos[]; };             // [ux, uy]
layout(std430, binding = 1) buffer b_vel { float vel[]; };             // [vx, vy]
layout(std430, binding = 2) buffer b_stress_old { float s_old[]; };    // [sxx, sxy_x, syy, sxy_y]
layout(std430, binding = 3) buffer b_force { float forces[]; };        // [fx, fy]
layout(std430, binding = 4) buffer b_ext { float ext_forces[]; };      // [bx, by]
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };       // [isddx1, isddx2, isddy1, isddy2, y_fd, x_fd, x_fs, y_fs, solid]
layout(std430, binding = 6) buffer b_stress_curr { float s_curr[]; };  // [sxx_c, sxy_xc, syy_c, sxy_yc]
layout(std430, binding = 7) buffer b_masks_flt { float m_flt[]; };     // [l2m_x, l2m_y, 2m_x, 2m_y]

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;
uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

int offset() { return width * height; }
int idx(int i, int j) { return i * width + j; }

// uxt = explicit_b * ux + G0 * vx (Identique à l'appel Numba)
float get_uxt(int i, int j) {
    int id = idx(i, j); return explicit_b * pos[id] + G0 * vel[id];
}
float get_uyt(int i, int j) {
    int id = idx(i, j); return explicit_b * pos[id + offset()] + G0 * vel[id + offset()];
}

struct Strains { float exx, eyy, exy, eyx; };

// Reproduit calc_def(i, j) de Numba
Strains calc_def(int i, int j) {
    Strains s; s.exx = 0; s.eyy = 0; s.exy = 0; s.eyx = 0;
    int id = idx(i, j);
    int off = offset();

    if (m_int[id] > 0)           { s.exx += (get_uxt(i+1, j) - get_uxt(i, j)); s.eyx += (get_uyt(i+1, j) - get_uyt(i, j)); }
    if (m_int[id + off] > 0)     { s.exx += (get_uxt(i+1, j+1) - get_uxt(i, j+1)); s.eyx += (get_uyt(i+1, j+1) - get_uyt(i, j+1)); }
    if (m_int[id + 2*off] > 0)   { s.eyy += (get_uyt(i, j+1) - get_uyt(i, j)); s.exy += (get_uxt(i, j+1) - get_uxt(i, j)); }
    if (m_int[id + 3*off] > 0)   { s.exy += (get_uxt(i+1, j+1) - get_uxt(i+1, j)); s.eyy += (get_uyt(i+1, j+1) - get_uyt(i+1, j)); }

    s.exx /= (2.0 * lm); s.eyy /= (2.0 * lm); s.exy /= (4.0 * lm); s.eyx /= (4.0 * lm);

    if (m_int[id + 4*off] > 0) { s.exx *= 2.0; s.eyx *= 2.0; }
    if (m_int[id + 5*off] > 0) { s.eyy *= 2.0; s.exy *= 2.0; }
    if (m_int[id + 6*off] > 0) { s.exx = coef * s.eyy; s.eyx = -s.exy; }
    if (m_int[id + 7*off] > 0) { s.eyy = coef * s.exx; s.exy = -s.eyx; }

    return s;
}

void main() {
    int j = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    int id = idx(i, j);
    int off = offset();

    // --- PHASE 1 : nfi_calc_stress ---
    if (i < height - 2 && j < width - 2) {
        float duxdx2 = 0.0, duydx2 = 0.0, duxdy2 = 0.0, duydy2 = 0.0;
        if (m_int[id + off] > 0) {
            duxdx2 = (get_uxt(i+1, j+1) - get_uxt(i, j+1)) / (2.0 * lm);
            duydx2 = (get_uyt(i+1, j+1) - get_uyt(i, j+1)) / (4.0 * lm);
        }
        if (m_int[id + 3*off] > 0) {
            duxdy2 = (get_uxt(i+1, j+1) - get_uxt(i+1, j)) / (4.0 * lm);
            duydy2 = (get_uyt(i+1, j+1) - get_uyt(i+1, j)) / (2.0 * lm);
        }

        Strains s00 = calc_def(i, j);
        Strains s01 = calc_def(i, j+1);
        Strains s10 = calc_def(i+1, j);

        // Calcul élastique + Masque (temp / 4.0 + du) * mask
        float sxx = ((s00.exx + s01.exx + 2.0*elas_lambda_ratio*(s00.eyy + s01.eyy))/4.0 + duxdx2) * m_flt[id];
        float syy = ((s00.eyy + s10.eyy + 2.0*elas_lambda_ratio*(s00.exx + s10.exx))/4.0 + duydy2) * m_flt[id + off];
        float sxy_x = ((2.0*(s00.exy + s01.exy) + s00.eyx + s01.eyx)/4.0 + duydx2) * m_flt[id + 2*off];
        float sxy_y = ((s00.exy + s10.exy + 2.0*(s00.eyx + s10.eyx))/4.0 + duxdy2) * m_flt[id + 3*off];

        // --- CORRECTION : Mise à jour viscoélastique AVANT la divergence ---
        // On stocke le résultat mis à jour dans s_curr pour que la phase 2 l'utilise
        s_curr[id]         = sxx * visco_fact_1 + s_old[id] * visco_fact_2;         // sxx_x
        s_curr[id + off]   = sxy_x * visco_fact_1 + s_old[id + off] * visco_fact_2; // sxy_x
        s_curr[id + 2*off] = syy * visco_fact_1 + s_old[id + 2*off] * visco_fact_2; // syy_y
        s_curr[id + 3*off] = sxy_y * visco_fact_1 + s_old[id + 3*off] * visco_fact_2; // sxy_y

        // Mise à jour du buffer permanent pour l'itération suivante (sxx_x_old = sxx_x)
        s_old[id]         = s_curr[id];
        s_old[id + off]   = s_curr[id + off];
        s_old[id + 2*off] = s_curr[id + 2*off];
        s_old[id + 3*off] = s_curr[id + 3*off];
    }

    // On attend que toutes les contraintes viscoélastiques soient écrites
    barrier();

    // --- PHASE 2 : Divergence & explicit_step ---
    if (i >= 1 && j >= 1 && i < height && j < width) {
        // Accès aux bords pour la divergence
        float sxx_prev = s_curr[idx(i-1, j-1)];
        float sxx_curr = s_curr[idx(i, j-1)];
        float sxy_x_prev = s_curr[idx(i-1, j-1) + off];
        float sxy_x_curr = s_curr[idx(i, j-1) + off];
        float sxy_y_prev = s_curr[idx(i-1, j-1) + 3*off];
        float sxy_y_curr = s_curr[idx(i-1, j) + 3*off];
        float syy_prev = s_curr[idx(i-1, j-1) + 2*off];
        float syy_curr = s_curr[idx(i-1, j) + 2*off];

        float c_sxx_dx = -sxx_prev + sxx_curr;
        float c_sxy_dx = -sxy_x_prev + sxy_x_curr;
        float c_sxy_dy = -sxy_y_prev + sxy_y_curr;
        float c_syy_dy = -syy_prev + syy_curr;

        float m = float(m_int[id + 8*off]) / lm; // solid_not_uimp
        float dvx = (c_sxx_dx + c_sxy_dy) * m - ext_forces[id];     // a_u_x - bx
        float dvy = (c_syy_dy + c_sxy_dx) * m - ext_forces[id + off]; // a_u_y - by

        dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

        vel[id] += dvx;
        vel[id + off] += dvy;
        forces[id] -= damping_eff * dvx;
        forces[id + off] -= damping_eff * dvy;
        pos[id] += vel[id] * dt;
        pos[id + off] += vel[id + off] * dt;
    }
}
#version 430

layout (local_size_x = 16, local_size_y = 16) in;

// --- DÉFINITION DES BUFFERS (Même structure que précédemment) ---
layout(std430, binding = 0) buffer b_pos { float pos[]; };             // [ux, uy]
layout(std430, binding = 1) buffer b_vel { float vel[]; };             // [vx, vy]
layout(std430, binding = 2) buffer b_stress_old { float s_old[]; };    // [sxx, sxy_x, syy, sxy_y]
layout(std430, binding = 3) buffer b_force { float forces[]; };        // [fx, fy]
layout(std430, binding = 4) buffer b_ext { float ext_forces[]; };      // [bx, by]
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };       // [isddx1, isddx2, isddy1, isddy2, y_f, x_f, x_fs, y_fs, solid]
layout(std430, binding = 6) buffer b_stress_curr { float s_curr[]; };  // [sxx_c, sxy_xc, syy_c, sxy_yc]
layout(std430, binding = 7) buffer b_masks_flt { float m_flt[]; };     // [l2m_x, l2m_y, 2m_x, 2m_y]

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;
uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

// Utilitaires pour les offsets
int offset() { return width * height; }
int idx(int i, int j) { return i * width + j; }

// Accesseurs pour les données empilées
float get_uxt(int i, int j) {
    int id = idx(i, j);
    int off = offset();
    return explicit_b * pos[id] + G0 * vel[id];
}
float get_uyt(int i, int j) {
    int id = idx(i, j);
    int off = offset();
    return explicit_b * pos[id + off] + G0 * vel[id + off];
}

struct Strains { float exx, eyy, exy, eyx; };

Strains calc_def(int i, int j) {
    Strains s; s.exx = 0; s.eyy = 0; s.exy = 0; s.eyx = 0;
    int id = idx(i, j);
    int off = offset();

    // Masques entiers (Binding 5)
    if (m_int[id] > 0)           { s.exx += (get_uxt(i+1, j) - get_uxt(i, j)); s.eyx += (get_uyt(i+1, j) - get_uyt(i, j)); } // isddx1
    if (m_int[id + off] > 0)     { s.exx += (get_uxt(i+1, j+1) - get_uxt(i, j+1)); s.eyx += (get_uyt(i+1, j+1) - get_uyt(i, j+1)); } // isddx2
    if (m_int[id + 2*off] > 0)   { s.eyy += (get_uyt(i, j+1) - get_uyt(i, j)); s.exy += (get_uxt(i, j+1) - get_uxt(i, j)); } // isddy1
    if (m_int[id + 3*off] > 0)   { s.exy += (get_uxt(i+1, j+1) - get_uxt(i+1, j)); s.eyy += (get_uyt(i+1, j+1) - get_uyt(i+1, j)); } // isddy2

    s.exx /= (2.0 * lm); s.eyy /= (2.0 * lm); s.exy /= (4.0 * lm); s.eyx /= (4.0 * lm);

    if (m_int[id + 4*off] > 0) { s.exx *= 2.0; s.eyx *= 2.0; } // y_frontier_def
    if (m_int[id + 5*off] > 0) { s.eyy *= 2.0; s.exy *= 2.0; } // x_frontier_def
    if (m_int[id + 6*off] > 0) { s.exx = coef * s.eyy; s.eyx = -s.exy; } // x_frontier_def_s
    if (m_int[id + 7*off] > 0) { s.eyy = coef * s.exx; s.exy = -s.eyx; } // y_frontier_def_s

    return s;
}

void main() {
    int j = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    int id = idx(i, j);
    int off = offset();

    // --- PHASE 1 : CALC_STRESS ---
    if (i < height - 2 && j < width - 2) {
        float duxdx2 = 0.0, duydx2 = 0.0, duxdy2 = 0.0, duydy2 = 0.0;
        if (m_int[id + off] > 0) { // isddx2
            duxdx2 = (get_uxt(i+1, j+1) - get_uxt(i, j+1)) / (2.0 * lm);
            duydx2 = (get_uyt(i+1, j+1) - get_uyt(i, j+1)) / (4.0 * lm);
        }
        if (m_int[id + 3*off] > 0) { // isddy2
            duxdy2 = (get_uxt(i+1, j+1) - get_uxt(i+1, j)) / (4.0 * lm);
            duydy2 = (get_uyt(i+1, j+1) - get_uyt(i+1, j)) / (2.0 * lm);
        }

        Strains s00 = calc_def(i, j);
        Strains s01 = calc_def(i, j+1);
        Strains s10 = calc_def(i+1, j);

        // Stress Calculation (Utilisation des masques float du Binding 7)
        s_curr[id]         = ((s00.exx + s01.exx + 2.0*elas_lambda_ratio*(s00.eyy + s01.eyy))/4.0 + duxdx2) * m_flt[id];         // sxx_c
        s_curr[id + off]   = ((2.0*(s00.exy + s01.exy) + s00.eyx + s01.eyx)/4.0 + duydx2) * m_flt[id + 2*off];                   // sxy_xc
        s_curr[id + 2*off] = ((s00.eyy + s10.eyy + 2.0*elas_lambda_ratio*(s00.exx + s10.exx))/4.0 + duydy2) * m_flt[id + off];   // syy_c
        s_curr[id + 3*off] = ((s00.exy + s10.exy + 2.0*(s00.eyx + s10.eyx))/4.0 + duxdy2) * m_flt[id + 3*off];                   // sxy_yc

        // Viscoelastic Update
        for(int k=0; k<4; k++) {
            s_old[id + k*off] = s_curr[id + k*off] * visco_fact_1 + s_old[id + k*off] * visco_fact_2;
        }
    }

    barrier(); // Synchronisation impérative avant la divergence

    // --- PHASE 2 : PHYSICS STEP ---
    if (i >= 1 && j >= 1 && i < height && j < width) {
        int id00 = idx(i-1, j-1);
        int id10 = idx(i, j-1);
        int id01 = idx(i-1, j);

        float c_sxx_dx = -s_old[id00] + s_old[id10];
        float c_sxy_dx = -s_old[id00 + off] + s_old[id10 + off];
        float c_sxy_dy = -s_old[id00 + 3*off] + s_old[id01 + 3*off];
        float c_syy_dy = -s_old[id00 + 2*off] + s_old[id01 + 2*off];

        float m = float(m_int[id + 8*off]) / lm; // solid_not_uimp
        float dvx = (c_sxx_dx + c_sxy_dy) * m - ext_forces[id];
        float dvy = (c_syy_dy + c_sxy_dx) * m - ext_forces[id + off];

        dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

        vel[id] += dvx;
        vel[id + off] += dvy;
        forces[id] -= damping_eff * dvx;
        forces[id + off] -= damping_eff * dvy;
        pos[id] += vel[id] * dt;
        pos[id + off] += vel[id + off] * dt;
    }
}
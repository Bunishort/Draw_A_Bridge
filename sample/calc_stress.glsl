#version 430

layout (local_size_x = 16, local_size_y = 16) in;

// --- MAPPING DES BUFFERS (Strictement identique à votre stack Python) ---
layout(std430, binding = 0) buffer b_pos { float pos[]; };
layout(std430, binding = 1) buffer b_vel { float vel[]; };
layout(std430, binding = 2) buffer b_stress_old { float s_old[]; };
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };
layout(std430, binding = 7) buffer b_masks_flt { float m_flt[]; };

uniform int width; uniform int height;
uniform float lm; uniform float coef; uniform float elas_lambda_ratio;
uniform float explicit_b; uniform float G0;
uniform float visco_fact_1; uniform float visco_fact_2;

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

        //  Mise à jour viscoélastique
        // Mise à jour du buffer permanent pour l'itération suivante (sxx_x_old = sxx_x)
        s_old[id]         *= visco_fact_2;
        s_old[id]         += sxx * visco_fact_1;
        s_old[id + off]   *= visco_fact_2;
        s_old[id + off]   += sxy_x * visco_fact_1;
        s_old[id + 2*off] *= visco_fact_2;
        s_old[id + 2*off] += syy * visco_fact_1;
        s_old[id + 3*off] *= visco_fact_2;
        s_old[id + 3*off] += sxy_y * visco_fact_1;
    }
}
#version 430

layout (local_size_x = 16, local_size_y = 16) in;

// --- MAPPING DES BUFFERS (Strictement identique à votre stack Python) ---
layout(std430, binding = 0) buffer b_pos { float pos[]; };
layout(std430, binding = 1) buffer b_vel { float vel[]; };
layout(std430, binding = 3) buffer b_force { float forces[]; };
layout(std430, binding = 4) buffer b_ext { float ext_forces[]; };
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };
layout(std430, binding = 6) buffer b_stress_curr { float s_curr[]; };

uniform int width; uniform int height;
uniform float lm; uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

int offset() { return width * height; }
int idx(int i, int j) { return i * width + j; }

void main() {
    int j = int(gl_GlobalInvocationID.x);
    int i = int(gl_GlobalInvocationID.y);
    int id = idx(i, j);
    int off = offset();

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
        float dvx = (c_sxx_dx + c_sxy_dy) * m - ext_forces[id] + forces[id];     // a_u_x - bx - damping force
        float dvy = (c_syy_dy + c_sxy_dx) * m - ext_forces[id + off] + forces[id+off]; // a_u_y - by - damping force

        dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

        vel[id] += dvx;
        vel[id + off] += dvy;
        forces[id] -= damping_eff * dvx;
        forces[id + off] -= damping_eff * dvy;
        pos[id] += vel[id] * dt;
        pos[id + off] += vel[id + off] * dt;
    }
}
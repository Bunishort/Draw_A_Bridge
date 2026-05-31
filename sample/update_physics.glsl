#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rg32f, binding = 4) uniform image2D img_ext;
layout(rg8, binding = 5) uniform image2D img_masks;
layout(rgba32f, binding = 2) uniform image2D img_stress_curr;

uniform int width; uniform int height;
uniform float lm; uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;
//Variables for attractor computation
uniform float u_mouse_active;
uniform float u_mouse_col; // Position X de la souris
uniform float u_mouse_row; // Position Y de la souris
uniform float u_f_attract;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.y >= 1 && p.x >= 1 && p.y < height && p.x < width) {
        // Loading stress values for divergence computation

        vec4 s_prev_prev = imageLoad(img_stress_curr, p + ivec2(-1, -1));
        vec4 s_curr_prev = imageLoad(img_stress_curr, p + ivec2(-1, 0));
        vec4 s_prev_curr = imageLoad(img_stress_curr, p + ivec2(0, -1));

        // Calculating stress divergence
        // c_sxx_dx = -sxx(i-1, j-1) + sxx(i, j-1)
        float c_sxx_dx = -s_prev_prev.x + s_curr_prev.x;
        // c_sxy_dx = -sxy_x(i-1, j-1) + sxy_x(i, j-1)
        float c_sxy_dx = -s_prev_prev.y + s_curr_prev.y;
        // c_sxy_dy = -sxy_y(i-1, j-1) + sxy_y(i-1, j)
        float c_sxy_dy = -s_prev_prev.w + s_prev_curr.w;
        // c_syy_dy = -syy(i-1, j-1) + syy(i-1, j)
        float c_syy_dy = -s_prev_prev.z + s_prev_curr.z;

        float m = imageLoad(img_masks, p).y / lm; //  solid_not_uimp /lm

        vec2 f_ext = imageLoad(img_ext, p).xy; // external forces
        vec4 pv = imageLoad(img_pos_vel, p); // previous position and velocities

        //UPdate external forces with attractor
        if (u_mouse_active > 0.0) {
            // Coordonnées déformées du point courant
            float def_col = float(p.x) + pv.y / lm;
            float def_row = float(p.y) + pv.x / lm;

            // Distances avec la souris
            float dx_col = u_mouse_col - def_col;
            float dy_row = u_mouse_row - def_row;

            float dist = sqrt(dx_col * dx_col + dy_row * dy_row) + 1.0;
            float f_mag = u_f_attract / dist;

            // On convertit la force en b (rappel de ton CPU : b = -f * solid)
            // bx s'applique à l'axe des lignes (ux) -> dy_row
            // by s'applique à l'axe des colonnes (uy) -> dx_col
            f_ext.x -= (f_mag * dy_row / dist);
            f_ext.y -= (f_mag * dx_col / dist);
        }

        float dvx = (c_sxx_dx + c_sxy_dy) * m - f_ext.x - damping_eff * pv.z; // velocity variation
        float dvy = (c_syy_dy + c_sxy_dx) * m - f_ext.y - damping_eff * pv.w;

        dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

        // Updating speed and position (euler integration)
        pv.z += dvx;      // vx
        pv.w += dvy;      // vy
        pv.x += pv.z * dt; // ux
        pv.y += pv.w * dt; // uy

        imageStore(img_pos_vel, p, pv);
    }
}
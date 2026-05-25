#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rg32f, binding = 4) uniform image2D img_ext;
layout(rg8, binding = 5) uniform image2D img_masks;
layout(rgba32f, binding = 2) uniform image2D img_stress_curr;

uniform int width; uniform int height;
uniform float lm; uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.y >= 1 && p.x >= 1 && p.y < height && p.x < width) {
        // Mapping Buffer vers Texture :
        // idx(i-1, j-1) -> p + ivec2(-1, -1)
        // idx(i, j-1)   -> p + ivec2(-1, 0)  (Même ligne, col préc.)
        // idx(i-1, j)   -> p + ivec2(0, -1)  (Même col, ligne préc.)

        vec4 s_prev_prev = imageLoad(img_stress_curr, p + ivec2(-1, -1));
        vec4 s_curr_prev = imageLoad(img_stress_curr, p + ivec2(-1, 0));
        vec4 s_prev_curr = imageLoad(img_stress_curr, p + ivec2(0, -1));

        // c_sxx_dx = -sxx(i-1, j-1) + sxx(i, j-1)
        float c_sxx_dx = -s_prev_prev.x + s_curr_prev.x;
        // c_sxy_dx = -sxy_x(i-1, j-1) + sxy_x(i, j-1)
        float c_sxy_dx = -s_prev_prev.y + s_curr_prev.y;
        // c_sxy_dy = -sxy_y(i-1, j-1) + sxy_y(i-1, j)
        float c_sxy_dy = -s_prev_prev.w + s_prev_curr.w;
        // c_syy_dy = -syy(i-1, j-1) + syy(i-1, j)
        float c_syy_dy = -s_prev_prev.z + s_prev_curr.z;

        int id = p.y * width + p.x;

        if(imageLoad(img_masks, p).y > 0.5){
            vec2 f_ext = imageLoad(img_ext, p).xy;
            vec4 pv = imageLoad(img_pos_vel, p);

            float dvx = (c_sxx_dx + c_sxy_dy) / lm - f_ext.x - damping_eff * pv.z;
            float dvy = (c_syy_dy + c_sxy_dx) / lm - f_ext.y - damping_eff * pv.w;

            dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

            pv.z += dvx;// vx
            pv.w += dvy;// vy
        }
        pv.x += pv.z * dt; // ux
        pv.y += pv.w * dt; // uy

        imageStore(img_pos_vel, p, pv);
    }
}
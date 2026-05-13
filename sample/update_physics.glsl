#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rg32f, binding = 4) uniform image2D img_ext;
layout(std430, binding = 5) buffer b_masks_int { int m_int[]; };
layout(rgba32f, binding = 2) uniform image2D img_stress_curr;

uniform int width; uniform int height;
uniform float lm; uniform float dt_by_vol_mass; uniform float damping_eff; uniform float dt;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.y >= 1 && p.x >= 1 && p.y < height && p.x < width) {
        // Accès aux voisins directs via ivec2
        vec4 s00 = imageLoad(img_stress_curr, p + ivec2(-1, -1));
        vec4 s10 = imageLoad(img_stress_curr, p + ivec2(0, -1));
        vec4 s01 = imageLoad(img_stress_curr, p + ivec2(-1, 0));

        // s.x=sxx, s.y=sxy_x, s.z=syy, s.w=sxy_y
        float c_sxx_dx = -s00.x + s10.x;
        float c_sxy_dx = -s00.y + s10.y;
        float c_sxy_dy = -s00.w + s01.w;
        float c_syy_dy = -s00.z + s01.z;

        int id = p.y * width + p.x;
        float m = float(m_int[id + 8*(width*height)]) / lm;

        vec2 f_ext = imageLoad(img_ext, p).xy;
        vec4 pv = imageLoad(img_pos_vel, p); // x=ux, y=uy, z=vx, w=vy

        float dvx = (c_sxx_dx + c_sxy_dy) * m - f_ext.x - damping_eff * pv.z;
        float dvy = (c_syy_dy + c_sxy_dx) * m - f_ext.y - damping_eff * pv.w;

        dvx *= dt_by_vol_mass; dvy *= dt_by_vol_mass;

        pv.z += dvx;      // New vx
        pv.w += dvy;      // New vy
        pv.x += pv.z * dt; // New ux
        pv.y += pv.w * dt; // New uy

        imageStore(img_pos_vel, p, pv);
    }
}
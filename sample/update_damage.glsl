#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rg32f, binding = 7) uniform image2D img_damage;
layout(rgba32f, binding = 2) uniform image2D img_stress_curr;
layout(rg8, binding = 5) uniform image2D img_masks;

uniform int width; uniform int height;
uniform float damage_stress; uniform float dt_by_tau_filter;
uniform float nu; uniform float dt_by_tau_damage;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    vec2 masks = imageLoad(img_masks, p).xy;

    //Calculate only if solid_not_uimp
    if (p.y >= 1 && p.x >= 1 && p.y < height && p.x < width && masks.y > 0.5) {
        // Loading stress values for divergence computation

        vec4 s_prev_prev = imageLoad(img_stress_curr, p + ivec2(-1, -1));
        vec4 s_curr_prev = imageLoad(img_stress_curr, p + ivec2(-1, 0));
        vec4 s_prev_curr = imageLoad(img_stress_curr, p + ivec2(0, -1));

        // Calculating stress average on each cell
        // sxx = 0.5*( sxx(i-1, j-1) + sxx(i, j-1))
        float sxx = 0.5 * (s_prev_prev.x + s_curr_prev.x);
        // sxy = 0.5*(sxy_x(i-1, j-1) + sxy_x(i, j-1))
        float sxy = 0.25 * (s_prev_prev.y + s_curr_prev.y + s_prev_prev.w + s_prev_curr.w);
        // syy = 0.5 * (syy(i-1, j-1) + syy(i-1, j))
        float syy = 0.5 * (s_prev_prev.z + s_prev_curr.z);

        float szz = nu * (sxx + syy);

        // Mean stress (hydrostatic component)
        float pp = (sxx + syy + szz) / 3.0;

        // Deviatoric stress tensor
        sxx = sxx - pp;
        syy = syy - pp;
        szz = szz - pp;

        // Second invariant of the deviatoric stress tensor
        float J2 = 0.5 * (sxx * sxx + syy * syy + szz * szz)
                   + sxy * sxy;

        // Von Mises equivalent stress
        float VMstress = sqrt(3.0 * J2);

        vec2 damage = imageLoad(img_damage, p).xy; // damage map

        float damage_old = damage.x * dt_by_tau_damage / dt_by_tau_damage;
        VMstress *= dt_by_tau_filter;
        damage.x *= (1 - dt_by_tau_filter);
        damage.x += VMstress;

        if (damage.x > damage_stress && damage.x > damage_old) {
            damage.y += (damage.x - damage_old) * (damage.x - damage_stress) / damage_stress / damage_stress;
        }

        imageStore(img_damage, p, vec4(damage, 0.0, 0.0));
    }
}
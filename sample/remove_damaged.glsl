#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rg32f, binding = 7) uniform image2D img_damage;
layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rg32f, binding = 4) uniform image2D img_ext;
layout(rg8, binding = 5) uniform image2D img_masks;

uniform int width; uniform int height; uniform float scalex; uniform float scaley;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    if (p.y >= 1 && p.x >= 1 && p.y < height && p.x < width) {
        // Loading stress values for divergence computation

        vec2 damage = imageLoad(img_damage, p).xy; // damage map
        vec4 dep = imageLoad(img_pos_vel, p);// ux uy vx vy

        vec2 final_pos = dep.xy;
        final_pos.x *= -scalex;
        final_pos.y *= scaley;

        final_pos.x += p.y;//x and y inversion because... idk
        final_pos.y += p.x;

        vec2 masks = imageLoad(img_masks, p).xy;
        vec2 ext = imageLoad(img_ext, p).xy;

        // Remove solid if damaged or out of screen
        if (damage.y > 1 || final_pos.x < 1 || final_pos.y <1
            || final_pos.x > height || final_pos.y > width) {



            // Solid and solid not uimp 0
            masks.x = 0;
            masks.y = 0;

            // Position and speed 0
            dep.x = 0.0;
            dep.y = 0.0;
            dep.z = 0.0;
            dep.w = 0.0;

            // External forces 0
            ext.x = 0.0;
            ext.y = 0.0;

        }
        imageStore(img_pos_vel, p, dep);
        imageStore(img_masks, p, vec4(masks, 0, 0));
        imageStore(img_ext, p, vec4(ext, 0.0, 0.0));

    }
}
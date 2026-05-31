#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rgba32f, binding = 0) uniform image2D img_pos_vel;
layout(rg32f, binding = 6) uniform image2D img_att;
layout(rg32f, binding = 4) uniform image2D img_ext;

layout(rg8, binding = 5) uniform image2D img_masks; // solid matrix

uniform int width; uniform int height;

//Variables for attractor computation
uniform float lm;
uniform float u_mouse_active;
uniform float u_mouse_col; // Position X de la souris
uniform float u_mouse_row; // Position Y de la souris
uniform float u_f_attract;

void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    // Do not compute if outside of physical image
    if (p.x >= width || p.y >= height) {
        return;
    }

    vec2 solid = imageLoad(img_masks, p).xy;
    if (solid.y >0.0) {

        //Update external forces with attractor
        vec2 f_att = imageLoad(img_att, p).xy;// previous attractor forces
        vec2 f_ext = imageLoad(img_ext, p).xy;// external forces
        vec4 pv = imageLoad(img_pos_vel, p);// position and velocities
        if (u_mouse_active > 0.0) {
            // Coordonnées déformées du point courant
            float def_col = float(p.x) + pv.y / lm;
            float def_row = float(p.y) + pv.x / lm;

            // Distances avec la souris
            float dx_col = u_mouse_col - def_col;
            float dy_row = u_mouse_row - def_row;

            float dist = sqrt(dx_col * dx_col + dy_row * dy_row) + 1.0;
            float f_mag = u_f_attract / dist;

            f_ext.x -= f_att.x;
            f_ext.y -= f_att.y;
            f_att.x = -(f_mag * dy_row / dist);
            f_att.y = -(f_mag * dx_col / dist);
            f_ext.x += f_att.x;
            f_ext.y += f_att.y;
        }
        else {
            f_ext.x -= f_att.x;
            f_ext.y -= f_att.y;
            f_att.x = 0.0;
            f_att.y = 0.0;
        }
        //Writing the results into buffer
        imageStore(img_att, p, vec4(f_att, 0.0, 0.0));
        imageStore(img_ext, p, vec4(f_ext, 0.0, 0.0));
    }
}
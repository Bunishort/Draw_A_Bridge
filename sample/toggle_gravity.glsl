#version 430
layout (local_size_x = 16, local_size_y = 16) in;

layout(rg32f, binding = 4) uniform image2D img_ext;

layout(rg8, binding = 5) uniform image2D img_masks; // solid matrix

uniform int width; uniform int height;

//Variables for gravity computation
uniform float fx;
uniform float fy;
uniform float activate;


void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);

    // Do not compute if outside of physical image
    if (p.x >= width || p.y >= height) {
        return;
    }

    vec2 solid = imageLoad(img_masks, p).xy;
    if (solid.y >0.0) {

        //Update external forces
        vec2 f_ext = imageLoad(img_ext, p).xy;// external forces

        // Using -fx and -fy instead of fx/fy to be consistent with calc_b function
        if (activate >0.0) {
            f_ext.x += -fx;
            f_ext.y += -fy;
        }
        else {
            f_ext.x -= -fx;
            f_ext.y -= -fy;
        }
        //Writing the results into buffer
        imageStore(img_ext, p, vec4(f_ext, 0.0, 0.0));
    }
}
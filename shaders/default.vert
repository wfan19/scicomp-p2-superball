#version 330

layout (location=0) in vec3 in_position;
layout (location=1) in vec2 in_textcoord;
layout (location=2) in vec3 in_normal;

out vec2 uv;
out vec3 normal;
out vec3 frag_pos;

uniform mat4 mat_projection;
uniform mat4 mat_view;  // Matrix representing camera_T_camera_world
uniform mat4 mat_model; // Matrix representing world_T_world_model

void main() {
    gl_Position = mat_projection * mat_view * mat_model * vec4(in_position, 1.0);  // Transform world frame points to camera/image frame

    frag_pos = vec3(mat_model * vec4(in_position, 1.0)); 
    normal = mat3(transpose(inverse(mat_model))) * normalize(in_normal);
    uv = in_textcoord;
}
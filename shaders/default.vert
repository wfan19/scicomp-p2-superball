#version 330

layout (location=0) in vec3 in_position;
layout (location=1) in vec2 in_textcoord;

out vec2 uv;

uniform mat4 mat_projection;
uniform mat4 mat_view;  // Matrix representing camera_T_camera_world
uniform mat4 mat_model; // Matrix representing world_T_world_model

void main() {
    gl_Position = mat_projection * mat_view * mat_model * vec4(in_position, 1.0);
    uv = in_textcoord;
}
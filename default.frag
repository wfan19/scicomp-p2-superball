#version 330

layout (location=0) out vec4 out_color;

in vec2 uv;

void main() {
    vec3 color = vec3(mod(uv.x / uv.y, 1), mod(uv.x, 2), mod(uv.y, 2));
    out_color = vec4(color, 1.0);
}
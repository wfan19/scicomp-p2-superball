#version 330

layout (location=0) out vec4 out_color;

in vec2 uv;
in vec3 normal;
in vec3 frag_pos;

struct Light {
    vec3 position;
    vec3 Ia;
    vec3 Id;
    vec3 Is;
};

uniform Light light;
uniform vec3 color;
uniform vec3 cam_posn;

vec3 get_light(vec3 color) {
    // ====== Implementaiton of Phong Lighting ======

    // Calculate ambient component
    vec3 ambient = light.Ia;
    
    // Calculate diffuse component
    vec3 pt_to_light = normalize(light.position - frag_pos);
    float alignment = max(0, dot(pt_to_light, normal));
    vec3 diffuse = light.Id * alignment;

    // Calculate specular component
    vec3 pt_to_cam = normalize(cam_posn - frag_pos);
    vec3 reflected_light = reflect(pt_to_light, normal);
    float reflection = pow(dot(pt_to_cam, reflected_light), 32);
    vec3 specular = light.Is * reflection;
    

    return color * (ambient + diffuse + specular);
}

void main() {
    out_color = vec4(get_light(color), 1.0);
}
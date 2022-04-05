# version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Uniforms {
    mat4 model_matrix;
    mat4 light_space_matrix;
};

void main() {
    gl_Position = light_space_matrix * model_matrix * vec4(position, 1.0);
}
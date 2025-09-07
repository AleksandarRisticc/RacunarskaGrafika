#version 330 core
layout(location=0) in vec3 aPos;     // pozicija (model space)
layout(location=1) in vec3 aNormal;  // normal po licu
layout(location=2) in vec2 aUV;      // UV za gornju plohu
layout(location=3) in float aTop;    // 1.0 = gornja; 0.0 = bočne/donja

uniform mat4 uModel;
uniform mat4 uVP;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUV;
out float vTop;

void main() {
    mat3 nrmMat = mat3(transpose(inverse(uModel)));
    vec4 wp     = uModel * vec4(aPos, 1.0);
    vWorldPos   = wp.xyz;
    vWorldNormal= normalize(nrmMat * aNormal);
    vUV         = aUV;
    vTop        = aTop;
    gl_Position = uVP * wp;
}

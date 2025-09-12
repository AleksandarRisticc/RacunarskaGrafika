#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in float aTop;

uniform mat4 uModel;
uniform mat4 uVP;
uniform mat4 uLightVP;

out vec3 vWorldPos;
out vec3 vWorldNormal;
out vec2 vUV;
out float vTop;
out vec4 vShadowCoord;

void main() {
    mat3 nrmMat = mat3(transpose(inverse(uModel)));
    vec4 wp     = uModel * vec4(aPos, 1.0);
    vWorldPos   = wp.xyz;
    vWorldNormal= normalize(nrmMat * aNormal);
    vUV         = aUV;
    vTop        = aTop;
    vShadowCoord= uLightVP * wp;
    gl_Position = uVP * wp;
}

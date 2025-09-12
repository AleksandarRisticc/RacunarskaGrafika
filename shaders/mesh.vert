#version 330 core
layout(location=0) in vec3  aPos;
layout(location=1) in vec3  aNormal;
layout(location=2) in vec2  aUV;
layout(location=3) in float aTop;

uniform mat4 uModel, uView, uProj;
uniform mat4 uLightVP;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;
out vec4 vShadowCoord;

void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;
    vNormal   = mat3(transpose(inverse(uModel))) * aNormal;
    vUV       = aUV;
    vShadowCoord = uLightVP * wp;
    gl_Position = uProj * uView * wp;
}

#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
in vec4 vShadowCoord;
out vec4 FragColor;

uniform vec3  uLightPos;
uniform vec3  uLightDir;
uniform float uSpotCosInner;
uniform float uSpotCosOuter;
uniform vec3  uAlbedo;
uniform vec3  uCamPos;
uniform sampler2D uRoughTex;
uniform float uTexTiling;

uniform sampler2D uShadowMap;
uniform int   uShadowsOn;

float pcfShadow(vec3 uvw, float bias){
    if(uvw.z > 1.0) return 0.0;
    vec2 texel = 1.0 / vec2(textureSize(uShadowMap,0));
    float s=0.0;
    for(int dy=-1; dy<=1; ++dy)
    for(int dx=-1; dx<=1; ++dx){
        float p = texture(uShadowMap, uvw.xy + vec2(dx,dy)*texel).r;
        s += (uvw.z - bias > p ? 1.0 : 0.0);
    }
    return s/9.0;
}

void main(){
    vec3 N = normalize(vNormal);
    vec3 L = normalize(uLightPos - vWorldPos);
    vec3 V = normalize(uCamPos - vWorldPos);
    vec3 H = normalize(L + V);

    float cd   = dot(-L, normalize(uLightDir));
    float spot = smoothstep(uSpotCosOuter, uSpotCosInner, cd);

    float diff  = max(dot(N, L), 0.0) * spot + 0.18*(1.0-spot);

    float rough = texture(uRoughTex, vUV * uTexTiling).r;
    float shin  = mix(64.0, 8.0, rough);
    float specW = mix(1.0, 0.25, rough);
    float spec  = pow(max(dot(N, H), 0.0), shin) * specW * spot;

    vec3  uvw   = (vShadowCoord.xyz / vShadowCoord.w) * 0.5 + 0.5;
    float bias  = max(0.0015*(1.0 - dot(N,L)), 0.0005);
    float sh    = (uShadowsOn!=0) ? pcfShadow(uvw, bias) : 0.0;

    float shade = mix(1.0, 0.35, sh);
    vec3 col = (uAlbedo * diff + vec3(1.0)*spec) * shade;
    FragColor = vec4(col, 1.0);
}

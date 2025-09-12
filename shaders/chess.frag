#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUV;
in float vTop;
in vec4 vShadowCoord;
out vec4 FragColor;

uniform int   uTiles;
uniform vec3  colorA;
uniform vec3  colorB;
uniform vec3  uSideColor;
uniform vec3  uLightPos;
uniform vec3  uLightDir;
uniform float uAmbient;

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

void main() {
    vec3 L = normalize(uLightPos - vWorldPos);
    float nDotL = max(dot(normalize(vWorldNormal), L), 0.0);
    float cd    = dot(-L, normalize(uLightDir));
    float spot  = smoothstep( cos(radians(55.0)), cos(radians(45.0)), cd );
    float lighting = clamp(uAmbient + nDotL * spot, 0.0, 1.0);

    vec3 base;
    if (vTop > 0.5) {
        int x = int(floor(vUV.x * uTiles));
        int y = int(floor(vUV.y * uTiles));
        vec3 tile = ((x + y) % 2 == 0) ? colorA : colorB;

        float fx   = fract(vUV.x * uTiles);
        float fy   = fract(vUV.y * uTiles);
        float dist = min(min(fx, 1.0 - fx), min(fy, 1.0 - fy));
        float edge = smoothstep(0.002, 0.010, dist);
        vec3 borderColor = vec3(0.05);
        base = mix(borderColor, tile, edge);
    } else {
        base = uSideColor;
    }

    vec3 uvw = (vShadowCoord.xyz / vShadowCoord.w) * 0.5 + 0.5;
    float sh = (uShadowsOn!=0) ? pcfShadow(uvw, 0.0008) : 0.0;
    float shade = mix(1.0, 0.40, sh);

    FragColor = vec4(base * lighting * shade, 1.0);
}

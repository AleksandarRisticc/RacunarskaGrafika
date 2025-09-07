#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUV;
in float vTop;

out vec4 FragColor;

uniform int   uTiles     = 8;                          // broj polja po stranici
uniform vec3  colorA     = vec3(0.93, 0.90, 0.78);     // svetlo drvo
uniform vec3  colorB     = vec3(0.35, 0.22, 0.12);     // tamno drvo
uniform vec3  uSideColor = vec3(0.22, 0.15, 0.09);     // boja debljine (stranice+donja)
uniform vec3  uLightDir  = normalize(vec3(-0.4, 1.0, 0.3));
uniform float uAmbient   = 0.50;

void main() {
    // Lambert osvetljenje
    float nDotL   = max(dot(normalize(vWorldNormal), normalize(uLightDir)), 0.0);
    float lighting = clamp(uAmbient + nDotL, 0.0, 1.0);

    if (vTop > 0.5) {
        // šahovnica + tanka AA ivica između polja
        int x = int(floor(vUV.x * uTiles));
        int y = int(floor(vUV.y * uTiles));
        vec3 tile = ((x + y) % 2 == 0) ? colorA : colorB;

        float fx   = fract(vUV.x * uTiles);
        float fy   = fract(vUV.y * uTiles);
        float dist = min(min(fx, 1.0 - fx), min(fy, 1.0 - fy));
        float edge = smoothstep(0.002, 0.010, dist); // prilagodi debljinu ivice po ukusu
        vec3 borderColor = vec3(0.05);

        vec3 col = mix(borderColor, tile, edge);
        FragColor = vec4(col * lighting, 1.0);
    } else {
        // stranice/dno — puna boja (ako želiš osvetljenje, zameni sa *lighting)
        FragColor = vec4(uSideColor, 1.0);
        // FragColor = vec4(uSideColor * lighting, 1.0); // opcija sa osvetljenjem
    }
}

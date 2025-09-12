// Main.cpp — OpenGL 3.3 (GLFW + GLEW + GLM)
// Proceduralna tabla, dva topa (hijerarhija), billboard čestice,
// Start/Exit GUI + tajmer, + VISEĆA SPOTLIGHT LAMPA SA SHADOW MAP-om.
//
// Kontrole (tokom igre):
//  - Beli:  strelice  (UP -> -Z, DOWN -> +Z; LEFT/RIGHT X)
//  - Crni:  WASD      (W -Z, S +Z, A -X, D +X)
//  - R: respawn, Esc: quit

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cctype>

#ifdef _WIN32
#include <windows.h>
#include <mmsystem.h>
#include <filesystem>
#include <iostream>
#pragma comment(lib, "winmm.lib")

static inline std::wstring exeDir() {
    wchar_t buf[MAX_PATH];
    DWORD n = GetModuleFileNameW(NULL, buf, MAX_PATH);
    std::wstring path(buf, n);
    size_t slash = path.find_last_of(L"\\/");
    return (slash == std::wstring::npos) ? L"." : path.substr(0, slash);
}

static inline void play_wav(const wchar_t* nameOrPath) {
    // Ako je prosleđeno samo ime, probaj pored .exe (build/bin dir)
    std::wstring path(nameOrPath);
    if (path.find(L'\\') == std::wstring::npos && path.find(L'/') == std::wstring::npos) {
        path = exeDir() + L"\\" + path;
    }
    if (!std::filesystem::exists(path)) {
        // Ne pišti – samo prijavi u konzoli.
        std::wcerr << L"[SFX] Not found: " << path << L"\n";
        return;
    }
    BOOL ok = PlaySoundW(path.c_str(), NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
    if (!ok) {
        std::wcerr << L"[SFX] Play failed: " << path << L"\n";
    }
}
#endif


// ---------- POST: globals ----------
static GLuint sceneFBO=0, sceneColor=0, sceneDepth=0;
static GLuint ppFBO[2]={0,0}, ppTex[2]={0,0};
static GLuint fsVAO=0, fsVBO=0;

static GLuint progBright=0, progBlur=0, progCombine=0;

static bool gGLReady = false;
static bool gBloomOn = true;

static int ppW=0, ppH=0;

#ifdef _WIN32
static const wchar_t* SFX_CLICK     = L"click.wav";
static const wchar_t* SFX_END       = L"end.wav";
static const wchar_t* SFX_EXPLOSION = L"explosion.wav";
static const wchar_t* SFX_MOVE      = L"movement.wav";
#endif

static void makeFullscreenQuad(){
    if(fsVAO) return;
    const float quad[] = {
        // pos      // uv
        -1.f,-1.f, 0.f,0.f,
         1.f,-1.f, 1.f,0.f,
         1.f, 1.f, 1.f,1.f,
        -1.f,-1.f, 0.f,0.f,
         1.f, 1.f, 1.f,1.f,
        -1.f, 1.f, 0.f,1.f
    };
    glGenVertexArrays(1,&fsVAO);
    glGenBuffers(1,&fsVBO);
    glBindVertexArray(fsVAO);
    glBindBuffer(GL_ARRAY_BUFFER, fsVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,4*sizeof(float),(void*)(2*sizeof(float)));
    glBindVertexArray(0);
}

static void destroyPost(){
    if(sceneFBO){ glDeleteFramebuffers(1,&sceneFBO); sceneFBO=0; }
    if(sceneColor){ glDeleteTextures(1,&sceneColor); sceneColor=0; }
    if(sceneDepth){ glDeleteRenderbuffers(1,&sceneDepth); sceneDepth=0; }
    for(int i=0;i<2;i++){
        if(ppFBO[i]){ glDeleteFramebuffers(1,&ppFBO[i]); ppFBO[i]=0; }
        if(ppTex[i]){ glDeleteTextures(1,&ppTex[i]); ppTex[i]=0; }
    }
}

static void createPostTargets(int w,int h){
    if(w<1) w=1; if(h<1) h=1;
    if(ppW==w && ppH==h && sceneFBO) return;
    destroyPost();
    ppW=w; ppH=h;

    // Scene HDR FBO (RGBA16F + depth)
    glGenFramebuffers(1,&sceneFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);

    glGenTextures(1,&sceneColor);
    glBindTexture(GL_TEXTURE_2D, sceneColor);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w,h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneColor, 0);

    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    glGenRenderbuffers(1,&sceneDepth);
    glBindRenderbuffer(GL_RENDERBUFFER, sceneDepth);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w,h);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, sceneDepth);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
        std::cerr<<"Scene FBO incomplete!\n";

    // Ping-pong FBOs (RGBA16F, no depth)
    glGenFramebuffers(2, ppFBO);
    glGenTextures(2, ppTex);
    for(int i=0;i<2;i++){
        glBindFramebuffer(GL_FRAMEBUFFER, ppFBO[i]);
        glBindTexture(GL_TEXTURE_2D, ppTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w,h, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ppTex[i], 0);
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE)
            std::cerr<<"PP FBO "<<i<<" incomplete!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


static const float BOARD_SIZE  = 8.4f;
static const int   BOARD_TILES = 8;

static const float LIGHT_HEIGHT = 4.2f;        // visina lampe iznad vrha table (y≈0)
static const int   SHADOW_RES   = 1024;        // rezolucija depth mape

//================ Shader helperi ================
static GLuint compile_src(GLenum type, const char* src){
    GLuint s = glCreateShader(type);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    GLint ok=GL_FALSE; glGetShaderiv(s,GL_COMPILE_STATUS,&ok);
    if(!ok){
        GLint len=0; glGetShaderiv(s,GL_INFO_LOG_LENGTH,&len);
        std::string log(len,'\0'); glGetShaderInfoLog(s,len,nullptr,log.data());
        std::cerr << (type==GL_VERTEX_SHADER? "Vertex":"Fragment")
                  << " shader error:\n" << log << std::endl;
        glDeleteShader(s); return 0;
    }
    return s;
}
static GLuint link_src(const char* vsrc, const char* fsrc){
    GLuint v=compile_src(GL_VERTEX_SHADER,vsrc);
    GLuint f=compile_src(GL_FRAGMENT_SHADER,fsrc);
    if(!v||!f){ if(v)glDeleteShader(v); if(f)glDeleteShader(f); return 0; }
    GLuint p=glCreateProgram(); glAttachShader(p,v); glAttachShader(p,f); glLinkProgram(p);
    GLint ok=GL_FALSE; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){
        GLint len=0; glGetProgramiv(p,GL_INFO_LOG_LENGTH,&len);
        std::string log(len,'\0'); glGetProgramInfoLog(p,len,nullptr,log.data());
        std::cerr<<"Link error:\n"<<log<<std::endl;
        glDeleteProgram(p); p=0;
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

//================ Mesh shader (figure) — spotlight + shadow ================
static const char* VERT_MESH = R"GLSL(
#version 330 core
layout(location=0) in vec3  aPos;
layout(location=1) in vec3  aNormal;
layout(location=2) in vec2  aUV;
layout(location=3) in float aTop; // unused

uniform mat4 uModel, uView, uProj;

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

void main(){
    vec4 wp = uModel * vec4(aPos,1.0);
    vWorldPos = wp.xyz;
    vNormal   = mat3(transpose(inverse(uModel))) * aNormal;
    vUV       = aUV;
    gl_Position = uProj * uView * wp;
}
)GLSL";

static const char* FRAG_MESH = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uAO; //ambient occlusion mapa

uniform vec3  uAlbedo;
uniform vec3  uCamPos;

// Roughness (procedural noise)
uniform sampler2D uRoughTex;
uniform float     uTexTiling;

// Spotlight
uniform vec3  uLightPos;
uniform vec3  uLightDir;     // normalizovan
uniform float uCosInner;
uniform float uCosOuter;

// Shadow map (spot perspective)
uniform sampler2D uShadowMap;
uniform mat4  uLightVP;
uniform vec2  uShadowTexel;  // 1.0/SHADOW_RES

// Ambient
uniform float uAmbient;

float shadowFactor(vec3 worldPos, vec3 N, vec3 L){
    // transform u light clip -> NDC -> [0,1]
    vec4 clip = uLightVP * vec4(worldPos, 1.0);
    vec3 ndc  = clip.xyz / clip.w;
    vec3 uvz  = ndc * 0.5 + 0.5;

    // van svetla / iza far plane-a: bez senke
    if(uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0 || uvz.z > 1.0) return 1.0;

    float current = uvz.z;
    float bias = max(0.0015, 0.003 * (1.0 - max(dot(normalize(N), normalize(L)), 0.0)));

    // PCF 3x3
    float occ = 0.0;
    for(int y=-1;y<=1;++y){
        for(int x=-1;x<=1;++x){
            vec2 off = vec2(x,y) * uShadowTexel;
            float depth = texture(uShadowMap, uvz.xy + off).r;
            occ += (current - bias) > depth ? 0.0 : 1.0;
        }
    }
    return occ / 9.0;
}

void main(){
    vec3 N = normalize(vNormal);

    // Spotlight lobe & attenuation
    vec3  toFrag = vWorldPos - uLightPos;
    float dist   = length(toFrag);
    vec3  L      = normalize(-toFrag);        // iz fragmenta ka svetlu
    float cosAng = dot(normalize(uLightDir), normalize(toFrag));
    float spot   = clamp((cosAng - uCosOuter) / max(uCosInner - uCosOuter, 1e-4), 0.0, 1.0);
    float atten  = 1.0 / (1.0 + 0.045*dist + 0.0075*dist*dist);

    // Difuzno + spekular (roughness → sjaj)
    float diff   = max(dot(N, L), 0.0);

    float rough  = texture(uRoughTex, vUV * uTexTiling).r;
    float ao     = texture(uAO, vUV * uTexTiling).r;
    ao = clamp(ao, 0.2, 1.0);
    float shin   = mix(64.0, 8.0, rough);
    float specW  = mix(1.0, 0.25, rough);

    vec3  V      = normalize(uCamPos - vWorldPos);
    vec3  H      = normalize(L + V);
    float spec   = pow(max(dot(N, H), 0.0), shin) * specW;

    // Shadow
    float shadow = shadowFactor(vWorldPos, N, L);

    vec3  light  = (uAmbient * ao + (diff + spec) * atten * spot * shadow) * vec3(1.0);
    vec3  col    = uAlbedo * light;

    FragColor = vec4(col, 1.0);
}
)GLSL";

//================ Procedural CHESS shader (tabla) — spotlight + shadow =====
static const char* VERT_CHESS = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aUV;
layout(location=3) in float aTop;

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
)GLSL";

static const char* FRAG_CHESS = R"GLSL(
#version 330 core
in vec3 vWorldPos;
in vec3 vWorldNormal;
in vec2 vUV;
in float vTop;

out vec4 FragColor;

uniform int   uTiles;
uniform vec3  colorA;      // bela
uniform vec3  colorB;      // crna
uniform vec3  uSideColor;  // braon stranice/dno

// Spotlight
uniform vec3  uLightPos;
uniform vec3  uLightDir;
uniform float uCosInner;
uniform float uCosOuter;

// Shadow map
uniform sampler2D uShadowMap;
uniform mat4  uLightVP;
uniform vec2  uShadowTexel;

// Ambient
uniform float uAmbient;

float shadowFactor(vec3 worldPos, vec3 N, vec3 L){
    vec4 clip = uLightVP * vec4(worldPos, 1.0);
    vec3 ndc  = clip.xyz / clip.w;
    vec3 uvz  = ndc * 0.5 + 0.5;
    if(uvz.x < 0.0 || uvz.x > 1.0 || uvz.y < 0.0 || uvz.y > 1.0 || uvz.z > 1.0) return 1.0;

    float current = uvz.z;
    float bias = max(0.0015, 0.003 * (1.0 - max(dot(normalize(N), normalize(L)), 0.0)));

    float occ = 0.0;
    for(int y=-1;y<=1;++y){
        for(int x=-1;x<=1;++x){
            vec2 off = vec2(x,y) * uShadowTexel;
            float depth = texture(uShadowMap, uvz.xy + off).r;
            occ += (current - bias) > depth ? 0.0 : 1.0;
        }
    }
    return occ / 9.0;
}

void main() {
    vec3 N = normalize(vWorldNormal);

    vec3  toFrag = vWorldPos - uLightPos;
    float dist   = length(toFrag);
    vec3  Ldir   = normalize(-toFrag);
    float cosAng = dot(normalize(uLightDir), normalize(toFrag));
    float spot   = clamp((cosAng - uCosOuter) / max(uCosInner - uCosOuter, 1e-4), 0.0, 1.0);
    float atten  = 1.0 / (1.0 + 0.045*dist + 0.0075*dist*dist);

    float diff   = max(dot(N, Ldir), 0.0);
    float shade  = (uAmbient + diff * atten * spot * shadowFactor(vWorldPos, N, Ldir));

    if (vTop > 0.5) {
        int x = int(floor(vUV.x * uTiles));
        int y = int(floor(vUV.y * uTiles));
        vec3 tile = ((x + y) % 2 == 0) ? colorA : colorB;

        float fx   = fract(vUV.x * uTiles);
        float fy   = fract(vUV.y * uTiles);
        float distE= min(min(fx, 1.0 - fx), min(fy, 1.0 - fy));
        float edge = smoothstep(0.002, 0.010, distE);
        vec3 borderColor = vec3(0.05);

        vec3 col = mix(borderColor, tile, edge);
        FragColor = vec4(col * shade, 1.0);
    } else {
        FragColor = vec4(uSideColor * shade, 1.0);
    }
}
)GLSL";

//================ UI (2D) — obojeni trugao + 3x5 “pixel font” ==============
static const char* VERT_UI = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aCol;
uniform vec2 uScreen; // (w,h) u pikselima
out vec4 vCol;
void main(){
    vec2 ndc = (aPos / uScreen) * 2.0 - 1.0;
    ndc.y = -ndc.y; // flip Y
    gl_Position = vec4(ndc, 0.0, 1.0);
    vCol = aCol;
}
)GLSL";

static const char* FRAG_UI = R"GLSL(
#version 330 core
in vec4 vCol;
out vec4 FragColor;
void main(){ FragColor = vCol; }
)GLSL";

//================ UNLIT (emissive) shader za lampu =========================
static const char* VERT_UNLIT = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uModel, uView, uProj;
void main(){ gl_Position = uProj * uView * uModel * vec4(aPos,1.0); }
)GLSL";

static const char* FRAG_UNLIT = R"GLSL(
#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main(){ FragColor = vec4(uColor, 1.0); }
)GLSL";

//================ SHADOW DEPTH pass (spot) ================================
static const char* VERT_SHADOW = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uModel, uLightVP;
void main(){ gl_Position = uLightVP * uModel * vec4(aPos,1.0); }
)GLSL";

static const char* FRAG_SHADOW = R"GLSL(
#version 330 core
void main(){ /* depth only */ }
)GLSL";

//================ Billboard particle shaderi (bez senke) ================
static const char* BILL_VS = R"GLSL(
#version 330 core
layout (location=0) in vec2 aCorner;      // -1..+1 quad
layout (location=1) in vec3 iPos;
layout (location=2) in float iLife;       // 0..1 (norm)
layout (location=3) in float iSpread;     // poluprečnik
layout (location=4) in float iGrounded;   // 0/1

uniform mat4 uVP;
uniform vec3 uRight;
uniform vec3 uUp;
uniform float uSizeBase;

out float vLife;
out vec2  vUV;

void main(){
    vLife = clamp(iLife, 0.0, 1.0);
    vUV   = aCorner * 0.5 + 0.5;

    vec3 rightCam = normalize(uRight);
    vec3 upCam    = normalize(uUp);

    vec3 rightBoard = vec3(1.0, 0.0, 0.0);
    vec3 upBoard    = vec3(0.0, 0.0, 1.0);

    vec3 right = mix(rightCam, rightBoard, iGrounded);
    vec3 up    = mix(upCam,    upBoard,    iGrounded);

    float flySize = uSizeBase * mix(1.8, 0.8, vLife);
    float flatSz  = max(iSpread, 0.02);

    float sz = mix(flySize, flatSz, iGrounded);
    vec3  wp = iPos + (right * aCorner.x + up * aCorner.y) * sz;

    gl_Position = uVP * vec4(wp, 1.0);
}
)GLSL";

static const char* BILL_FS = R"GLSL(
#version 330 core
in float vLife;
in vec2  vUV;
out vec4 FragColor;

void main(){
    vec2 p = vUV*2.0 - 1.0;
    float r2 = dot(p,p);
    if(r2>1.0) discard;

    float alpha = smoothstep(1.0, 0.65, 1.0 - r2) * vLife;
    vec3 col = mix(vec3(1.0, 0.55, 0.15), vec3(1.0, 0.9, 0.6), 1.0 - vLife);
    FragColor = vec4(col, alpha);
}
)GLSL";
// ---------- POST: shaders ----------
static const char* FS_BRIGHT = R"GLSL(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord; // set by pipeline
uniform sampler2D uScene;
uniform float uThreshold; // e.g. 1.0 (HDR)
void main(){
    vec3 col = texture(uScene, TexCoord).rgb;
    float luma = max(max(col.r,col.g), col.b);
    vec3 bright = (luma > uThreshold) ? col : vec3(0.0);
    FragColor = vec4(bright, 1.0);
}
)GLSL";

static const char* VS_FSQUAD = R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec2 aUV;
out vec2 TexCoord;
void main(){
    TexCoord = aUV;
    gl_Position = vec4(aPos,0.0,1.0);
}
)GLSL";

static const char* FS_BLUR = R"GLSL(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D uTex;
uniform vec2 uTexel; // 1/width, 1/height
uniform int uHorizontal; // 1=H, 0=V
void main(){
    vec2 off = uHorizontal==1 ? vec2(uTexel.x,0.0) : vec2(0.0,uTexel.y);
    // 5-tap gaussian-ish
    float w0=0.204164, w1=0.304005, w2=0.093913; // normalized
    vec3 c = texture(uTex, TexCoord).rgb * w0;
    c += (texture(uTex, TexCoord + off*1.384615).rgb +
          texture(uTex, TexCoord - off*1.384615).rgb) * w1;
    c += (texture(uTex, TexCoord + off*3.230769).rgb +
          texture(uTex, TexCoord - off*3.230769).rgb) * w2;
    FragColor = vec4(c,1.0);
}
)GLSL";

static const char* FS_COMBINE = R"GLSL(
#version 330 core
out vec4 FragColor;
in vec2 TexCoord;
uniform sampler2D uScene; // HDR scene
uniform sampler2D uBloom; // blurred bright
uniform float uBloomIntensity; // e.g. 0.7
uniform int   uDoTonemap; // 1 to apply tonemap
vec3 tonemapACES(vec3 x){
    // ACES approx (Narkowicz)
    const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}
void main(){
    vec3 hdr  = texture(uScene, TexCoord).rgb;
    vec3 bloom= texture(uBloom, TexCoord).rgb * uBloomIntensity;
    vec3 col  = hdr + bloom;

    if(uDoTonemap==1){
        col = tonemapACES(col);
        // gamma
        col = pow(col, vec3(1.0/2.2));
    }
    FragColor = vec4(col, 1.0);
}
)GLSL";


//================ Geometrija =====================
struct Vertex { glm::vec3 p; glm::vec3 n; glm::vec2 uv; float top=1.f; };
struct Mesh {
    GLuint vao=0,vbo=0,ebo=0; GLsizei idx=0;
    void upload(const std::vector<Vertex>& V, const std::vector<unsigned>& I){
        idx=(GLsizei)I.size();
        glGenVertexArrays(1,&vao); glGenBuffers(1,&vbo); glGenBuffers(1,&ebo);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,V.size()*sizeof(Vertex),V.data(),GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,I.size()*sizeof(unsigned),I.data(),GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,p));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,n));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,uv));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,sizeof(Vertex),(void*)offsetof(Vertex,top));
        glBindVertexArray(0);
    }
    void draw() const { glBindVertexArray(vao); glDrawElements(GL_TRIANGLES,idx,GL_UNSIGNED_INT,0); glBindVertexArray(0); }
};

static void appendXform(
    std::vector<Vertex>& DV, std::vector<unsigned>& DI,
    const std::vector<Vertex>& SV, const std::vector<unsigned>& SI,
    const glm::mat4& M)
{
    glm::mat3 N = glm::mat3(glm::transpose(glm::inverse(M)));
    unsigned base=(unsigned)DV.size();
    for(auto v:SV){ v.p=glm::vec3(M*glm::vec4(v.p,1)); v.n=glm::normalize(N*v.n); DV.push_back(v); }
    for(unsigned i:SI) DI.push_back(base+i);
}

static void makeBox(std::vector<Vertex>& V, std::vector<unsigned>& I, float sx,float sy,float sz){
    glm::vec3 h(sx*0.5f,sy*0.5f,sz*0.5f);
    glm::vec3 P[8]={
        {-h.x,-h.y,-h.z},{ h.x,-h.y,-h.z},{ h.x, h.y,-h.z},{-h.x, h.y,-h.z},
        {-h.x,-h.y, h.z},{ h.x,-h.y, h.z},{ h.x, h.y, h.z},{-h.x, h.y, h.z}
    };
    auto face=[&](int a,int b,int c,int d, glm::vec3 n, float topFlag,
                  glm::vec2 uv0, glm::vec2 uv1, glm::vec2 uv2, glm::vec2 uv3){
        unsigned s=(unsigned)V.size();
        V.push_back({P[a],n,uv0,topFlag}); V.push_back({P[b],n,uv1,topFlag});
        V.push_back({P[c],n,uv2,topFlag}); V.push_back({P[d],n,uv3,topFlag});
        I.insert(I.end(),{s,s+1,s+2, s,s+2,s+3});
    };
    V.clear(); I.clear();
    // BOTTOM
    face(0,1,5,4,{0,-1,0},0.0f, {0,0},{1,0},{1,1},{0,1});
    // FRONT
    face(4,5,6,7,{0,0,1},0.0f,  {0,0},{1,0},{1,1},{0,1});
    // BACK
    face(1,0,3,2,{0,0,-1},0.0f, {0,0},{1,0},{1,1},{0,1});
    // LEFT
    face(0,4,7,3,{-1,0,0},0.0f, {0,0},{1,0},{1,1},{0,1});
    // RIGHT
    face(5,1,2,6,{1,0,0},0.0f,  {0,0},{1,0},{1,1},{0,1});
    // TOP — UV 0..1, aTop=1
    face(3,2,6,7,{0,1,0},1.0f,  {0,0},{1,0},{1,1},{0,1});
}
static void makeCylinder(std::vector<Vertex>& V, std::vector<unsigned>& I, float r,float h,int seg){
    V.clear(); I.clear();
    for(int i=0;i<=seg;i++){
        float t=(float)i/seg, a=t*glm::two_pi<float>();
        float c=std::cos(a), s=std::sin(a);
        glm::vec3 n=glm::normalize(glm::vec3(c,0,s));
        V.push_back({{r*c,-h*0.5f,r*s},n,{t,0},0.0f});
        V.push_back({{r*c, h*0.5f,r*s},n,{t,1},0.0f});
    }
    for(int i=0;i<seg;i++){ unsigned k=i*2; I.insert(I.end(),{k,k+1,k+3, k,k+3,k+2}); }
    // top cap
    unsigned bt=(unsigned)V.size(); V.push_back({{0,h*0.5f,0},{0,1,0},{0.5f,0.5f},0.0f});
    for(int i=0;i<=seg;i++){ float t=(float)i/seg,a=t*glm::two_pi<float>();
        V.push_back({{r*std::cos(a),h*0.5f,r*std::sin(a)},{0,1,0},{(std::cos(a)+1)*0.5f,(std::sin(a)+1)*0.5f},0.0f});
    }
    for(int i=0;i<seg;i++) I.insert(I.end(),{bt,bt+1+i,bt+2+i});
    // bottom cap
    unsigned bb=(unsigned)V.size(); V.push_back({{0,-h*0.5f,0},{0,-1,0},{0.5f,0.5f},0.0f});
    for(int i=0;i<=seg;i++){ float t=(float)i/seg,a=t*glm::two_pi<float>();
        V.push_back({{r*std::cos(a),-h*0.5f,r*std::sin(a)},{0,-1,0},{(std::cos(a)+1)*0.5f,(std::sin(a)+1)*0.5f},0.0f});
    }
    for(int i=0;i<seg;i++) I.insert(I.end(),{bb,bb+2+i,bb+1+i});
}

static Mesh gBoardMesh, gRookBodyMesh, gRookCrenelMesh, gUnitCyl;

static Mesh buildRookBodyMesh(){
    std::vector<Vertex> V; std::vector<unsigned> I; std::vector<Vertex> vtmp; std::vector<unsigned> itmp;
    makeCylinder(vtmp,itmp,0.35f,1.2f,32);      appendXform(V,I,vtmp,itmp,glm::mat4(1));
    makeCylinder(vtmp,itmp,0.50f,0.12f,32);     appendXform(V,I,vtmp,itmp,glm::translate(glm::mat4(1),{0,-0.66f,0}));
    makeCylinder(vtmp,itmp,0.60f,0.10f,32);     appendXform(V,I,vtmp,itmp,glm::translate(glm::mat4(1),{0,-0.71f,0}));
    makeCylinder(vtmp,itmp,0.45f,0.10f,32);     appendXform(V,I,vtmp,itmp,glm::translate(glm::mat4(1),{0, 0.66f,0}));
    Mesh m; m.upload(V,I); return m;
}
static Mesh buildRookCrenelMesh(){
    std::vector<Vertex> V; std::vector<unsigned> I;
    makeBox(V,I,0.18f,0.12f,0.18f);
    Mesh m; m.upload(V,I); return m;
}
static Mesh buildChessBoardMesh(){
    std::vector<Vertex> V; std::vector<unsigned> I;
    makeBox(V,I,BOARD_SIZE,0.18f,BOARD_SIZE);
    Mesh m; m.upload(V,I); return m;
}
static Mesh buildUnitCylinder(){
    std::vector<Vertex> V; std::vector<unsigned> I;
    makeCylinder(V,I,0.5f,1.0f,32);
    Mesh m; m.upload(V,I); return m;
}

//================ Pieces (po gridu) =================
struct Piece {
    int cx=0, cz=0;        // 0..7
    bool alive=true;
    glm::vec3 color{1,1,1};
};

static const float ROOK_SCALE = 0.75f;
static const float TILE_Y     = 0.72f * ROOK_SCALE;

static glm::vec3 gridToWorld(int cx,int cz){
    const float tile = BOARD_SIZE / (float)BOARD_TILES;
    const float origin = -BOARD_SIZE * 0.5f;
    float x = origin + (cx + 0.5f) * tile;
    float z = origin + (cz + 0.5f) * tile;
    return { x, TILE_Y, z };
}

//================ Billboard Particles (blast + spill) ================
static const int   PCOUNT     = 2500;
static const float LIFE_MIN   = 1.2f;
static const float LIFE_MAX   = 2.6f;
static const float SPD_MIN    = 1.2f;
static const float SPD_MAX    = 2.6f;
static const float GRAVITY_Y  = -2.4f;
static const float FRICTION   = 1.8f;
static const float SPREAD_GROW= 0.45f;

struct PCPU { float px,py,pz, vx,vy,vz, life; bool grounded; float spread; };
struct PGPU { float x,y,z, lifeN, spread, grounded; };

struct BillboardParticles {
    std::vector<PCPU> ps;
    std::vector<PGPU> inst;
    GLuint vao=0, vboQuad=0, eboQuad=0, vboInst=0;
    GLuint prog=0;
    bool   alive=false;

    void init(GLuint program){
        prog = program;
        ps.resize(PCOUNT);
        inst.resize(PCOUNT);
        for(auto& p: ps){ p.life=0.0f; p.grounded=false; p.spread=0.0f; }

        static const float quad[8] = {-1,-1,  1,-1,  1,1,  -1,1};
        static const unsigned idx[6] = {0,1,2, 2,3,0};

        glGenVertexArrays(1,&vao);
        glBindVertexArray(vao);

        glGenBuffers(1,&vboQuad);
        glBindBuffer(GL_ARRAY_BUFFER, vboQuad);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,2*sizeof(float),(void*)0);

        glGenBuffers(1,&eboQuad);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboQuad);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

        glGenBuffers(1,&vboInst);
        glBindBuffer(GL_ARRAY_BUFFER, vboInst);
        glBufferData(GL_ARRAY_BUFFER, inst.size()*sizeof(PGPU), nullptr, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(1); glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,sizeof(PGPU),(void*)0);                     glVertexAttribDivisor(1,1);
        glEnableVertexAttribArray(2); glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(PGPU),(void*)(3*sizeof(float)));     glVertexAttribDivisor(2,1);
        glEnableVertexAttribArray(3); glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,sizeof(PGPU),(void*)(4*sizeof(float)));     glVertexAttribDivisor(3,1);
        glEnableVertexAttribArray(4); glVertexAttribPointer(4,1,GL_FLOAT,GL_FALSE,sizeof(PGPU),(void*)(5*sizeof(float)));     glVertexAttribDivisor(4,1);

        glBindVertexArray(0);
    }

    inline float urand(float a,float b){ return a + (b-a) * (float)rand()/RAND_MAX; }

    void blast(const glm::vec3& c){
        for(auto& p: ps){
            float ang = urand(0.0f, glm::two_pi<float>());
            float r   = urand(0.0f, 0.22f);
            float up  = urand(0.85f, 1.0f);
            float spd = urand(SPD_MIN, SPD_MAX);

            p.px = c.x; p.py = 0.02f; p.pz = c.z;
            p.vx = std::cos(ang)*r*spd;
            p.vy = up*spd;
            p.vz = std::sin(ang)*r*spd;

            p.life = urand(LIFE_MIN, LIFE_MAX);
            p.grounded=false; p.spread=0.0f;
        }
        alive = true;
    }

    void clear(){ alive=false; for(auto& p:ps){ p.life=0.0f; p.grounded=false; p.spread=0.0f; } }

    void update(float dt){
        if(!alive) return;
        const float boardTopY = 0.0f;
        bool anyAlive=false;

        for(auto& p: ps){
            if(p.life<=0.0f) continue;
            anyAlive=true;

            if(!p.grounded){
                p.life -= dt;
                p.vy   += GRAVITY_Y * dt;
                p.px   += p.vx * dt;
                p.py   += p.vy * dt;
                p.pz   += p.vz * dt;

                if(p.py < boardTopY){
                    p.py = boardTopY;
                    p.vy = 0.0f;
                    p.grounded = true;
                    p.vx *= 0.35f; p.vz *= 0.35f;
                    p.spread = 0.06f;
                }
            }else{
                p.life -= dt*1.4f;
                p.px   += p.vx * dt;
                p.pz   += p.vz * dt;
                float damp = std::exp(-FRICTION * dt);
                p.vx *= damp; p.vz *= damp;
                p.spread = std::min(p.spread + SPREAD_GROW * dt, 0.45f);
            }
        }
        alive = anyAlive;
    }

    void upload(){
        if(!alive) return;
        for(size_t i=0;i<ps.size();++i){
            auto& p=ps[i];
            float lifeN = (p.life<=0.0f)?0.0f:(p.life / LIFE_MAX);
            inst[i] = {p.px,p.py,p.pz, lifeN, p.spread, p.grounded?1.0f:0.0f};
        }
        glBindBuffer(GL_ARRAY_BUFFER, vboInst);
        glBufferSubData(GL_ARRAY_BUFFER, 0, inst.size()*sizeof(PGPU), inst.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void draw(const glm::mat4& VP, const glm::vec3& camRight, const glm::vec3& camUp){
        if(!alive) return;
        glUseProgram(prog);
        glUniformMatrix4fv(glGetUniformLocation(prog,"uVP"),1,GL_FALSE,glm::value_ptr(VP));
        glUniform3fv(glGetUniformLocation(prog,"uRight"),1,glm::value_ptr(camRight));
        glUniform3fv(glGetUniformLocation(prog,"uUp"),1,glm::value_ptr(camUp));
        glUniform1f(glGetUniformLocation(prog,"uSizeBase"), 0.28f);

        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glBindVertexArray(vao);
        glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0, (GLsizei)inst.size());
        glBindVertexArray(0);

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_TRUE);
        glUseProgram(0);
    }
} gBill;

//================= Procedural ROUGHNESS TEXTURE =================
static GLuint makeRoughnessTex(int W=256,int H=256){
    std::vector<unsigned char> img(W*H);
    auto idx=[&](int x,int y){ return y*W + x; };
    for(int y=0;y<H;++y) for(int x=0;x<W;++x)
        img[idx(x,y)] = (unsigned char)(rand()%256);

    auto blur1 = img;
    auto blur2 = img;
    auto pass=[&](const std::vector<unsigned char>& in, std::vector<unsigned char>& out){
        for(int y=0;y<H;++y) for(int x=0;x<W;++x){
            int sum=0, cnt=0;
            for(int dy=-1; dy<=1; ++dy)
            for(int dx=-1; dx<=1; ++dx){
                int xx=(x+dx+W)%W, yy=(y+dy+H)%H;
                sum += in[idx(xx,yy)]; cnt++;
            }
            out[idx(x,y)] = (unsigned char)(sum/cnt);
        }
    };
    pass(img, blur1);
    pass(blur1, blur2);

    GLuint tex=0; glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W,H, 0, GL_RED, GL_UNSIGNED_BYTE, blur2.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}
static GLuint makeAOtex(int W=256,int H=256){
    std::vector<unsigned char> img(W*H);
    auto idx=[&](int x,int y){ return y*W + x; };

    // noise
    for(int y=0;y<H;++y)
        for(int x=0;x<W;++x)
            img[idx(x,y)] = (unsigned char)(rand()%256);

    // 3 blura
    auto pass = [&](const std::vector<unsigned char>& in, std::vector<unsigned char>& out){
        for(int y=0;y<H;++y){
            for(int x=0;x<W;++x){
                int sum=0, cnt=0;
                for(int dy=-1; dy<=1; ++dy)
                    for(int dx=-1; dx<=1; ++dx){
                        int xx=(x+dx+W)%W, yy=(y+dy+H)%H;
                        sum += in[idx(xx,yy)]; cnt++;
                    }
                out[idx(x,y)] = (unsigned char)(sum/cnt);
            }
        }
    };

    std::vector<unsigned char> b1=img, b2=img, b3=img;
    pass(img, b1);
    pass(b1,  b2);
    pass(b2,  b3);

    for(int i=0;i<W*H;++i){
        float v = b3[i] / 255.0f;
        v = powf(v, 0.8f);
        b3[i] = (unsigned char)std::round(std::clamp(v, 0.0f, 1.0f) * 255.0f);
    }

    GLuint tex=0; glGenTextures(1,&tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, W,H, 0, GL_RED, GL_UNSIGNED_BYTE, b3.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
    return tex;
}


//================ Kamera / prozor =================
int gW=1280,gH=720; glm::mat4 gProj(1.f);
static void framebuffer_size(GLFWwindow*, int w, int h){
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    gW = w; gH = h;
    glViewport(0, 0, w, h);
    float aspect = (float)w / (float)h;
    if (!std::isfinite(aspect) || aspect <= 0.0f) aspect = 1.0f;
    gProj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
    if (gGLReady) {
        createPostTargets(w, h);
    }
}

//================== UI helperi (3x5 font + draw list) ======================
struct UIVertex { float x,y; float r,g,b,a; };

static GLuint progUI=0, uiVAO=0, uiVBO=0;
static std::vector<UIVertex> uiVerts;

static void uiInit(GLuint program){
    progUI = program;
    glGenVertexArrays(1,&uiVAO);
    glGenBuffers(1,&uiVBO);
    glBindVertexArray(uiVAO);
    glBindBuffer(GL_ARRAY_BUFFER, uiVBO);
    glBufferData(GL_ARRAY_BUFFER, 1024 * 1024, nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0); glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(UIVertex),(void*)0);
    glEnableVertexAttribArray(1); glVertexAttribPointer(1,4,GL_FLOAT,GL_FALSE,sizeof(UIVertex),(void*)(2*sizeof(float)));
    glBindVertexArray(0);
}
static void uiBegin(){ uiVerts.clear(); }
static void uiFlush(int screenW,int screenH){
    if(uiVerts.empty()) return;
    GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
    glDisable(GL_DEPTH_TEST);

    glUseProgram(progUI);
    glUniform2f(glGetUniformLocation(progUI,"uScreen"), (float)screenW, (float)screenH);
    glBindVertexArray(uiVAO);
    glBindBuffer(GL_ARRAY_BUFFER, uiVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, uiVerts.size()*sizeof(UIVertex), uiVerts.data());
    glDrawArrays(GL_TRIANGLES, 0, (GLsizei)uiVerts.size());
    glBindVertexArray(0);
    glUseProgram(0);

    if(depthWas) glEnable(GL_DEPTH_TEST);
}
static void uiRect(float x,float y,float w,float h, glm::vec4 col){
    UIVertex v[6] = {
        {x,y, col.r,col.g,col.b,col.a}, {x+w,y, col.r,col.g,col.b,col.a}, {x+w,y+h, col.r,col.g,col.b,col.a},
        {x,y, col.r,col.g,col.b,col.a}, {x+w,y+h, col.r,col.g,col.b,col.a}, {x,y+h, col.r,col.g,col.b,col.a}
    };
    uiVerts.insert(uiVerts.end(), v, v+6);
}
static uint16_t glyph3x5(char c){
    switch(std::toupper((unsigned char)c)){
        case 'A': return 0b010111101011111;
        case 'B': return 0b110110111101111;
        case 'C': return 0b011100100100011;
        case 'D': return 0b110101101101110;
        case 'E': return 0b111100110100111;
        case 'F': return 0b111100110100100;
        case 'H': return 0b101101111101101;
        case 'I': return 0b111010010010111;
        case 'K': return 0b101101110101101;
        case 'L': return 0b100100100100111;
        case 'N': return 0b101111111101101;
        case 'O': return 0b010101101101010;
        case 'R': return 0b110101110101101;
        case 'S': return 0b011100010001110;
        case 'T': return 0b111010010010010;
        case 'W': return 0b101101111111101;
        case 'X': return 0b101101010101101;
        case 'Y': return 0b101101010010010;
        case 'Z': return 0b111001010100111;
        case 'V': return 0b101101101101010;
        case 'U': return 0b101101101101111;
        case 'M': return 0b111111101101101;
        case 'P': return 0b110101110100100;
        case ' ': return 0;
        case ':': return 0b000010000010000;
        case '0': return 0b111101101101111;
        case '1': return 0b010110010010111;
        case '2': return 0b111001111100111;
        case '3': return 0b111001111001111;
        case '4': return 0b101101111001001;
        case '5': return 0b111100111001111;
        case '6': return 0b111100111101111;
        case '7': return 0b111001001001001;
        case '8': return 0b111101111101111;
        case '9': return 0b111101111001111;
        default:  return 0;
    }
}
static void uiText3x5(float x,float y, float scale, glm::vec4 col, const std::string& s){
    const float px = 3.0f * scale;
    const float py = 5.0f * scale;
    float penx = x;
    for(char cc : s){
        if(cc=='\n'){ y += (py+scale); penx = x; continue; }
        uint16_t bits = glyph3x5(cc);
        for(int r=0;r<5;++r){
            for(int c=0;c<3;++c){
                int bitIndex = (4 - r)*3 + (2 - c);
                if(bits & (1<<bitIndex)){
                    uiRect(penx + c*scale, y + r*scale, scale, scale, col);
                }
            }
        }
        penx += (px + scale);
    }
}

//================== Main =========================
enum class GameState { MENU, PLAYING, RESULT };

int main(){
    if(!glfwInit()){ std::cerr<<"Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(gW,gH,"Rook Duel + Spotlight Shadows (no-audio)",nullptr,nullptr);
    if(!win){ std::cerr<<"Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, framebuffer_size);
    glfwSwapInterval(1);
    int fbw, fbh; glfwGetFramebufferSize(win,&fbh,&fbw); // intentionally swap? nope—fix:
    glfwGetFramebufferSize(win,&fbw,&fbh); framebuffer_size(win,fbw,fbh);

    glewExperimental=GL_TRUE;
    if(glewInit()!=GLEW_OK){ std::cerr<<"Failed to init GLEW\n"; return -1; }
    gGLReady = true;

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Programs
    GLuint progMesh   = link_src(VERT_MESH,   FRAG_MESH);
    GLuint progChess  = link_src(VERT_CHESS,  FRAG_CHESS);
    GLuint progBill   = link_src(BILL_VS,     BILL_FS);
    GLuint progUIsh   = link_src(VERT_UI,     FRAG_UI);
    GLuint progUnlit  = link_src(VERT_UNLIT,  FRAG_UNLIT);
    GLuint progShadow = link_src(VERT_SHADOW, FRAG_SHADOW);
    if(!progMesh||!progChess||!progBill||!progUIsh||!progUnlit||!progShadow){ std::cerr<<"FATAL: shader program failed\n"; return -1; }
    // POST: programs
    progBright  = link_src(VS_FSQUAD, FS_BRIGHT);
    progBlur    = link_src(VS_FSQUAD, FS_BLUR);
    progCombine = link_src(VS_FSQUAD, FS_COMBINE);
    if(!progBright || !progBlur || !progCombine){
        std::cerr<<"FATAL: post shaders failed\n";
        return -1;
    }

    makeFullscreenQuad();
    createPostTargets(gW, gH);

    // Meshes
    gBoardMesh      = buildChessBoardMesh();
    gRookBodyMesh   = buildRookBodyMesh();
    gRookCrenelMesh = buildRookCrenelMesh();
    gUnitCyl        = buildUnitCylinder();
    gBill.init(progBill);
    uiInit(progUIsh);
    GLuint gRoughTex = makeRoughnessTex();
    GLuint gAOtex    = makeAOtex();

    // Camera
    glm::vec3 eye    = glm::vec3(0.0f, 2.6f, 8.0f);
    glm::vec3 center = glm::vec3(0.0f, 0.8f, 0.0f);
    glm::vec3 up     = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::mat4 view   = glm::lookAt(eye, center, up);

    // Spotlight params
    glm::vec3 lightPos = glm::vec3(0.0f, LIGHT_HEIGHT-0.45f, 0.0f); // centar
    glm::vec3 lightDir = glm::normalize(glm::vec3(0.0f,-1.0f,0.0f));
    float degInner = 22.0f, degOuter = 28.0f;
    float cosInner = std::cos(glm::radians(degInner));
    float cosOuter = std::cos(glm::radians(degOuter));
    glm::mat4 lightProj = glm::perspective(glm::radians(2.0f*degOuter), 1.0f, 0.1f, 25.0f);
    glm::mat4 lightView = glm::lookAt(lightPos, lightPos + lightDir, glm::vec3(0,0,-1));
    glm::mat4 lightVP   = lightProj * lightView;

    // Shadow FBO
    GLuint shadowFBO=0, shadowTex=0;
    glGenFramebuffers(1,&shadowFBO);
    glGenTextures(1,&shadowTex);
    glBindTexture(GL_TEXTURE_2D, shadowTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, SHADOW_RES, SHADOW_RES, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    float border[4] = {1,1,1,1};
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border);
    glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadowTex, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    if(glCheckFramebufferStatus(GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE){
        std::cerr<<"Shadow FBO incomplete!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Pieces init
    Piece white; white.cx=0; white.cz=7; white.color={0.92f,0.92f,0.95f};
    Piece black; black.cx=7; black.cz=0; black.color={0.08f,0.08f,0.10f};

    // Input edges
    bool pUp=false,pDown=false,pLeft=false,pRight=false,pR=false;
    bool pW=false,pA=false,pS=false,pD=false;
    bool pB=false;
    bool mousePrev=false;

    // Game state
    enum class GameState { MENU, PLAYING, RESULT };
    GameState state = GameState::MENU;
    float timeLeft  = 30.0f;
    float resultHold= 2.5f;
    std::string winnerText;

    // Uniform locs
    GLint uM_uModel = glGetUniformLocation(progMesh,"uModel");
    GLint uM_uView  = glGetUniformLocation(progMesh,"uView");
    GLint uM_uProj  = glGetUniformLocation(progMesh,"uProj");
    GLint uM_uAlb   = glGetUniformLocation(progMesh,"uAlbedo");
    GLint uM_uCam   = glGetUniformLocation(progMesh,"uCamPos");
    GLint uM_uTex   = glGetUniformLocation(progMesh,"uRoughTex");
    GLint uM_uTile  = glGetUniformLocation(progMesh,"uTexTiling");
    GLint uM_uLPos  = glGetUniformLocation(progMesh,"uLightPos");
    GLint uM_uLDir  = glGetUniformLocation(progMesh,"uLightDir");
    GLint uM_uCosI  = glGetUniformLocation(progMesh,"uCosInner");
    GLint uM_uCosO  = glGetUniformLocation(progMesh,"uCosOuter");
    GLint uM_uLVP   = glGetUniformLocation(progMesh,"uLightVP");
    GLint uM_uShTex = glGetUniformLocation(progMesh,"uShadowMap");
    GLint uM_uShPx  = glGetUniformLocation(progMesh,"uShadowTexel");
    GLint uM_uAmb   = glGetUniformLocation(progMesh,"uAmbient");
    GLint uM_uAO    = glGetUniformLocation(progMesh,"uAO");

    GLint uC_uModel = glGetUniformLocation(progChess,"uModel");
    GLint uC_uVP    = glGetUniformLocation(progChess,"uVP");
    GLint uC_uTiles = glGetUniformLocation(progChess,"uTiles");
    GLint uC_cA     = glGetUniformLocation(progChess,"colorA");
    GLint uC_cB     = glGetUniformLocation(progChess,"colorB");
    GLint uC_side   = glGetUniformLocation(progChess,"uSideColor");
    GLint uC_lPos   = glGetUniformLocation(progChess,"uLightPos");
    GLint uC_lDir   = glGetUniformLocation(progChess,"uLightDir");
    GLint uC_cI     = glGetUniformLocation(progChess,"uCosInner");
    GLint uC_cO     = glGetUniformLocation(progChess,"uCosOuter");
    GLint uC_lVP    = glGetUniformLocation(progChess,"uLightVP");
    GLint uC_sh     = glGetUniformLocation(progChess,"uShadowMap");
    GLint uC_shPx   = glGetUniformLocation(progChess,"uShadowTexel");
    GLint uC_amb    = glGetUniformLocation(progChess,"uAmbient");

    GLint uS_uModel = glGetUniformLocation(progShadow,"uModel");
    GLint uS_uLVP   = glGetUniformLocation(progShadow,"uLightVP");

    double lastT=glfwGetTime();
    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();
        if(glfwGetKey(win,GLFW_KEY_ESCAPE)==GLFW_PRESS){ glfwSetWindowShouldClose(win,1); }

        // dt
        double now=glfwGetTime(); float dt=(float)std::min(0.033, now-lastT); lastT=now;
        float t = (float)now;

        // (A) Shadow pass (depth only)
        glViewport(0,0,SHADOW_RES,SHADOW_RES);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFBO);
        glClearDepth(1.0);
        glClear(GL_DEPTH_BUFFER_BIT);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_FRONT); // smanji acne
        glUseProgram(progShadow);
        glUniformMatrix4fv(uS_uLVP,1,GL_FALSE,glm::value_ptr(lightVP));

        auto drawBoardDepth = [&](){
            glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(0.0f, -0.09f, 0.0f));
            glUniformMatrix4fv(uS_uModel,1,GL_FALSE,glm::value_ptr(model));
            gBoardMesh.draw();
        };
        auto drawRookDepth=[&](const Piece& P){
            if(!P.alive) return;
            glm::mat4 Troot = glm::translate(glm::mat4(1), gridToWorld(P.cx,P.cz))
                            * glm::scale(glm::mat4(1), glm::vec3(ROOK_SCALE));
            glUniformMatrix4fv(uS_uModel,1,GL_FALSE,glm::value_ptr(Troot));
            gRookBodyMesh.draw();
            float ringR = 0.42f, yTop=0.72f, ang0=t*0.6f, bob=std::sin(t*1.7f)*0.02f;
            for(int i=0;i<4;i++){
                float a = ang0 + i * glm::half_pi<float>();
                glm::vec3 pos = { ringR*std::cos(a), yTop + bob, ringR*std::sin(a) };
                glm::mat4 Mchild = Troot
                    * glm::translate(glm::mat4(1), pos)
                    * glm::rotate(glm::mat4(1), a + t*1.2f, glm::vec3(0,1,0));
                glUniformMatrix4fv(uS_uModel,1,GL_FALSE,glm::value_ptr(Mchild));
                gRookCrenelMesh.draw();
            }
        };
        drawBoardDepth();
        drawRookDepth(white);
        drawRookDepth(black);
        glUseProgram(0);
        glCullFace(GL_BACK);
        glDisable(GL_CULL_FACE);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // (B) Main scene into HDR FBO
        glViewport(0,0,gW,gH);
        glBindFramebuffer(GL_FRAMEBUFFER, sceneFBO);
        glClearColor(0.06f,0.07f,0.10f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

        // 0) LAMPA (unlit) — vizuelna reprezentacija
        {
            glUseProgram(progUnlit);
            GLint uUL_Model = glGetUniformLocation(progUnlit,"uModel");
            GLint uUL_View  = glGetUniformLocation(progUnlit,"uView");
            GLint uUL_Proj  = glGetUniformLocation(progUnlit,"uProj");
            GLint uUL_Color = glGetUniformLocation(progUnlit,"uColor");
            glUniformMatrix4fv(uUL_View,1,GL_FALSE,glm::value_ptr(view));
            glUniformMatrix4fv(uUL_Proj,1,GL_FALSE,glm::value_ptr(gProj));

            const float ceilingTop = LIGHT_HEIGHT + 1.2f;
            const float cableTop   = ceilingTop;
            const float cableBot   = LIGHT_HEIGHT;
            const float cableLen   = cableTop - cableBot;
            const float cableCY    = 0.5f*(cableTop + cableBot);

            // kabel
            glm::mat4 M = glm::translate(glm::mat4(1), glm::vec3(0.0f, cableCY, 0.0f))
                        * glm::scale(glm::mat4(1), glm::vec3(0.06f, cableLen, 0.06f));
            glUniformMatrix4fv(uUL_Model,1,GL_FALSE,glm::value_ptr(M));
            glUniform3f(uUL_Color, 0.18f,0.18f,0.18f);
            gUnitCyl.draw();

            // grlo
            M = glm::translate(glm::mat4(1), glm::vec3(0.0f, LIGHT_HEIGHT-0.10f, 0.0f))
              * glm::scale(glm::mat4(1), glm::vec3(0.22f, 0.20f, 0.22f));
            glUniformMatrix4fv(uUL_Model,1,GL_FALSE,glm::value_ptr(M));
            glUniform3f(uUL_Color, 0.22f,0.22f,0.22f);
            gUnitCyl.draw();

            // abažur
            M = glm::translate(glm::mat4(1), glm::vec3(0.0f, LIGHT_HEIGHT-0.32f, 0.0f))
              * glm::scale(glm::mat4(1), glm::vec3(0.55f, 0.25f, 0.55f));
            glUniformMatrix4fv(uUL_Model,1,GL_FALSE,glm::value_ptr(M));
            glUniform3f(uUL_Color, 0.10f,0.10f,0.10f);
            gUnitCyl.draw();

            // sijalica (emissive)
            M = glm::translate(glm::mat4(1), glm::vec3(0.0f, LIGHT_HEIGHT-0.45f, 0.0f))
              * glm::scale(glm::mat4(1), glm::vec3(0.18f, 0.22f, 0.18f));
            glUniformMatrix4fv(uUL_Model,1,GL_FALSE,glm::value_ptr(M));
            glUniform3f(uUL_Color, 1.0f, 0.96f, 0.80f);
            gUnitCyl.draw();

            glUseProgram(0);
        }

        // 1) Tabla (sa senkama)
        {
            glUseProgram(progChess);
            glm::mat4 model = glm::translate(glm::mat4(1), glm::vec3(0.0f, -0.09f, 0.0f)); // vrh y≈0
            glm::mat4 VP    = gProj * view;
            glUniformMatrix4fv(uC_uModel,1,GL_FALSE,glm::value_ptr(model));
            glUniformMatrix4fv(uC_uVP,   1,GL_FALSE,glm::value_ptr(VP));
            glUniform1i(uC_uTiles, BOARD_TILES);
            glUniform3f(uC_cA, 0.92f,0.92f,0.92f);
            glUniform3f(uC_cB, 0.08f,0.08f,0.08f);
            glUniform3f(uC_side, 0.18f,0.12f,0.08f);
            glUniform3fv(uC_lPos,1,glm::value_ptr(lightPos));
            glUniform3fv(uC_lDir,1,glm::value_ptr(lightDir));
            glUniform1f(uC_cI, cosInner);
            glUniform1f(uC_cO, cosOuter);
            glUniformMatrix4fv(uC_lVP,1,GL_FALSE,glm::value_ptr(lightVP));
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowTex);
            glUniform1i(uC_sh, 1);
            glUniform2f(uC_shPx, 1.0f/SHADOW_RES, 1.0f/SHADOW_RES);
            glUniform1f(uC_amb, 0.35f);
            gBoardMesh.draw();
            glBindTexture(GL_TEXTURE_2D,0);
            glUseProgram(0);
        }

        // 2) Figure (sa senkama)
        {
            glUseProgram(progMesh);
            glUniformMatrix4fv(uM_uView,1,GL_FALSE,glm::value_ptr(view));
            glUniformMatrix4fv(uM_uProj,1,GL_FALSE,glm::value_ptr(gProj));
            glUniform3fv(uM_uCam,1,glm::value_ptr(eye));
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, gRoughTex);
            glUniform1i(uM_uTex, 0);
            glUniform1f(uM_uTile, 3.0f);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, gAOtex);
            glUniform1i(uM_uAO, 2);
            glUniform3fv(uM_uLPos,1,glm::value_ptr(lightPos));
            glUniform3fv(uM_uLDir,1,glm::value_ptr(lightDir));
            glUniform1f(uM_uCosI, cosInner);
            glUniform1f(uM_uCosO, cosOuter);
            glUniformMatrix4fv(uM_uLVP,1,GL_FALSE,glm::value_ptr(lightVP));
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, shadowTex);
            glUniform1i(uM_uShTex, 1);
            glUniform2f(uM_uShPx, 1.0f/SHADOW_RES, 1.0f/SHADOW_RES);
            glUniform1f(uM_uAmb, 0.28f);

            auto drawRookAnimated=[&](const Piece& P){
                if(!P.alive) return;
                glm::mat4 Troot = glm::translate(glm::mat4(1), gridToWorld(P.cx,P.cz))
                                * glm::scale(glm::mat4(1), glm::vec3(ROOK_SCALE));
                glUniform3fv(uM_uAlb,1,glm::value_ptr(P.color));
                glUniformMatrix4fv(uM_uModel,1,GL_FALSE,glm::value_ptr(Troot));
                gRookBodyMesh.draw();

                float ringR = 0.42f;
                float yTop  = 0.72f;
                float ang0  = t * 0.6f;
                float bob   = std::sin(t*1.7f)*0.02f;
                for(int i=0;i<4;i++){
                    float a = ang0 + i * glm::half_pi<float>();
                    glm::vec3 pos = { ringR*std::cos(a), yTop + bob, ringR*std::sin(a) };
                    glm::mat4 Mchild = Troot
                        * glm::translate(glm::mat4(1), pos)
                        * glm::rotate(glm::mat4(1), a + t*1.2f, glm::vec3(0,1,0));
                    glUniformMatrix4fv(uM_uModel,1,GL_FALSE,glm::value_ptr(Mchild));
                    gRookCrenelMesh.draw();
                }
            };
            drawRookAnimated(white);
            drawRookAnimated(black);

            glBindTexture(GL_TEXTURE_2D, 0);
            glUseProgram(0);
        }

        // 3) Particles (bez senke, additive)
        {
            glm::mat4 VP = gProj * view;
            glm::vec3 fwd = glm::normalize(center - eye);
            glm::vec3 right = glm::normalize(glm::cross(fwd, up));
            glm::vec3 cup   = glm::normalize(glm::cross(right, fwd));
            gBill.draw(VP, right, cup);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, ppFBO[0]); // 1) Bright pass
        glUseProgram(progBright);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneColor);
        glUniform1i(glGetUniformLocation(progBright,"uScene"), 0);
        glUniform1f(glGetUniformLocation(progBright,"uThreshold"), 0.3f); // tweak if needed (0.9–1.3)
        glBindVertexArray(fsVAO);
        glDisable(GL_DEPTH_TEST);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // 2) Blur ping-pong (npr. 6–8 prolaza)
        bool horizontal = true;
        int blurPasses = 8;
        for(int i=0;i<blurPasses;i++){
            glBindFramebuffer(GL_FRAMEBUFFER, ppFBO[horizontal?1:0]);
            glUseProgram(progBlur);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, ppTex[horizontal?0:1]); // čitamo iz prethodnog
            glUniform1i(glGetUniformLocation(progBlur,"uTex"), 0);
            glUniform2f(glGetUniformLocation(progBlur,"uTexel"), 1.0f/gW, 1.0f/gH);
            glUniform1i(glGetUniformLocation(progBlur,"uHorizontal"), horizontal?1:0);
            glBindVertexArray(fsVAO);
            glDrawArrays(GL_TRIANGLES,0,6);
            horizontal = !horizontal;
        }

        // 3) Combine (scene HDR + blurred bright) -> default framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0,0,gW,gH);
        glUseProgram(progCombine);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sceneColor);
        glUniform1i(glGetUniformLocation(progCombine,"uScene"), 0);
        glActiveTexture(GL_TEXTURE1);
        // ako je broj prolaza paran, poslednji blur je u ppTex[0], inače u ppTex[1]
        GLuint finalBloomTex = ppTex[horizontal?0:1];
        glBindTexture(GL_TEXTURE_2D, finalBloomTex);
        glUniform1i(glGetUniformLocation(progCombine,"uBloom"), 1);
        glUniform1f(glGetUniformLocation(progCombine,"uBloomIntensity"),
            gBloomOn ? 0.7f : 0.0f);
        glUniform1i(glGetUniformLocation(progCombine,"uDoTonemap"), 1);
        glBindVertexArray(fsVAO);
        glDrawArrays(GL_TRIANGLES,0,6);

        // restore state (ako treba)
        glEnable(GL_DEPTH_TEST);
        glUseProgram(0);

        // -------- GAME LOGIC + UI --------
        double mx, my; glfwGetCursorPos(win,&mx,&my);
        bool mouseNow = (glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
        bool mouseClick = (mouseNow && !mousePrev);
        mousePrev = mouseNow;

        auto pointInRect=[&](float x,float y,float w,float h,double px,double py){
            return (px>=x && px<=x+w && py>=y && py<=y+h);
        };

        if(state == GameState::MENU){
            uiBegin();
            uiRect(0,0,(float)gW,(float)gH, {0,0,0,0.35f});
            float bw=220, bh=60;
            float bx = gW*0.5f - bw*0.5f;
            float by1= gH*0.55f, by2 = by1 + 80;

            glm::vec4 btnColStart = {0.1f,0.6f,0.2f,0.9f};
            glm::vec4 btnColExit  = {0.6f,0.1f,0.1f,0.9f};
            if(pointInRect(bx,by1,bw,bh,mx,my)) btnColStart = {0.2f,0.8f,0.3f,1.0f};
            if(pointInRect(bx,by2,bw,bh,mx,my)) btnColExit  = {0.8f,0.2f,0.2f,1.0f};

            uiRect(bx,by1,bw,bh, btnColStart);
            uiRect(bx,by2,bw,bh, btnColExit);

            uiText3x5(gW*0.5f - 180, gH*0.22f, 6.0f, {1,1,1,1}, "ROOK DUEL");
            uiText3x5(gW*0.5f - 280, gH*0.32f, 3.5f, {0.9f,0.9f,0.9f,1},
                      "WHITE: ARROWS\nBLACK: WASD\nCAPTURE BLACK BEFORE TIME HITS 0");
            char bloomLine[64];
            std::snprintf(bloomLine, sizeof(bloomLine),
                          "TURN BLOOM ON :  B");
            uiText3x5(gW*0.5f - 280, gH*0.42f, 3.5f, {0.9f,0.9f,0.9f,1}, bloomLine);
            uiText3x5(bx+60, by1+22, 3.8f, {0,0,0,1}, "START");
            uiText3x5(bx+72, by2+22, 3.8f, {0,0,0,1}, "EXIT");
            uiFlush(gW,gH);

            if(mouseClick){
                if(pointInRect(bx,by1,bw,bh,mx,my)){
                    state = GameState::PLAYING;
                    timeLeft = 30.0f;
                    white.cx=0; white.cz=7; white.alive=true;
                    black.cx=7; black.cz=0; black.alive=true;
                    gBill.clear();
             #ifdef _WIN32
                    play_wav(SFX_CLICK);
             #endif
                }else if(pointInRect(bx,by2,bw,bh,mx,my)){
             #ifdef _WIN32
                    play_wav(SFX_CLICK);
             #endif
                    glfwSetWindowShouldClose(win,1);
                }
            }

        }
        else if(state == GameState::PLAYING){
            auto moveIfEdge=[&](bool cur,bool& prev, int& cx,int& cz, int dcx, int dcz)->bool{
                if(cur && !prev){
                    int ncx = std::clamp(cx+dcx, 0, BOARD_TILES-1);
                    int ncz = std::clamp(cz+dcz, 0, BOARD_TILES-1);
                    if(ncx!=cx || ncz!=cz){
                        cx=ncx; cz=ncz; prev = cur;
                 #ifdef _WIN32
                        play_wav(SFX_MOVE);
                 #endif
                        return true;
                    }
                }
                prev = cur; return false;
            };
            bool upK    = glfwGetKey(win,GLFW_KEY_UP)==GLFW_PRESS;
            bool downK  = glfwGetKey(win,GLFW_KEY_DOWN)==GLFW_PRESS;
            bool leftK  = glfwGetKey(win,GLFW_KEY_LEFT)==GLFW_PRESS;
            bool rightK = glfwGetKey(win,GLFW_KEY_RIGHT)==GLFW_PRESS;
            bool rK     = glfwGetKey(win,GLFW_KEY_R)==GLFW_PRESS;
            bool wK     = glfwGetKey(win,GLFW_KEY_W)==GLFW_PRESS;
            bool aK     = glfwGetKey(win,GLFW_KEY_A)==GLFW_PRESS;
            bool sK     = glfwGetKey(win,GLFW_KEY_S)==GLFW_PRESS;
            bool dK     = glfwGetKey(win,GLFW_KEY_D)==GLFW_PRESS;
            bool bK = glfwGetKey(win, GLFW_KEY_B) == GLFW_PRESS;
            if(bK && !pB){
                gBloomOn = !gBloomOn;
            }
            pB = bK;

            moveIfEdge(upK,   pUp,    white.cx, white.cz, 0,-1);
            moveIfEdge(downK, pDown,  white.cx, white.cz, 0,+1);
            moveIfEdge(leftK, pLeft,  white.cx, white.cz,-1, 0);
            moveIfEdge(rightK,pRight, white.cx, white.cz,+1, 0);
            moveIfEdge(wK,    pW,     black.cx, black.cz, 0,-1);
            moveIfEdge(sK,    pS,     black.cx, black.cz, 0,+1);
            moveIfEdge(aK,    pA,     black.cx, black.cz,-1, 0);
            moveIfEdge(dK,    pD,     black.cx, black.cz,+1, 0);

            if(rK && !pR){
                white.cx=0; white.cz=7; white.alive=true;
                black.cx=7; black.cz=0; black.alive=true;
                gBill.clear();
                timeLeft = 30.0f;
            }
            pR=rK;

            if(white.alive && black.alive && white.cx==black.cx && white.cz==black.cz){
                black.alive=false;
                gBill.blast( gridToWorld(black.cx,black.cz) );
                winnerText = "WHITE WINS";
                state = GameState::RESULT;
                resultHold = 2.5f;
            #ifdef _WIN32
                play_wav(SFX_EXPLOSION);
            #endif
            }

            if(white.alive && black.alive){
                timeLeft -= dt;
                if(timeLeft <= 0.0f){
                    timeLeft = 0.0f;
                    winnerText = "BLACK WINS";
                    state = GameState::RESULT;
                    resultHold = 2.5f;
                 #ifdef _WIN32
                    play_wav(SFX_END);
                 #endif
                }
            }

            uiBegin();
            char buf[32];
            int tsec = (int)std::ceil(timeLeft);
            std::snprintf(buf,sizeof(buf),"TIME: %02d", std::max(0,tsec));
            uiText3x5(gW - 190, 20, 4.5f, {1,1,1,1}, buf);
            uiFlush(gW,gH);
        }
        else if(state == GameState::RESULT){
            resultHold -= dt;
            uiBegin();
            uiRect(0,0,(float)gW,(float)gH, {0,0,0,0.35f});
            uiText3x5(gW*0.5f - 120, gH*0.45f, 6.0f, {1,1,0.4f,1}, winnerText);
            uiFlush(gW,gH);
            if(resultHold <= 0.0f){
                state = GameState::MENU;
            }
        }

        // Particles update
        gBill.update(std::min(dt, 0.05f));
        gBill.upload();

        glfwSwapBuffers(win);
    }

    glfwTerminate();
    return 0;
}

// Main.cpp
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>

// ================== Particles config (uvećano) ==================
static const int   kParticleCount = 2500;
static const float kSpawnRate     = 900.0f;
static const float kLifetimeMin   = 1.2f;
static const float kLifetimeMax   = 2.6f;
static const float kSpeedMin      = 1.2f;
static const float kSpeedMax      = 2.6f;
static const float kGravity       = -2.4f;
// ================================================================

struct Particle { float px, py, pz; float vx, vy, vz; float life; };
struct GPUVertex { float x, y, z; float life; };

static int gWinW = 1000, gWinH = 700;

static bool  g_emitOn        = true;
static bool  g_showBoard     = true;
static bool  g_showParticles = true;

static glm::vec3 gCamPos    = glm::vec3(0.0f, 2.3f, 8.0f);
static glm::vec3 gCamTarget = glm::vec3(0.0f, 0.6f, 0.0f);
static glm::vec3 gCamUp     = glm::vec3(0.0f, 1.0f, 0.0f);
static float     gFovDeg    = 60.0f;

// ================== helperi za compile/link iz STRINGA (za particles) ==================
static GLuint compile_src(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[4096]; glGetShaderInfoLog(s, 4096, nullptr, log);
        std::cerr << (type==GL_VERTEX_SHADER? "Vertex" : "Fragment") << " shader error:\n" << log << "\n";
    }
    return s;
}
static GLuint link_src(const char* vs, const char* fs) {
    GLuint v = compile_src(GL_VERTEX_SHADER, vs);
    GLuint f = compile_src(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram(); glAttachShader(p, v); glAttachShader(p, f); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok){ char log[4096]; glGetProgramInfoLog(p,4096,nullptr,log); std::cerr<<"Link error:\n"<<log<<"\n"; }
    glDeleteShader(v); glDeleteShader(f); return p;
}

// ================== helperi za compile/link iz FAJLA (za chess.vert/frag) ==================
static std::string slurp_file(const char* path){
    std::ifstream in(path, std::ios::binary);
    if(!in){ std::cerr << "[shader] cannot open: " << path << "\n"; return {}; }
    std::string s; in.seekg(0,std::ios::end);
    s.resize((size_t)in.tellg()); in.seekg(0);
    in.read(&s[0], s.size());
    return s;
}
static GLuint compile_from_file(GLenum type, const char* path){
    std::string src = slurp_file(path);
    if(src.empty()) return 0;
    const char* c = src.c_str();
    GLuint sh = glCreateShader(type);
    glShaderSource(sh,1,&c,nullptr);
    glCompileShader(sh);
    GLint ok=0; glGetShaderiv(sh,GL_COMPILE_STATUS,&ok);
    if(!ok){ char log[4096]; glGetShaderInfoLog(sh,4096,nullptr,log);
        std::cerr << "[shader] compile error (" << path << "):\n" << log << "\n";
    }
    return sh;
}
static GLuint link_from_files(const char* vpath, const char* fpath){
    GLuint vs = compile_from_file(GL_VERTEX_SHADER,   vpath);
    GLuint fs = compile_from_file(GL_FRAGMENT_SHADER, fpath);
    GLuint p  = glCreateProgram();
    glAttachShader(p,vs); glAttachShader(p,fs); glLinkProgram(p);
    GLint ok=0; glGetProgramiv(p,GL_LINK_STATUS,&ok);
    if(!ok){ char log[4096]; glGetProgramInfoLog(p,4096,nullptr,log);
        std::cerr << "[shader] link error:\n" << log << "\n";
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

static void fbSizeCB(GLFWwindow*, int w, int h){
    gWinW = (w>0? w:1); gWinH = (h>0? h:1);
    glViewport(0,0,gWinW,gWinH);
}

// ================== PARTICLE SHADERS (inline) ==================
static const char* kParticleVS = R"(#version 330 core
layout (location=0) in vec3 aPos;
layout (location=1) in float aLife;

uniform mat4 uVP;

out float vLife;

void main(){
    vLife = clamp(aLife, 0.0, 1.0);
    vec4 clip = uVP * vec4(aPos, 1.0);
    gl_Position = clip;

    // 🔥 Velike čestice, ali i dalje se smanjuju sa daljinom
    float base = 240.0;                        // bilo ~42.0
    float size = base / max(clip.w, 0.1);      // perspektivno skaliranje
    gl_PointSize = clamp(size, 28.0, 256.0);   // veći min/max
}
)";
static const char* kParticleFS = R"(#version 330 core
in float vLife;
out vec4 FragColor;
void main(){
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r = dot(uv, uv);
    float alpha = smoothstep(1.0, 0.55, r);
    vec3  color = vec3(1.0, 0.88, 0.55);
    FragColor = vec4(color, alpha * vLife);
}
)";

// ---------- input edge trigger ----------
struct KeyState { int prev=GLFW_RELEASE; };
static bool pressedOnce(GLFWwindow* w, int key, KeyState& ks){
    int cur = glfwGetKey(w, key);
    bool fired = (ks.prev == GLFW_RELEASE && cur == GLFW_PRESS);
    ks.prev = cur; return fired;
}

int main(){
    if (!glfwInit()) { std::cerr<<"Failed to init GLFW\n"; return -1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* win = glfwCreateWindow(gWinW,gWinH,"3D Board (thick) + Bigger Particles",nullptr,nullptr);
    if(!win){ std::cerr<<"Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1);
    glfwSetFramebufferSizeCallback(win, fbSizeCB);

    glewExperimental = GL_TRUE;
    if (glewInit()!=GLEW_OK){ std::cerr<<"Failed to init GLEW\n"; return -1; }
    glEnable(GL_PROGRAM_POINT_SIZE);


    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // === kreiraj programe ===
    // 1) chess.vert/chess.frag iz fajlova (preko SHADER_DIR definisanog u CMakeLists)
    std::string vsPath = std::string(SHADER_DIR) + "/chess.vert";
    std::string fsPath = std::string(SHADER_DIR) + "/chess.frag";
    std::cout << "Loading shaders from:\n  " << vsPath << "\n  " << fsPath << std::endl;

    GLuint progBoard = link_from_files(vsPath.c_str(), fsPath.c_str()); // << koristi fajlove
    if (!progBoard) { std::cerr << "FATAL: progBoard == 0 (shader load/link fail)\n"; return -1; }

    // 2) particles iz inline stringova
    GLuint progPart  = link_src(kParticleVS, kParticleFS);

    // ---------- GEOMETRIJA TABLE (KUTIJA) ----------
    const float hs = 0.5f;   // half side (pre skaliranja)
    const float ht = 0.05f;  // half thickness (pre skaliranja)

    std::vector<float> boardVerts; // pos(3), normal(3), uv(2), isTop(1) => 9 floats
    boardVerts.reserve(24 * 9);

    auto pushFace = [&](glm::vec3 n, float topFlag,
                        glm::vec3 p0, glm::vec2 uv0,
                        glm::vec3 p1, glm::vec2 uv1,
                        glm::vec3 p2, glm::vec2 uv2,
                        glm::vec3 p3, glm::vec2 uv3)
    {
        glm::vec3 ps[4] = {p0,p1,p2,p3};
        glm::vec2 uvs[4]= {uv0,uv1,uv2,uv3};
        for (int i=0;i<4;++i){
            boardVerts.push_back(ps[i].x); boardVerts.push_back(ps[i].y); boardVerts.push_back(ps[i].z);
            boardVerts.push_back(n.x);     boardVerts.push_back(n.y);     boardVerts.push_back(n.z);
            boardVerts.push_back(uvs[i].x);boardVerts.push_back(uvs[i].y);
            boardVerts.push_back(topFlag);
        }
    };

    // TOP (y = +ht)
    pushFace(glm::vec3(0,1,0), 1.0f,
             glm::vec3(-hs,+ht,-hs), glm::vec2(0,0),
             glm::vec3(+hs,+ht,-hs), glm::vec2(1,0),
             glm::vec3(+hs,+ht,+hs), glm::vec2(1,1),
             glm::vec3(-hs,+ht,+hs), glm::vec2(0,1));
    // BOTTOM (y = -ht)
    pushFace(glm::vec3(0,-1,0), 0.0f,
             glm::vec3(-hs,-ht,+hs), glm::vec2(0,0),
             glm::vec3(+hs,-ht,+hs), glm::vec2(1,0),
             glm::vec3(+hs,-ht,-hs), glm::vec2(1,1),
             glm::vec3(-hs,-ht,-hs), glm::vec2(0,1));
    // RIGHT (x = +hs)
    pushFace(glm::vec3(1,0,0), 0.0f,
             glm::vec3(+hs,-ht,-hs), glm::vec2(0,0),
             glm::vec3(+hs,-ht,+hs), glm::vec2(1,0),
             glm::vec3(+hs,+ht,+hs), glm::vec2(1,1),
             glm::vec3(+hs,+ht,-hs), glm::vec2(0,1));
    // LEFT (x = -hs)
    pushFace(glm::vec3(-1,0,0), 0.0f,
             glm::vec3(-hs,-ht,+hs), glm::vec2(0,0),
             glm::vec3(-hs,-ht,-hs), glm::vec2(1,0),
             glm::vec3(-hs,+ht,-hs), glm::vec2(1,1),
             glm::vec3(-hs,+ht,+hs), glm::vec2(0,1));
    // FRONT (z = +hs)
    pushFace(glm::vec3(0,0,1), 0.0f,
             glm::vec3(-hs,-ht,+hs), glm::vec2(0,0),
             glm::vec3(+hs,-ht,+hs), glm::vec2(1,0),
             glm::vec3(+hs,+ht,+hs), glm::vec2(1,1),
             glm::vec3(-hs,+ht,+hs), glm::vec2(0,1));
    // BACK (z = -hs)
    pushFace(glm::vec3(0,0,-1), 0.0f,
             glm::vec3(+hs,-ht,-hs), glm::vec2(0,0),
             glm::vec3(-hs,-ht,-hs), glm::vec2(1,0),
             glm::vec3(-hs,+ht,-hs), glm::vec2(1,1),
             glm::vec3(+hs,+ht,-hs), glm::vec2(0,1));

    // Indeksi (6 po licu)
    std::vector<unsigned int> boardIdx; boardIdx.reserve(36);
    for (int f=0; f<6; ++f){
        unsigned base = f*4;
        unsigned int idx[6] = { base+0, base+1, base+2, base+2, base+3, base+0 };
        boardIdx.insert(boardIdx.end(), idx, idx+6);
    }

    GLuint boardVAO, boardVBO, boardEBO;
    glGenVertexArrays(1,&boardVAO);
    glGenBuffers(1,&boardVBO);
    glGenBuffers(1,&boardEBO);
    glBindVertexArray(boardVAO);
    glBindBuffer(GL_ARRAY_BUFFER, boardVBO);
    glBufferData(GL_ARRAY_BUFFER, boardVerts.size()*sizeof(float), boardVerts.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boardEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, boardIdx.size()*sizeof(unsigned int), boardIdx.data(), GL_STATIC_DRAW);
    GLsizei stride = 9 * sizeof(float);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,stride,(void*)0);                     glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE,stride,(void*)(3*sizeof(float)));     glEnableVertexAttribArray(1);
    glVertexAttribPointer(2,2,GL_FLOAT,GL_FALSE,stride,(void*)(6*sizeof(float)));     glEnableVertexAttribArray(2);
    glVertexAttribPointer(3,1,GL_FLOAT,GL_FALSE,stride,(void*)(8*sizeof(float)));     glEnableVertexAttribArray(3);

    // ---------- PARTICLES ----------
    std::vector<Particle> ps(kParticleCount);
    std::vector<GPUVertex> gpu(kParticleCount);
    for (auto& p: ps) p.life = 0.0f;

    GLuint pVAO, pVBO;
    glGenVertexArrays(1,&pVAO);
    glGenBuffers(1,&pVBO);
    glBindVertexArray(pVAO);
    glBindBuffer(GL_ARRAY_BUFFER, pVBO);
    glBufferData(GL_ARRAY_BUFFER, gpu.size()*sizeof(GPUVertex), nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(GPUVertex),(void*)0);          glEnableVertexAttribArray(0);
    glVertexAttribPointer(1,1,GL_FLOAT,GL_FALSE,sizeof(GPUVertex),(void*)(3*sizeof(float))); glEnableVertexAttribArray(1);

    std::mt19937 rng(12345);
    auto urand = [&](float a, float b){ std::uniform_real_distribution<float> d(a,b); return d(rng); };

    // skala table (da je lako menjati debljinu i veličinu)
    const float BOARD_XZ_SCALE = 5.0f;
    const float BOARD_Y_SCALE  = 2.0f;  // debljina (menjaj ovde)
    auto respawn = [&](Particle& p){
        float angle = urand(0.0f, 6.2831853f);
        float r     = urand(0.0f, 0.22f);
        float up    = urand(0.85f, 1.0f);
        float spd   = urand(kSpeedMin, kSpeedMax);
        float boardTopY = 0.05f * BOARD_Y_SCALE; // ht * scaleY
        p.px=0.0f; p.py=boardTopY + 0.02f; p.pz=0.0f;
        p.vx = std::cos(angle)*r*spd;
        p.vy = up*spd;
        p.vz = std::sin(angle)*r*spd;
        p.life = urand(kLifetimeMin, kLifetimeMax);
    };

    auto now = std::chrono::high_resolution_clock::now();
    double spawnAccum = 0.0;

    KeyState k1,k2,k3,kP,kO,kR,kEsc;

    while(!glfwWindowShouldClose(win)){
        auto newNow = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(newNow - now).count();
        now = newNow;

        auto on = [&](int key, KeyState& ks){ return pressedOnce(win,key,ks); };
        if (on(GLFW_KEY_ESCAPE,kEsc)) glfwSetWindowShouldClose(win,1);
        if (on(GLFW_KEY_1,k1)) { g_showBoard=true;  g_showParticles=false; }
        if (on(GLFW_KEY_2,k2)) { g_showBoard=false; g_showParticles=true;  }
        if (on(GLFW_KEY_3,k3)) { g_showBoard=true;  g_showParticles=true;  }
        if (on(GLFW_KEY_P,kP)) { g_emitOn=true;  }
        if (on(GLFW_KEY_O,kO)) { g_emitOn=false; }
        if (on(GLFW_KEY_R,kR)) { for(auto& p: ps) p.life=0.0f; }

        float aspect = (float)gWinW / (float)gWinH;
        glm::mat4 proj = glm::perspective(glm::radians(gFovDeg), aspect, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(gCamPos, gCamTarget, gCamUp);
        glm::mat4 vp   = proj * view;

        if (g_emitOn){
            spawnAccum += kSpawnRate * dt;
            int toSpawn = (int)spawnAccum;
            spawnAccum -= toSpawn;
            for (auto& p : ps){
                if (toSpawn <= 0) break;
                if (p.life <= 0.0f){ respawn(p); --toSpawn; }
            }
        }
        float boardTopY = 0.05f * BOARD_Y_SCALE; // zbog odbijanja
        for (auto& p : ps){
            if (p.life <= 0.0f) continue;
            p.life -= dt;
            p.vy  += kGravity * dt;
            p.px  += p.vx * dt;
            p.py  += p.vy * dt;
            p.pz  += p.vz * dt;
            if (p.py < boardTopY) { p.py = boardTopY; p.vy *= -0.35f; }
        }

        for (size_t i=0;i<ps.size();++i){
            const auto& p = ps[i];
            float lifeNorm = p.life <= 0.0f ? 0.0f : (p.life / kLifetimeMax);
            gpu[i] = { p.px, p.py, p.pz, lifeNorm };
        }
        glBindBuffer(GL_ARRAY_BUFFER, pVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, gpu.size()*sizeof(GPUVertex), gpu.data());

        glClearColor(0.06f, 0.07f, 0.10f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 1) ŠAHOVSKA TABLA (shader iz FAJLA)
        if (g_showBoard) {
            glUseProgram(progBoard);

            glm::mat4 model(1.0f);
            model = glm::scale(model, glm::vec3(BOARD_XZ_SCALE, BOARD_Y_SCALE, BOARD_XZ_SCALE));
            glUniformMatrix4fv(glGetUniformLocation(progBoard, "uModel"), 1, GL_FALSE, &model[0][0]);
            glUniformMatrix4fv(glGetUniformLocation(progBoard, "uVP"),    1, GL_FALSE, &vp[0][0]);

            // podesive uniforme u chess.frag
            glUniform1i(glGetUniformLocation(progBoard, "uTiles"), 8);
            glUniform3f(glGetUniformLocation(progBoard, "colorA"), 0.96f,0.96f,0.96f);
            glUniform3f(glGetUniformLocation(progBoard, "colorB"), 0.08f,0.08f,0.08f);
            glUniform3f(glGetUniformLocation(progBoard, "uSideColor"), 0.22f, 0.15f, 0.09f);
            glUniform3f(glGetUniformLocation(progBoard, "uLightDir"),  -0.4f, 1.0f, 0.3f);
            glUniform1f(glGetUniformLocation(progBoard, "uAmbient"),   0.50f);

            glBindVertexArray(boardVAO);
            glDepthMask(GL_TRUE);
            glDrawElements(GL_TRIANGLES, (GLsizei)boardIdx.size(), GL_UNSIGNED_INT, 0);
        }

        // 2) Čestice
        if (g_showParticles){
            glUseProgram(progPart);
            glUniformMatrix4fv(glGetUniformLocation(progPart,"uVP"),1,GL_FALSE,&vp[0][0]);
            glBindVertexArray(pVAO);
            glDepthMask(GL_FALSE);
            glDrawArrays(GL_POINTS, 0, (GLsizei)gpu.size());
            glDepthMask(GL_TRUE);
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glfwDestroyWindow(win);
    glfwTerminate();
    return 0;
}

#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <cuda_gl_interop.h>

#include "BarnesHut.cuh"
#include "BodiesSoA.hpp"
#include "BruteForce.cuh"
#include "Common.cuh"
#include "EglContext.hpp"
#include "GLCompat.hpp"
#include "Interop.cuh"
#include "Ppm.hpp"
#include "Scenarios.hpp"

#ifndef GL_PROGRAM_POINT_SIZE
#define GL_PROGRAM_POINT_SIZE 0x8642
#endif

namespace {

// --- tiny column-major 4x4 matrix helpers (just enough for a static camera) ---

void perspective(float* m, float fovyRad, float aspect, float zNear, float zFar) {
    float f = 1.0f / std::tan(fovyRad * 0.5f);
    for (int i = 0; i < 16; ++i) m[i] = 0.0f;
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (zFar + zNear) / (zNear - zFar);
    m[11] = -1.0f;
    m[14] = (2.0f * zFar * zNear) / (zNear - zFar);
}

void cross(const float* a, const float* b, float* out) {
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
}

void normalize(float* v) {
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (len > 1e-8f) { v[0] /= len; v[1] /= len; v[2] /= len; }
}

void lookAt(float* m, const float* eye, const float* center, const float* upHint) {
    float forward[3] = {center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]};
    normalize(forward);
    float side[3]; cross(forward, upHint, side); normalize(side);
    float up[3]; cross(side, forward, up);

    m[0] = side[0]; m[4] = side[1]; m[8] = side[2];
    m[1] = up[0]; m[5] = up[1]; m[9] = up[2];
    m[2] = -forward[0]; m[6] = -forward[1]; m[10] = -forward[2];
    m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
    m[12] = -(side[0] * eye[0] + side[1] * eye[1] + side[2] * eye[2]);
    m[13] = -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]);
    m[14] = forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2];
}

void multiply4x4(float* out, const float* a, const float* b) {
    float tmp[16];
    for (int col = 0; col < 4; ++col)
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) sum += a[k * 4 + row] * b[col * 4 + k];
            tmp[col * 4 + row] = sum;
        }
    for (int i = 0; i < 16; ++i) out[i] = tmp[i];
}

// --- shader helpers ---

const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform float uPointSize;
void main() {
    gl_Position = uViewProj * vec4(aPos, 1.0);
    gl_PointSize = uPointSize;
}
)";

const char* kFragmentSrc = R"(#version 330 core
out vec4 FragColor;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    FragColor = vec4(1.0, 0.85, 0.5, 0.6);
}
)";

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = gl.CreateShader(type);
    gl.ShaderSource(s, 1, &src, nullptr);
    gl.CompileShader(s);
    GLint ok = 0;
    gl.GetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        gl.GetShaderInfoLog(s, sizeof(log), nullptr, log);
        std::fprintf(stderr, "shader compile failed:\n%s\n", log);
        return 0;
    }
    return s;
}

GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint p = gl.CreateProgram();
    gl.AttachShader(p, vs);
    gl.AttachShader(p, fs);
    gl.LinkProgram(p);
    GLint ok = 0;
    gl.GetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        gl.GetProgramInfoLog(p, sizeof(log), nullptr, log);
        std::fprintf(stderr, "program link failed:\n%s\n", log);
        return 0;
    }
    return p;
}

struct Args {
    int n = 50000;
    int frames = 300;
    float dt = 0.01f;
    float theta = 0.5f;
    std::string mode = "barnes_hut";  // bruteforce | barnes_hut
    std::string scenario = "random";
    unsigned seed = 42;
    int width = 1280, height = 720;
    int dumpEvery = 30;
    std::string dumpDir = "frames";
    float pointSize = 3.0f;
    float viewDistance = 180.0f;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto val = [&](const char* flag) -> const char* {
            size_t len = std::strlen(flag);
            return arg.rfind(flag, 0) == 0 ? arg.c_str() + len : nullptr;
        };
        if (auto v = val("--n=")) a.n = std::atoi(v);
        else if (auto v = val("--frames=")) a.frames = std::atoi(v);
        else if (auto v = val("--dt=")) a.dt = std::atof(v);
        else if (auto v = val("--theta=")) a.theta = std::atof(v);
        else if (auto v = val("--mode=")) a.mode = v;
        else if (auto v = val("--scenario=")) a.scenario = v;
        else if (auto v = val("--seed=")) a.seed = static_cast<unsigned>(std::atoi(v));
        else if (auto v = val("--width=")) a.width = std::atoi(v);
        else if (auto v = val("--height=")) a.height = std::atoi(v);
        else if (auto v = val("--dump-every=")) a.dumpEvery = std::atoi(v);
        else if (auto v = val("--dump-dir=")) a.dumpDir = v;
        else if (auto v = val("--point-size=")) a.pointSize = std::atof(v);
        else if (auto v = val("--view-distance=")) a.viewDistance = std::atof(v);
    }
    return a;
}

struct RenderState {
    cudaGraphicsResource_t cudaVbo = nullptr;
    GLuint fbo = 0, colorTex = 0, vao = 0, vbo = 0, program = 0;
    GLint locViewProj = -1, locPointSize = -1;
    float viewProj[16];
    float pointSize = 3.0f;
    int width = 0, height = 0;
    int dumpEvery = 0;
    std::string dumpDir;
    std::vector<uint8_t> readbackBuf;  // sized width*height*4, reused every dump
};

void renderFrame(const BodiesSoA& d_bodies, int step, void* userPtr) {
    auto* s = reinterpret_cast<RenderState*>(userPtr);

    size_t numBytes = 0;
    float* devPtr = nullptr;
    CUDA_CHECK(cudaGraphicsMapResources(1, &s->cudaVbo, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devPtr), &numBytes, s->cudaVbo));
    copyPositionsToVBO(d_bodies, devPtr);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &s->cudaVbo, 0));

    gl.BindFramebuffer(GL_FRAMEBUFFER, s->fbo);
    glViewport(0, 0, s->width, s->height);
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    gl.UseProgram(s->program);
    gl.UniformMatrix4fv(s->locViewProj, 1, GL_FALSE, s->viewProj);
    gl.Uniform1f(s->locPointSize, s->pointSize);
    gl.BindVertexArray(s->vao);
    glDrawArrays(GL_POINTS, 0, d_bodies.n);

    if (s->dumpEvery > 0 && step % s->dumpEvery == 0) {
        if (s->readbackBuf.empty()) s->readbackBuf.resize(static_cast<size_t>(s->width) * s->height * 4);
        glReadPixels(0, 0, s->width, s->height, GL_RGBA, GL_UNSIGNED_BYTE, s->readbackBuf.data());
        char path[512];
        std::snprintf(path, sizeof(path), "%s/frame_%06d.ppm", s->dumpDir.c_str(), step);
        writePPM(path, s->width, s->height, s->readbackBuf.data());
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);
    mkdir(args.dumpDir.c_str(), 0755);  // ignore EEXIST

    BodiesSoA h_bodies = BodiesSoA::allocHost(args.n);
    if (args.scenario == "orbit") {
        if (args.n != 3) { std::fprintf(stderr, "orbit scenario requires n=3\n"); return 1; }
        initOrbitScenario(h_bodies);
    } else {
        initRandomScenario(h_bodies, args.seed);
    }

    BodiesSoA d_bodies = BodiesSoA::allocDevice(args.n);
    d_bodies.copyToDevice(h_bodies);

    EglContext egl;
    if (!egl.init(args.width, args.height)) {
        std::fprintf(stderr, "Failed to create a headless EGL/OpenGL context\n");
        return 1;
    }
    if (!gl.loadAll()) {
        std::fprintf(stderr, "Failed to load required OpenGL entry points\n");
        return 1;
    }

    RenderState state;
    state.width = args.width; state.height = args.height;
    state.dumpEvery = args.dumpEvery; state.dumpDir = args.dumpDir;
    state.pointSize = args.pointSize;

    glGenTextures(1, &state.colorTex);
    glBindTexture(GL_TEXTURE_2D, state.colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, args.width, args.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    gl.GenFramebuffers(1, &state.fbo);
    gl.BindFramebuffer(GL_FRAMEBUFFER, state.fbo);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, state.colorTex, 0);
    if (gl.CheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "Framebuffer incomplete\n");
        return 1;
    }

    GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentSrc);
    if (!vs || !fs) return 1;
    state.program = linkProgram(vs, fs);
    if (!state.program) return 1;
    gl.DeleteShader(vs); gl.DeleteShader(fs);
    state.locViewProj = gl.GetUniformLocation(state.program, "uViewProj");
    state.locPointSize = gl.GetUniformLocation(state.program, "uPointSize");

    gl.GenVertexArrays(1, &state.vao);
    gl.BindVertexArray(state.vao);
    gl.GenBuffers(1, &state.vbo);
    gl.BindBuffer(GL_ARRAY_BUFFER, state.vbo);
    gl.BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(args.n) * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    gl.EnableVertexAttribArray(0);
    gl.VertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&state.cudaVbo, state.vbo, cudaGraphicsRegisterFlagsWriteDiscard));

    float eye[3] = {0.0f, args.viewDistance * 0.4f, args.viewDistance};
    float center[3] = {0.0f, 0.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float view[16], proj[16];
    lookAt(view, eye, center, up);
    perspective(proj, 45.0f * 3.14159265f / 180.0f, static_cast<float>(args.width) / args.height, 1.0f,
                args.viewDistance * 10.0f);
    multiply4x4(state.viewProj, proj, view);

    std::fprintf(stderr, "Running n=%d mode=%s frames=%d -> dumping every %d frame(s) to %s/\n", args.n,
                 args.mode.c_str(), args.frames, args.dumpEvery, args.dumpDir.c_str());

    auto t0 = std::chrono::high_resolution_clock::now();
    if (args.mode == "bruteforce") {
        runBruteForceSimulation(d_bodies, args.frames, args.dt, renderFrame, &state);
    } else {
        runBarnesHutSimulation(d_bodies, args.frames, args.dt, args.theta, renderFrame, &state);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    glFinish();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("[render %s] n=%d frames=%d total=%.1f ms  avg=%.3f ms/frame  (%.1f fps)\n", args.mode.c_str(),
                args.n, args.frames, ms, ms / args.frames, 1000.0 * args.frames / ms);

    cudaGraphicsUnregisterResource(state.cudaVbo);
    egl.destroy();
    d_bodies.freeDevice();
    h_bodies.freeHost();
    return 0;
}

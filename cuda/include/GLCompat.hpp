#pragma once

// Minimal manually-loaded subset of OpenGL core functions needed for the
// renderer. We deliberately don't link GLEW: GLEW's proc-address loader
// defaults to assuming a GLX context, and the renderer uses a windowless
// EGL context (no X server on a Colab box), so glewInit() would be loading
// the wrong way. Everything beyond GL 1.1 (VBOs, shaders, FBOs) has to be
// fetched by hand via eglGetProcAddress instead -- that's what loadGL()
// below does. GL 1.0/1.1 calls (glViewport, glClear, glDrawArrays,
// glReadPixels, ...) are still linked normally via -lGL.
#include <GL/gl.h>
#include <GL/glext.h>

struct GLFunctions {
    PFNGLGENBUFFERSPROC GenBuffers = nullptr;
    PFNGLBINDBUFFERPROC BindBuffer = nullptr;
    PFNGLBUFFERDATAPROC BufferData = nullptr;
    PFNGLGENVERTEXARRAYSPROC GenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAYPROC BindVertexArray = nullptr;
    PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer = nullptr;
    PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray = nullptr;
    PFNGLCREATESHADERPROC CreateShader = nullptr;
    PFNGLSHADERSOURCEPROC ShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC CompileShader = nullptr;
    PFNGLGETSHADERIVPROC GetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = nullptr;
    PFNGLCREATEPROGRAMPROC CreateProgram = nullptr;
    PFNGLATTACHSHADERPROC AttachShader = nullptr;
    PFNGLLINKPROGRAMPROC LinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC GetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = nullptr;
    PFNGLUSEPROGRAMPROC UseProgram = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = nullptr;
    PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv = nullptr;
    PFNGLUNIFORM1FPROC Uniform1f = nullptr;
    PFNGLGENFRAMEBUFFERSPROC GenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC BindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus = nullptr;
    PFNGLDELETESHADERPROC DeleteShader = nullptr;
    PFNGLDELETEPROGRAMPROC DeleteProgram = nullptr;

    // Loads every pointer above via eglGetProcAddress. Returns false (and
    // leaves a null pointer) if any single one failed to resolve, since a
    // silently-null function pointer would otherwise segfault far from
    // the actual cause.
    bool loadAll();
};

extern GLFunctions gl;

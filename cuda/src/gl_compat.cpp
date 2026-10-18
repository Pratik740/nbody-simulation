#include "GLCompat.hpp"

#include <EGL/egl.h>
#include <cstdio>

GLFunctions gl;

namespace {
template <typename T>
bool load(T& fn, const char* name) {
    fn = reinterpret_cast<T>(eglGetProcAddress(name));
    if (!fn) std::fprintf(stderr, "eglGetProcAddress failed for %s\n", name);
    return fn != nullptr;
}
}  // namespace

bool GLFunctions::loadAll() {
    bool ok = true;
#define LOAD(field, name) ok &= load(field, name)
    LOAD(GenBuffers, "glGenBuffers");
    LOAD(BindBuffer, "glBindBuffer");
    LOAD(BufferData, "glBufferData");
    LOAD(GenVertexArrays, "glGenVertexArrays");
    LOAD(BindVertexArray, "glBindVertexArray");
    LOAD(VertexAttribPointer, "glVertexAttribPointer");
    LOAD(EnableVertexAttribArray, "glEnableVertexAttribArray");
    LOAD(CreateShader, "glCreateShader");
    LOAD(ShaderSource, "glShaderSource");
    LOAD(CompileShader, "glCompileShader");
    LOAD(GetShaderiv, "glGetShaderiv");
    LOAD(GetShaderInfoLog, "glGetShaderInfoLog");
    LOAD(CreateProgram, "glCreateProgram");
    LOAD(AttachShader, "glAttachShader");
    LOAD(LinkProgram, "glLinkProgram");
    LOAD(GetProgramiv, "glGetProgramiv");
    LOAD(GetProgramInfoLog, "glGetProgramInfoLog");
    LOAD(UseProgram, "glUseProgram");
    LOAD(GetUniformLocation, "glGetUniformLocation");
    LOAD(UniformMatrix4fv, "glUniformMatrix4fv");
    LOAD(Uniform1f, "glUniform1f");
    LOAD(GenFramebuffers, "glGenFramebuffers");
    LOAD(BindFramebuffer, "glBindFramebuffer");
    LOAD(FramebufferTexture2D, "glFramebufferTexture2D");
    LOAD(CheckFramebufferStatus, "glCheckFramebufferStatus");
    LOAD(DeleteShader, "glDeleteShader");
    LOAD(DeleteProgram, "glDeleteProgram");
#undef LOAD
    return ok;
}

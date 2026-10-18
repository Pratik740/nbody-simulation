#pragma once

#include <EGL/egl.h>

// A windowless (pbuffer) EGL context bound to an NVIDIA device, so OpenGL
// calls work on a headless box with no X server -- which is what a Colab
// VM is. `width`/`height` size the pbuffer surface, which doubles as the
// default framebuffer's dimensions before we attach our own FBO.
struct EglContext {
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;

    // Returns false (with a message on stderr) instead of aborting, since
    // the first run of this on an unfamiliar Colab image is exactly where
    // a driver/EGL mismatch is most likely to show up.
    bool init(int width, int height);
    void destroy();
};

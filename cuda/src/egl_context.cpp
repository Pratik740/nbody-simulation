#include "EglContext.hpp"

#include <EGL/eglext.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace {
typedef EGLBoolean (*PFNEGLQUERYDEVICESEXTPROC_)(EGLint, EGLDeviceEXT*, EGLint*);
typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC_)(EGLenum, void*, const EGLint*);
typedef const char* (*PFNEGLQUERYDEVICESTRINGEXTPROC_)(EGLDeviceEXT, EGLint);
}  // namespace

bool EglContext::init(int width, int height) {
    // Headless NVIDIA boxes (Colab included) have no X server, so there's
    // no "default" X display to hand to eglGetDisplay. EGL_EXT_platform_device
    // lets us enumerate GPU devices directly and open a display on one of
    // them instead -- this is the standard recipe for EGL-without-X.
    auto queryDevices =
        reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC_>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto getPlatformDisplay =
        reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC_>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
    auto queryDeviceString =
        reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC_>(eglGetProcAddress("eglQueryDeviceStringEXT"));

    if (!queryDevices || !getPlatformDisplay) {
        std::fprintf(stderr,
                     "EGL_EXT_platform_device not available; falling back to eglGetDisplay(EGL_DEFAULT_DISPLAY)\n");
        display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    } else {
        EGLint numDevices = 0;
        queryDevices(0, nullptr, &numDevices);
        if (numDevices <= 0) {
            std::fprintf(stderr, "eglQueryDevicesEXT reported 0 EGL devices\n");
            return false;
        }
        std::vector<EGLDeviceEXT> devices(numDevices);
        queryDevices(numDevices, devices.data(), &numDevices);

        // Boxes with both the NVIDIA driver and Mesa installed can expose
        // more than one EGL device (a real NVIDIA one plus Mesa's software
        // llvmpipe one), and devices[0] isn't guaranteed to be the NVIDIA
        // one -- picking the wrong one silently gives a CPU-rendered
        // context that CUDA-OpenGL interop can't register a buffer against.
        // EGL_NV_device_cuda is only advertised by the real CUDA-capable
        // NVIDIA device, so prefer whichever device has it.
        EGLDeviceEXT chosen = devices[0];
        if (queryDeviceString) {
            for (EGLint i = 0; i < numDevices; ++i) {
                const char* exts = queryDeviceString(devices[i], EGL_EXTENSIONS);
                if (exts && std::strstr(exts, "EGL_NV_device_cuda")) {
                    chosen = devices[i];
                    break;
                }
            }
        }
        display = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, chosen, nullptr);
    }

    if (display == EGL_NO_DISPLAY) {
        std::fprintf(stderr, "Failed to obtain an EGL display\n");
        return false;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(display, &major, &minor)) {
        std::fprintf(stderr, "eglInitialize failed (error 0x%x)\n", eglGetError());
        return false;
    }
    std::fprintf(stderr, "EGL %d.%d initialized\n", major, minor);

    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE,
    };
    EGLConfig config;
    EGLint numConfigs = 0;
    if (!eglChooseConfig(display, configAttribs, &config, 1, &numConfigs) || numConfigs == 0) {
        std::fprintf(stderr, "eglChooseConfig failed (error 0x%x)\n", eglGetError());
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::fprintf(stderr, "eglBindAPI(EGL_OPENGL_API) failed\n");
        return false;
    }

    const EGLint pbufferAttribs[] = {EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE};
    surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
    if (surface == EGL_NO_SURFACE) {
        std::fprintf(stderr, "eglCreatePbufferSurface failed (error 0x%x)\n", eglGetError());
        return false;
    }

    const EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_NONE,
    };
    context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        std::fprintf(stderr, "eglCreateContext failed (error 0x%x)\n", eglGetError());
        return false;
    }

    if (!eglMakeCurrent(display, surface, surface, context)) {
        std::fprintf(stderr, "eglMakeCurrent failed (error 0x%x)\n", eglGetError());
        return false;
    }

    return true;
}

void EglContext::destroy() {
    if (display != EGL_NO_DISPLAY) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context != EGL_NO_CONTEXT) eglDestroyContext(display, context);
        if (surface != EGL_NO_SURFACE) eglDestroySurface(display, surface);
        eglTerminate(display);
    }
    display = EGL_NO_DISPLAY;
    context = EGL_NO_CONTEXT;
    surface = EGL_NO_SURFACE;
}

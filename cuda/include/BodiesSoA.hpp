#pragma once

#include <cuda_runtime.h>

#include "Common.cuh"

// Structure-of-Arrays body set: each field is its own contiguous array so a
// warp reading body[i] across threads 0..31 issues one coalesced transaction
// per field, instead of striding through an AoS struct's padding/other fields.
struct BodiesSoA {
    int n = 0;
    float* px = nullptr;
    float* py = nullptr;
    float* pz = nullptr;
    float* vx = nullptr;
    float* vy = nullptr;
    float* vz = nullptr;
    float* mass = nullptr;

    static BodiesSoA allocDevice(int n) {
        BodiesSoA b;
        b.n = n;
        size_t bytes = static_cast<size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMalloc(&b.px, bytes));
        CUDA_CHECK(cudaMalloc(&b.py, bytes));
        CUDA_CHECK(cudaMalloc(&b.pz, bytes));
        CUDA_CHECK(cudaMalloc(&b.vx, bytes));
        CUDA_CHECK(cudaMalloc(&b.vy, bytes));
        CUDA_CHECK(cudaMalloc(&b.vz, bytes));
        CUDA_CHECK(cudaMalloc(&b.mass, bytes));
        return b;
    }

    static BodiesSoA allocHost(int n) {
        BodiesSoA b;
        b.n = n;
        b.px = new float[n];
        b.py = new float[n];
        b.pz = new float[n];
        b.vx = new float[n];
        b.vy = new float[n];
        b.vz = new float[n];
        b.mass = new float[n];
        return b;
    }

    void freeDevice() {
        cudaFree(px); cudaFree(py); cudaFree(pz);
        cudaFree(vx); cudaFree(vy); cudaFree(vz);
        cudaFree(mass);
    }

    void freeHost() {
        delete[] px; delete[] py; delete[] pz;
        delete[] vx; delete[] vy; delete[] vz;
        delete[] mass;
    }

    void copyToDevice(const BodiesSoA& host) {
        size_t bytes = static_cast<size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(px, host.px, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(py, host.py, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(pz, host.pz, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vx, host.vx, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vy, host.vy, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vz, host.vz, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(mass, host.mass, bytes, cudaMemcpyHostToDevice));
    }

    void copyToHost(const BodiesSoA& device) {
        size_t bytes = static_cast<size_t>(n) * sizeof(float);
        CUDA_CHECK(cudaMemcpy(px, device.px, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(py, device.py, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(pz, device.pz, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vx, device.vx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vy, device.vy, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(vz, device.vz, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(mass, device.mass, bytes, cudaMemcpyDeviceToHost));
    }
};

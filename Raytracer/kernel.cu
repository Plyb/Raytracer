#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "PpmWriter.h"
#include "raytracer.cuh"
#include "Sphere.cuh"

#define checkCudaErrors(val) checkCuda( (val), #val, __FILE__, __LINE__ )
void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

void raytrace(byte* imgBuffer, int imgHeight, int imgWidth, int reflectionDepth);

int main()
{
    // Parameters
    const int imgHeight = 256;
    const int imgWidth = 256;
    const int reflectionDepth = 5;
    const bool p6 = false;

    byte* imgBuffer = new byte[imgHeight * imgWidth * 3];

    raytrace(imgBuffer, imgHeight, imgWidth, reflectionDepth);

    PpmWriter writer("./out.ppm");
    writer.write(imgBuffer, imgHeight, imgWidth, p6);
    free(imgBuffer);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void raytrace(byte* imgBuffer, int imgHeight, int imgWidth, int reflectionDepth)
{
    byte* devImgBuffer = 0;
    size_t imgBufferSize = imgHeight * imgWidth * 3 * sizeof(byte);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));
    const Scene** devScene;

    // Allocate GPU buffer for the image buffer
    checkCudaErrors(cudaMalloc((void**)&devImgBuffer, imgBufferSize));

    checkCudaErrors(cudaMalloc((void**)&devScene, sizeof(Scene)));
    createScene<<<1, 1 >>>(devScene);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Launch a kernel on the GPU with one thread for each pixel.
    const int MAX_THREADS_PER_BLOCK = 1024;
    int numBlocks = (imgHeight * imgWidth / MAX_THREADS_PER_BLOCK) + 1;
    tracePixel<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(devImgBuffer, imgHeight, imgWidth, reflectionDepth, devScene);

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCudaErrors(cudaMemcpy(imgBuffer, devImgBuffer, imgBufferSize, cudaMemcpyDeviceToHost));
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "PpmWriter.h"
#include "raytracer.cuh"

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

void raytrace(byte* imgBuffer, int imgHeight, int imgWidth, Scene* scene);

int main()
{
    // Parameters
    const int imgHeight = 2048;
    const int imgWidth = 2048;
    /*Sphere whiteSphere(Vec3(0.45f, -0.0f, -0.15f), 0.15f, 0.8f, 0.1f, 0.3f,
        Color(1.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 4.0f);
    Sphere redSphere(Vec3(0.0f, 0.0f, -0.1f), 0.2f, 0.6f, 0.3f, 0.1f,
        Color(1.0f, 0.0f, 0.0), Color(1.0f, 1.0f, 1.0f), 32.0f);
    Sphere greenSphere(Vec3(-0.6f, 0.0f, 0.0f), 0.3f, 0.7f, 0.2f, 0.1f,
        Color(0.0f, 1.0f, 0.0f), Color(0.5f, 1.0f, 0.5f), 64.0f);
    Sphere blueSphere(Vec3(0.0f, -10000.5, 0.0f), 10000.0f, 0.9f, 0.0f, 0.1f,
        Color(0.0f, 0.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 16.0);

    Sphere purpleSphere(Vec3(0.0f, 0.0f, 0.0f), 0.4f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 0.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 16.0f);*/

    Sphere bottom(Vec3(0.0f, -0.5f, 0.0f), 0.5f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 64.0f);
    Sphere middle(Vec3(0.0f, -0.0f, 0.0f), 0.4f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 64.0f);
    Sphere top(Vec3(0.0f, 0.4f, 0.0f), 0.3f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 64.0f);
    Sphere eyeLeft(Vec3(-0.1f, 0.4f, 0.25f), 0.05f, 0.5f, 0.4f, 0.1f,
        Color(0.05f, 0.05f, 0.05f), Color(1.0f, 1.0f, 1.0f), 128.0f);
    Sphere eyeRight(Vec3(0.1f, 0.4f, 0.25f), 0.05f, 0.5f, 0.4f, 0.1f,
        Color(0.05f, 0.05f, 0.05f), Color(1.0f, 1.0f, 1.0f), 128.0f);
    Sphere buttonUpper(Vec3(0.0f, 0.1f, 0.4f), 0.03f, 0.5f, 0.4f, 0.1f,
        Color(0.05f, 0.05f, 0.05f), Color(1.0f, 1.0f, 1.0f), 128.0f);
    Sphere buttonLower(Vec3(0.0f, -0.1f, 0.4f), 0.03f, 0.5f, 0.4f, 0.1f,
        Color(0.05f, 0.05f, 0.05f), Color(1.0f, 1.0f, 1.0f), 128.0f);
    Sphere nose(Vec3(0.0f, 0.35f, 0.3f), 0.03f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 0.5f, 0.0f), Color(1.0f, 1.0f, 1.0f), 16.0f);

    Sphere spheres[] = { bottom, middle, top, eyeLeft, eyeRight, buttonUpper, buttonLower, nose };
    int numSpheres = sizeof(spheres) / sizeof(Sphere);
    /*Scene scene(spheres, numSpheres, Vec3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 1.0f, 0.0f),
        Color(1.0f, 1.0f, 1.0f), Color(0.0f, 0.0f, 0.0f), Color(0.2f, 0.2f, 0.2f));*/
    /*Scene scene(spheres, numSpheres, Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 1.0f, 1.0f),
        Color(1.0f, 1.0f, 1.0f), Color(0.1f, 0.1f, 0.1f), Color(0.2f, 0.2f, 0.2f));*/
    Scene scene(spheres, numSpheres, Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 3.0f, 1.0f),
            Color(1.0f, 1.0f, 1.0f), Color(0.7f, 0.7f, 0.9f), Color(0.5f, 0.5f, 0.8f));

    byte* imgBuffer = new byte[imgHeight * imgWidth * 3];

    raytrace(imgBuffer, imgHeight, imgWidth, &scene);

    PpmWriter writer("./out.ppm");
    writer.write(imgBuffer, imgHeight, imgWidth, true);
    free(imgBuffer);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
void raytrace(byte* imgBuffer, int imgHeight, int imgWidth, Scene* scene)
{
    byte* devImgBuffer = 0;
    size_t imgBufferSize = imgHeight * imgWidth * 3 * sizeof(byte);

    // Choose which GPU to run on, change this on a multi-GPU system.
    checkCudaErrors(cudaSetDevice(0));

    Sphere* devSpheres;
    Scene* devScene = scene;

    // Allocate GPU buffer for the image buffer
    checkCudaErrors(cudaMalloc((void**)&devImgBuffer, imgBufferSize));
    checkCudaErrors(cudaMalloc((void**)&devSpheres, scene->numSpheres * sizeof(Sphere)));
    checkCudaErrors(cudaMemcpy(devSpheres, scene->spheres, scene->numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    devScene->spheres = devSpheres; // I don't love this, but not sure how else to pass the whole object to the GPU
    checkCudaErrors(cudaMalloc((void**)&devScene, sizeof(Scene)));
    checkCudaErrors(cudaMemcpy(devScene, scene, sizeof(Scene), cudaMemcpyHostToDevice));

    // Launch a kernel on the GPU with one thread for each pixel.
    const int MAX_THREADS_PER_BLOCK = 1024;
    int numBlocks = (imgHeight * imgWidth / MAX_THREADS_PER_BLOCK) + 1;
    tracePixel<<<numBlocks, MAX_THREADS_PER_BLOCK>>>(devImgBuffer, imgHeight, imgWidth, devScene);

    // Check for any errors launching the kernel
    checkCudaErrors(cudaGetLastError());
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy output vector from GPU buffer to host memory.
    checkCudaErrors(cudaMemcpy(imgBuffer, devImgBuffer, imgBufferSize, cudaMemcpyDeviceToHost));
}

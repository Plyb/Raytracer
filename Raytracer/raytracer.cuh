#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.cuh"
#include "Scene.cuh"
#include "Ray.cuh"
#include "Sphere.cuh"
using namespace std;

inline __global__
void createScene(const Scene** scene) {
    if (!(threadIdx.x == 0 && blockIdx.x == 0)) {
        return;
    }

    Sphere* whiteSphere = new Sphere(Vec3(0.45f, -0.0f, -0.15f), 0.15f, 0.3f, 0.1f, 0.3f, 0.3f,
        Color(1.0f, 1.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 4.0f);
    Sphere* redSphere = new Sphere(Vec3(0.0f, 0.0f, -0.1f), 0.2f, 0.5f, 0.3f, 0.1f, 0.1f,
        Color(1.0f, 0.0f, 0.0), Color(1.0f, 1.0f, 1.0f), 32.0f);
    Sphere* greenSphere = new Sphere(Vec3(-0.6f, 0.0f, 0.0f), 0.3f, 0.05f, 0.025f, 0.025f, 0.9f,
        Color(0.0f, 1.0f, 0.0f), Color(0.5f, 1.0f, 0.5f), 64.0f);
    Sphere* blueSphere = new Sphere(Vec3(0.0f, -10000.5, 0.0f), 10000.0f, 0.1f, 0.0f, 0.1f, 0.8f,
        Color(0.0f, 0.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 16.0);

    /*Sphere purpleSphere(Vec3(0.0f, 0.0f, 0.0f), 0.4f, 0.7f, 0.2f, 0.1f,
        Color(1.0f, 0.0f, 1.0f), Color(1.0f, 1.0f, 1.0f), 16.0f);*/

        /*Sphere bottom(Vec3(0.0f, -0.5f, 0.0f), 0.5f, 0.7f, 0.2f, 0.1f,
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
            Color(1.0f, 0.5f, 0.0f), Color(1.0f, 1.0f, 1.0f), 16.0f);*/

    int numHittables = 4;
    const Hittable** hittables = new const Hittable*[numHittables]{ whiteSphere, redSphere, greenSphere, blueSphere };
    /*Scene scene(spheres, numSpheres, Vec3(0.0f, 0.0f, 1.0f), Vec3(0.0f, 1.0f, 0.0f),
        Color(1.0f, 1.0f, 1.0f), Color(0.0f, 0.0f, 0.0f), Color(0.2f, 0.2f, 0.2f));*/
    *(scene) = new Scene(hittables, numHittables, Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 1.0f, 1.0f),
        Color(1.0f, 1.0f, 1.0f), Color(0.1f, 0.1f, 0.1f), Color(0.2f, 0.2f, 0.2f));
    /*Scene scene(spheres, numSpheres, Vec3(0.0f, 0.0f, 1.0f), Vec3(1.0f, 3.0f, 1.0f),
            Color(1.0f, 1.0f, 1.0f), Color(0.7f, 0.7f, 0.9f), Color(0.5f, 0.5f, 0.8f));*/

}

inline __global__
void tracePixel(byte* imgBuffer, int imgHeight, int imgWidth, int reflectionDepth, const Scene** scene) {
    int pixelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelX = pixelIndex % imgWidth;
    int pixelY = (imgHeight - pixelIndex / imgWidth - 1);
    int i = pixelIndex * 3;

    if (pixelIndex >= imgWidth * imgHeight) {
        return;
    }

    float sy = pixelY / float(imgHeight >> 1) - 1.0f;
    float sx = pixelX / float(imgHeight >> 1) - 1.0f;

    Ray* ray = new Ray(Vec3((*scene)->camPos), (Vec3(sx, sy, 0.0f) - (*scene)->camPos).normalize(), 1.0f);
    const Hittable* excluded = NULL;
    Color color(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < reflectionDepth + 1; i++) {
        if (ray == NULL) {
            break;
        }
        Ray::RayResult res = ray->getColor((*scene), excluded);
        color = color + res.color;
        delete ray;
        ray = res.reflectedRay;
        excluded = res.hittable;
    }

    imgBuffer[i + 0] = color.byteR();
    imgBuffer[i + 1] = color.byteG();
    imgBuffer[i + 2] = color.byteB();
}
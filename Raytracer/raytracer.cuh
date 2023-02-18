#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.cuh"
#include "Sphere.cuh"
#include "Scene.cuh"
#include "Ray.cuh"
using namespace std;

inline __global__
void tracePixel(byte* imgBuffer, int imgHeight, int imgWidth, int reflectionDepth, const Scene* scene) {
    int pixelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelX = pixelIndex % imgWidth;
    int pixelY = (imgHeight - pixelIndex / imgWidth - 1);
    int i = pixelIndex * 3;

    if (pixelIndex >= imgWidth * imgHeight) {
        return;
    }

    float sy = pixelY / float(imgHeight >> 1) - 1.0f;
    float sx = pixelX / float(imgHeight >> 1) - 1.0f;
    Ray* ray = new Ray(Vec3(scene->camPos), (Vec3(sx, sy, 0.0f) - scene->camPos).normalize(), 1.0f);
    const Sphere* excluded = NULL;
    Color color(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < reflectionDepth + 1; i++) {
        if (ray == NULL) {
            break;
        }
        Ray::RayResult res = ray->getColor(scene, excluded);
        color = color + res.color;
        delete ray;
        ray = res.reflectedRay;
        excluded = res.sphere;
    }

    imgBuffer[i + 0] = color.byteR();
    imgBuffer[i + 1] = color.byteG();
    imgBuffer[i + 2] = color.byteB();
}
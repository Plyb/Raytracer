#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Vec3.cuh"
#include "Sphere.cuh"
#include "Scene.cuh"
using namespace std;

inline __device__
Vec3 tToVec3(float t, const Vec3 r0, const Vec3 rd) {
    return r0 + (rd * t);
}

inline __device__
bool sphereRayIntersect(const Vec3 r0, const Vec3 rd, const Vec3 center, float r, Vec3* res) {
    float b = 2 * (rd.x * (r0.x - center.x) + rd.y * (r0.y - center.y) + rd.z * (r0.z - center.z));
    float c = r0.x * r0.x - 2 * r0.x * center.x + center.x * center.x +
        r0.y * r0.y - 2 * r0.y * center.y + center.y * center.y +
        r0.z * r0.z - 2 * r0.z * center.z + center.z * center.z - r * r;
    float d = b * b - 4 * c;

    if (d >= 0) {
        float t0 = (-b - sqrtf(d)) / 2;
        if (t0 > 0) {
            res->set(tToVec3(t0, r0, rd));
            return true;
        }
        else {
            float t1 = (-b + sqrtf(d)) / 2;
            if (t1 > 0) {
                res->set(tToVec3(t1, r0, rd));
                return true;
            }
        }
    }
    return false;
}

inline __device__
Vec3 sphereNormal(const Vec3 p, const Sphere* sphere) {
    Vec3 center = sphere->center;
    float r = sphere->radius;
    return Vec3((p.x - center.x) / r, (p.y - center.y) / r, (p.z - center.z) / r);
}

inline __device__
Color phong(const Sphere* sphere, const Scene* scene, const Vec3 intersection) {
    Vec3 v = (scene->camPos - intersection).normalize();
    Vec3 normal = sphereNormal(intersection, sphere);
    float ldn = scene->lightDirection.dot(normal);
    Color Id = ldn > 0.0f
        ? scene->lightColor * sphere->diffuseColor * sphere->kd * ldn
        : Color(0.0f, 0.0f, 0.0f);
    Color Is = ldn > 0.0f
        ? scene->lightColor * sphere->specularColor * sphere->ks *
            powf((normal * 2 * ldn - scene->lightDirection).dot(v), sphere->kGls)
        : Color(0.0f, 0.0f, 0.0f);
    Color Ia = scene->ambientLightColor * sphere->ka;
    return Ia + Id + Is;
}

inline __global__
void tracePixel(byte* imgBuffer, int imgHeight, int imgWidth, const Scene* scene) {
    int pixelIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int pixelX = pixelIndex % imgWidth;
    int pixelY = (imgHeight - pixelIndex / imgWidth - 1);
    int i = pixelIndex * 3;

    if (pixelIndex >= imgWidth * imgHeight) {
        return;
    }

    float sy = pixelY / float(imgHeight >> 1) - 1.0f;
    float sx = pixelX / float(imgHeight >> 1) - 1.0f;
    Vec3 rd = (Vec3(sx, sy, 0.0f) - scene->camPos).normalize();

    Vec3 closestIntersection = Vec3(0.0f, 0.0f, 0.0f);
    Sphere closestSphere = Sphere();
    float closestDistance = -1.0f;
    for (int i = 0; i < scene->numSpheres; i++) {
        Vec3* intersection = new Vec3(0.0f, 0.0f, 0.0f);
        Sphere sphere = scene->spheres[i];
        if (sphereRayIntersect(scene->camPos, rd, sphere.center, sphere.radius, intersection)) {
            float distance = scene->camPos.distance(*intersection);
            if (closestDistance == -1.0f || distance < closestDistance) {
                closestIntersection = *intersection;
                closestDistance = distance;
                closestSphere = sphere;
            }
        }
        free(intersection);
    }
    bool intersected = closestDistance != -1.0f;

    Color color = intersected ? phong(&closestSphere, scene, closestIntersection) : scene->bgColor;

    imgBuffer[i + 0] = color.byteR();
    imgBuffer[i + 1] = color.byteG();
    imgBuffer[i + 2] = color.byteB();
}
#pragma once
#include "Vec3.cuh"
class Ray {
public:
	Vec3 origin;
	Vec3 dir;
	__device__ Ray(Vec3 origin, Vec3 dir) : origin(origin), dir(dir) {}

	__device__ Color getColor(const Scene* scene) {
        Vec3 closestIntersection = Vec3(0.0f, 0.0f, 0.0f);
        Sphere closestSphere = Sphere();
        float closestDistance = -1.0f;
        for (int i = 0; i < scene->numSpheres; i++) {
            Vec3* intersection = new Vec3(0.0f, 0.0f, 0.0f);
            Sphere sphere = scene->spheres[i];
            if (intersectsSphere(sphere.center, sphere.radius, intersection)) {
                float distance = origin.distance(*intersection);
                if (closestDistance == -1.0f || distance < closestDistance) {
                    closestIntersection = *intersection;
                    closestDistance = distance;
                    closestSphere = sphere;
                }
            }
            free(intersection);
        }
        bool intersected = closestDistance != -1.0f;
        if (!intersected) {
            return scene->bgColor;
        }

        return phong(&closestSphere, scene, closestIntersection);
	}

private:
    __device__
    bool intersectsSphere(const Vec3 center, float r, Vec3* res) {
        float b = 2 * (dir.x * (origin.x - center.x) + dir.y * (origin.y - center.y) + dir.z * (origin.z - center.z));
        float c = origin.x * origin.x - 2 * origin.x * center.x + center.x * center.x +
            origin.y * origin.y - 2 * origin.y * center.y + center.y * center.y +
            origin.z * origin.z - 2 * origin.z * center.z + center.z * center.z - r * r;
        float d = b * b - 4 * c;

        if (d >= 0) {
            float t0 = (-b - sqrtf(d)) / 2;
            if (t0 > 0) {
                res->set(tToVec3(t0));
                return true;
            }
            else {
                float t1 = (-b + sqrtf(d)) / 2;
                if (t1 > 0) {
                    res->set(tToVec3(t1));
                    return true;
                }
            }
        }
        return false;
    }

    __device__
    Color phong(const Sphere* sphere, const Scene* scene, const Vec3 intersection) {
        Vec3 v = (origin - intersection).normalize();
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
    
    __device__
    Vec3 tToVec3(float t) {
        return origin + (dir * t);
    }
    
    __device__
    Vec3 sphereNormal(const Vec3 p, const Sphere* sphere) {
        Vec3 center = sphere->center;
        float r = sphere->radius;
        return Vec3((p.x - center.x) / r, (p.y - center.y) / r, (p.z - center.z) / r);
    }
};
#pragma once
#include "Vec3.cuh"
#include "RayHit.cuh"
#include "RayHit.cuh"
class Ray {
public:
	Vec3 origin;
	Vec3 dir;
	__device__ Ray(Vec3 origin, Vec3 dir) : origin(origin), dir(dir) {}

	__device__ Color getColor(const Scene* scene, const Sphere* excludeSphere) {
        RayHit hit = getFirstIntersection(scene, excludeSphere);
        if (!hit.hasIntersection()) {
            return scene->bgColor;
        }

        bool inShadow = Ray(hit.point, scene->lightDirection)
            .getFirstIntersection(scene, hit.sphere)
            .hasIntersection();

        return phong(hit.sphere, scene, hit.point, inShadow);
	}

private:
    __device__ RayHit getFirstIntersection(const Scene* scene, const Sphere* excludeSphere) {
        RayHit closestHit = RayHit();
        for (int i = 0; i < scene->numSpheres; i++) {
            const Sphere* sphere = &scene->spheres[i];
            if (sphere == excludeSphere) {
                break;
            }

            RayHit hit = intersectsSphere(sphere);
            if (hit.hasIntersection()) {
                if (hit.distance < closestHit.distance) {
                    closestHit = hit;
                }
            }
        }
        return closestHit;
    }

    __device__
    RayHit intersectsSphere(const Sphere* sphere) {
        float b = 2 * (dir.x * (origin.x - sphere->center.x) +
            dir.y * (origin.y - sphere->center.y) +
            dir.z * (origin.z - sphere->center.z));
        float c = origin.x * origin.x - 2 * origin.x * sphere->center.x + sphere->center.x * sphere->center.x +
            origin.y * origin.y - 2 * origin.y * sphere->center.y + sphere->center.y * sphere->center.y +
            origin.z * origin.z - 2 * origin.z * sphere->center.z + sphere->center.z * sphere->center.z -
            sphere->radius * sphere->radius;
        float d = b * b - 4 * c;

        if (d >= 0) {
            float t0 = (-b - sqrtf(d)) / 2;
            if (t0 > 0) {
                Vec3 point = tToVec3(t0);
                return RayHit(sphere, point, origin.distance(point));
            }
            else {
                float t1 = (-b + sqrtf(d)) / 2;
                if (t1 > 0) {
                    Vec3 point = tToVec3(t1);
                    return RayHit(sphere, point, origin.distance(point));
                }
            }
        }
        return RayHit();
    }

    __device__
    Color phong(const Sphere* sphere, const Scene* scene, const Vec3 intersection, bool inShadow) {
        Color Ia = scene->ambientLightColor * sphere->ka;
        if (inShadow) {
            return Ia;
        }

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
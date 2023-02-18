#pragma once
#include "Vec3.cuh"
#include "RayHit.cuh"
#include "RayHit.cuh"
class Ray {
public:
    class RayResult {
    public:
        Color color;
        Ray* reflectedRay;
        const Sphere* sphere;
        __device__ RayResult(Color color, Ray* reflectedRay, const Sphere* sphere) : reflectedRay(reflectedRay), color(color), sphere(sphere) {}
    };

	Vec3 origin;
	Vec3 dir;
    float attenuation;
	__device__ Ray(Vec3 origin, Vec3 dir, float attenuation) :
        origin(origin), dir(dir), attenuation(attenuation) {}

	__device__ RayResult getColor(const Scene* scene, const Sphere* excludeSphere) {
        RayHit hit = getClosestIntersection(scene, excludeSphere);
        if (!hit.hasIntersection()) {
            return RayResult(scene->bgColor * attenuation, NULL, NULL);
        }

        bool inShadow = Ray(hit.point, scene->lightDirection, 0.0f)
            .hasIntersection(scene, hit.sphere);

        return RayResult(phong(scene, &hit, inShadow) * attenuation, reflection(scene, &hit), hit.sphere);
	}

private:
    __device__ RayHit getClosestIntersection(const Scene* scene, const Sphere* excludeSphere) {
        RayHit closestHit = RayHit();
        for (int i = 0; i < scene->numSpheres; i++) {
            const Sphere* sphere = scene->spheres + i;
            if (sphere == excludeSphere) {
                continue;
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

    __device__ bool hasIntersection(const Scene* scene, const Sphere* excludeSphere) {
        for (int i = 0; i < scene->numSpheres; i++) {
            const Sphere* sphere = &scene->spheres[i];
            if (sphere == excludeSphere) {
                break;
            }

            RayHit hit = intersectsSphere(sphere);
            if (hit.hasIntersection()) {
                return true;
            }
        }
        return false;
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
                return RayHit(sphere, point, origin.distance(point), dir);
            }
            else {
                float t1 = (-b + sqrtf(d)) / 2;
                if (t1 > 0) {
                    Vec3 point = tToVec3(t1);
                    return RayHit(sphere, point, origin.distance(point), dir);
                }
            }
        }
        return RayHit();
    }

    __device__
    Color phong(const Scene* scene, RayHit* hit, bool inShadow) {
        const Material* material = &hit->sphere->material;
        Color Ia = scene->ambientLightColor * material->ka;
        if (inShadow) {
            return Ia;
        }

        Vec3 v = (origin - hit->point).normalize();
        Vec3 normal = hit->getNormal();
        float ldn = scene->lightDirection.dot(normal);
        Color Id = ldn > 0.0f
            ? scene->lightColor * material->diffuseColor * material->kd * ldn
            : Color(0.0f, 0.0f, 0.0f);
        Color Is = ldn > 0.0f
            ? scene->lightColor * material->specularColor * material->ks *
            powf((normal * 2 * ldn - scene->lightDirection).dot(v), material->kGls)
            : Color(0.0f, 0.0f, 0.0f);
        return Ia + Id + Is;
    }

    __device__
    Ray* reflection(const Scene* scene, RayHit* hit) {
        Vec3 reflectedDir = hit->dir - hit->getNormal() * 2 * hit->dir.dot(hit->getNormal());
        return new Ray(hit->point, reflectedDir, attenuation * hit->sphere->material.kr);
    }
    
    __device__
    Vec3 tToVec3(float t) {
        return origin + (dir * t);
    }
};
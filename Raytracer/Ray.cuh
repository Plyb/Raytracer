#pragma once
#include "Vec3.cuh"
#include "RayHit.cuh"

class Ray {
public:
    class RayResult {
    public:
        Color color;
        Ray* reflectedRay;
        const Hittable* hittable;
        __device__ RayResult(Color color, Ray* reflectedRay, const Hittable* hittable) : reflectedRay(reflectedRay), color(color), hittable(hittable) {}
    };

	Vec3 origin;
	Vec3 dir;
    float attenuation;
	__device__ Ray(Vec3 origin, Vec3 dir, float attenuation) :
        origin(origin), dir(dir), attenuation(attenuation) {}

	__device__ RayResult getColor(const Scene* scene, const Hittable* excludeHittable) {
        RayHit hit = getClosestIntersection(scene, excludeHittable);
        if (!hit.hasIntersection()) {
            return RayResult(scene->bgColor * attenuation, NULL, NULL);
        }

        bool inShadow = Ray(hit.point, scene->lightDirection, 0.0f)
            .hasIntersection(scene, hit.hittable);

        return RayResult(phong(scene, &hit, inShadow) * attenuation, reflection(scene, &hit), hit.hittable);
	}

private:
    __device__ RayHit getClosestIntersection(const Scene* scene, const Hittable* excludeHittable) {
        RayHit closestHit = RayHit();
        for (int i = 0; i < scene->numHittables; i++) {
            const Hittable* hittable = scene->hittables[i];
            if (hittable == excludeHittable) {
                continue;
            }

            RayHit hit = hittable->intersects(origin, dir);
            if (hit.hasIntersection()) {
                if (hit.distance < closestHit.distance) {
                    closestHit = hit;
                }
            }
        }
        return closestHit;
    }

    __device__ bool hasIntersection(const Scene* scene, const Hittable* excludeHittable) {
        for (int i = 0; i < scene->numHittables; i++) {
            const Hittable* hittable = scene->hittables[i];
            if (hittable == excludeHittable) {
                break;
            }

            RayHit hit = hittable->intersects(origin, dir);
            if (hit.hasIntersection()) {
                return true;
            }
        }
        return false;
    }

    __device__
    Color phong(const Scene* scene, RayHit* hit, bool inShadow) {
        const Material* material = &hit->hittable->material;
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
        return new Ray(hit->point, reflectedDir, attenuation * hit->hittable->material.kr);
    }
};
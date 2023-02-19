#pragma once
#include "Vec3.cuh"
#include "Material.cuh"
class Sphere : public Hittable {
public:
	Vec3 center;
	float radius;
	__device__ Sphere(Vec3 center, float radius, float kd, float ks, float ka, float kr,
			Color diffuseColor, Color specularColor, float kGls) :
		center(center), radius(radius), Hittable(kd, ks, ka, kr, diffuseColor, specularColor, kGls) {}

	__device__ Sphere() {}

	__device__ RayHit intersects(Vec3 origin, Vec3 dir) const override {
        float b = 2 * (dir.x * (origin.x - center.x) +
            dir.y * (origin.y - center.y) +
            dir.z * (origin.z - center.z));
        float c = origin.x * origin.x - 2 * origin.x * center.x + center.x * center.x +
            origin.y * origin.y - 2 * origin.y * center.y + center.y * center.y +
            origin.z * origin.z - 2 * origin.z * center.z + center.z * center.z -
            radius * radius;
        float d = b * b - 4 * c;

        if (d >= 0) {
            float t0 = (-b - sqrtf(d)) / 2;
            if (t0 > 0) {
                Vec3 point = tToVec3(origin, dir, t0);
                return RayHit(this, point, origin.distance(point), dir);
            }
            else {
                float t1 = (-b + sqrtf(d)) / 2;
                if (t1 > 0) {
                    Vec3 point = tToVec3(origin, dir, t1);
                    return RayHit(this, point, origin.distance(point), dir);
                }
            }
        }
        return RayHit();
	}

    __device__ Vec3 getNormal(Vec3 point) const override {
        float r = radius;
        Vec3 p = point;
        return Vec3((p.x - center.x) / r, (p.y - center.y) / r, (p.z - center.z) / r);
    }
};
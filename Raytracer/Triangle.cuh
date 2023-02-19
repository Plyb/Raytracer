#pragma once
#include "RayHit.cuh"
#include "Vec3.cuh"

class Triangle : public Hittable {
public:
	Vec3 points[3];
	__device__ Triangle(Vec3 p1, Vec3 p2, Vec3 p3, float kd, float ks, float ka, float kr,
			Color diffuseColor, Color specularColor, float kGls) :
		points{ p1, p2, p3 }, Hittable(kd, ks, ka, kr, diffuseColor, specularColor, kGls) {}

	__device__ Triangle() {}

	__device__ RayHit intersects(Vec3 origin, Vec3 dir) const override {
		Vec3 n = getNormal(Vec3());
		float nDotDir = n.dot(dir);
		float interiorMultiplier = 1.0f;
		if (nDotDir < 0) {
			n = -n;
			nDotDir = -nDotDir;
			interiorMultiplier = -1.0f;
		}
		else if (nDotDir == 0) {
			return RayHit();
		}

		float d = - n.dot(points[0]);

		float t = -(n.dot(origin) + d) / (n.dot(dir));
		if (t <= 0) {
			return RayHit();
		}

		Vec3 point = tToVec3(origin, dir, t);

		for (int i = 0; i < 3; i++) {
			if (interiorMultiplier * (point - points[i]).cross(points[(i + 1) % 3] - points[i]).dot(n) > 0.0f) {
				return RayHit();
			}
		}
		return RayHit(this, point, origin.distance(point), dir);
	}

	__device__ Vec3 getNormal(Vec3 point) const override {
		Vec3 v1 = points[1] - points[0];
		Vec3 v2 = points[2] - points[1];
		return v1.cross(v2).normalize();
	}
};
#pragma once
#include "Sphere.cuh"
#include <cmath>
class RayHit {
public:
	const Sphere* sphere;
	Vec3 point;
	float distance;
	Vec3 dir;
	__device__ RayHit(const Sphere* sphere, Vec3 point, float distance, Vec3 dir) :
		sphere(sphere), point(point), distance(distance), dir(dir) {}
	__device__ RayHit() : sphere(NULL), point(Vec3()), distance(INFINITY), dir(Vec3()) {}

	__device__ bool hasIntersection() {
		return distance != INFINITY;
	}

	__device__ Vec3 getNormal() {
		if (normalCalculated) {
			return normal;
		}
		Vec3 center = sphere->center;
		float r = sphere->radius;
		Vec3 p = point;
		normal = Vec3((p.x - center.x) / r, (p.y - center.y) / r, (p.z - center.z) / r);
		normalCalculated = true;

		return normal;
	}

private:
	bool normalCalculated = false;
	Vec3 normal = Vec3();
};
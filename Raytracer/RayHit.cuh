#pragma once
#include "Sphere.cuh"
#include <cmath>
class RayHit {
public:
	const Sphere* sphere;
	Vec3 point;
	float distance;
	__device__ RayHit(const Sphere* sphere, Vec3 point, float distance) :
		sphere(sphere), point(point), distance(distance) {}
	__device__ RayHit() : sphere(NULL), point(Vec3()), distance(INFINITY) {}

	__device__ bool hasIntersection() {
		return distance != INFINITY;
	}
};
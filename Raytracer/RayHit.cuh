#pragma once
#include <cmath>
#include "Material.cuh"

class RayHit;

class Hittable {
public:
	Material material;

	__device__ virtual RayHit intersects(Vec3 origin, Vec3 dir) const = 0;
	__device__ virtual Vec3 getNormal(Vec3 point) const = 0;

	__device__ Hittable(float kd, float ks, float ka, float kr, Color diffuseColor, Color specularColor, float kGls) :
		material(Material(ka, kd, ks, kr, diffuseColor, specularColor, kGls)) {}

	__device__ Hittable() {}
};

class RayHit {
public:
	const Hittable* hittable;
	Vec3 point;
	float distance;
	Vec3 dir;
	__device__ RayHit(const Hittable* hittable, Vec3 point, float distance, Vec3 dir) :
		hittable(hittable), point(point), distance(distance), dir(dir) {}
	__device__ RayHit() : hittable(NULL), point(Vec3()), distance(INFINITY), dir(Vec3()) {}

	__device__ bool hasIntersection() {
		return distance != INFINITY;
	}

	__device__ Vec3 getNormal() {
		if (normalCalculated) {
			return normal;
		}
		normal = hittable->getNormal(point);
		normalCalculated = true;

		return normal;
	}

private:
	bool normalCalculated = false;
	Vec3 normal = Vec3();
};
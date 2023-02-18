#pragma once
#include "Vec3.cuh"
#include "Material.cuh"
class Sphere {
public:
	Vec3 center;
	float radius;
	Material material;
	Sphere(Vec3 center, float radius, float kd, float ks, float ka, float kr,
			Color diffuseColor, Color specularColor, float kGls) :
		center(center), radius(radius), material(Material(ka, kd, ks, kr, diffuseColor, specularColor, kGls)) {}

	__device__ Sphere() {}
};
#pragma once
#include "Vec3.cuh"
class Sphere {
public:
	Vec3 center;
	float radius;
	float ka;
	float kd;
	float ks;
	float kr;
	Color diffuseColor;
	Color specularColor;
	float kGls;
	Sphere(Vec3 center, float radius, float kd, float ks, float ka, float kr,
			Color diffuseColor, Color specularColor, float kGls) :
		center(center), radius(radius), ka(ka), kd(kd), ks(ks), kr(kr), diffuseColor(diffuseColor),
		specularColor(specularColor), kGls(kGls) {}

	__device__ Sphere() {}
};
#pragma once
#include "Color.cuh"

class Material {
public:
	float ka;
	float kd;
	float ks;
	float kr;
	Color diffuseColor;
	Color specularColor;
	float kGls;
	Material(float ka, float kd, float ks, float kr, Color diffuseColor, Color specularColor, float kGls) :
		ka(ka), kd(kd), ks(ks), kr(kr), diffuseColor(diffuseColor), specularColor(specularColor), kGls(kGls) {}

	__device__ Material() {}
};
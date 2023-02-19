#pragma once
#include "RayHit.cuh"

class Scene {
public:
	const Hittable** hittables;
	int numHittables;
	Vec3 camPos;
	Vec3 lightDirection;
	Color lightColor;
	Color ambientLightColor;
	Color bgColor;
	__device__ Scene(const Hittable** hittables, int numHittables, Vec3 camPos, Vec3 lightDirection,
			Color lightColor, Color ambientLightColor, Color bgColor) :
		hittables(hittables), numHittables(numHittables), camPos(camPos), lightDirection(lightDirection.normalize()),
		lightColor(lightColor), ambientLightColor(ambientLightColor), bgColor(bgColor) {}
};
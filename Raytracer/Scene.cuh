#pragma once
#include "Sphere.cuh"
class Scene {
public:
	const Sphere* spheres;
	int numSpheres;
	Vec3 camPos;
	Vec3 lightDirection;
	Color lightColor;
	Color ambientLightColor;
	Color bgColor;
	Scene(const Sphere* spheres, int numSpheres, Vec3 camPos, Vec3 lightDirection, Color lightColor,
			Color ambientLightColor, Color bgColor) :
		spheres(spheres), numSpheres(numSpheres), camPos(camPos), lightDirection(lightDirection.normalize()),
		lightColor(lightColor), ambientLightColor(ambientLightColor), bgColor(bgColor) {}
};
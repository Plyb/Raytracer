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
	__device__ Material(float ka, float kd, float ks, float kr, Color diffuseColor, Color specularColor, float kGls) :
		ka(ka), kd(kd), ks(ks), kr(kr), diffuseColor(diffuseColor), specularColor(specularColor), kGls(kGls) {}

	__device__ Material() {}

    __device__
    Color phong(bool inShadow, Vec3 viewingDirection, Vec3 normal, Vec3 lightDirection, Color lightColor, Color ambientLightColor) const {
        Color Ia = ambientLightColor * ka;
        if (inShadow) {
            return Ia;
        }

        float ldn = lightDirection.dot(normal);
        Color Id = ldn > 0.0f
            ? lightColor * diffuseColor * kd * ldn
            : Color(0.0f, 0.0f, 0.0f);
        Color Is = ldn > 0.0f
            ? lightColor * specularColor * ks *
            powf((normal * 2 * ldn - lightDirection).dot(viewingDirection), kGls)
            : Color(0.0f, 0.0f, 0.0f);
        return Ia + Id + Is;
    }
};
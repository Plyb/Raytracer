#pragma once
#include "cuda_runtime.h"
#include "Color.cuh"

class Vec3
{
public:
	__host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__device__ Vec3() {}
	float x, y, z;

	__device__ void set(Vec3 o) {
		x = o.x;
		y = o.y;
		z = o.z;
	}

	__host__ __device__ float length() const {
		return sqrtf(x * x + y * y + z * z);
	}

	__host__ __device__ Vec3 normalize() const {
		float length = this->length();
		return Vec3(x / length, y / length, z / length);
	}

	__device__ Vec3 operator-(Vec3 o) const {
		return Vec3(x - o.x, y - o.y, z - o.z);
	}

	__device__ Vec3 operator+(Vec3 o) const {
		return Vec3(x + o.x, y + o.y, z + o.z);
	}

	__device__ Vec3 operator*(float o) const {
		return Vec3(x * o, y * o, z * o);
	}

	__device__ float dot(Vec3 o) const {
		return x * o.x + y * o.y + z * o.z;
	}

	__device__ Color toColor() const {
		return Color((x + 1.0f) / 2.0f, (y + 1.0f) / 2.0f, (z + 1.0f) / 2.0f);
	}

	__device__ float distance(const Vec3 o) const {
		float dx = x - o.x;
		float dy = y - o.y;
		float dz = z - o.z;
		return sqrtf(dx * dx + dy * dy + dz * dz);
	}
};


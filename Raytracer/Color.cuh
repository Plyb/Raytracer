#pragma once
#include "cuda_runtime.h"

typedef unsigned char byte;

class Color
{
public:
	__host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b) {}
	Color(float i) : r(i), g(i), b(i) {}
	__device__ Color() : r(0), g(0), b(0) {}
	float r, g, b;
	
	__device__ byte byteR() const {
		return unsigned char(r * 255);
	}
	__device__ byte byteG() const {
		return unsigned char(g * 255);
	}
	__device__ byte byteB() const {
		return unsigned char(b * 255);
	}

	__device__ Color operator*(float o) const {
		return Color(r * o, g * o, b * o);
	}
	__device__ Color operator*(Color o) const {
		return Color(r * o.r, g * o.g, b * o.b);
	}
	__device__ Color operator+(Color o) const {
		Color res = Color(r + o.r, g + o.g, b + o.b);
		if (res.r > 1.0f) {
			res.r = 1.0f;
		}
		if (res.g > 1.0f) {
			res.g = 1.0f;
		}
		if (res.b > 1.0f) {
			res.b = 1.0f;
		}
		return res;
	}
};


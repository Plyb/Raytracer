#pragma once
#include "Vec3.cuh"

__device__
void Vec3::set(Vec3 o) {
	x = o.x;
	y = o.y;
	z = o.z;
}

__device__
Vec3 Vec3::operator-(Vec3 o) {
	return Vec3(x - o.x, y - o.y, z - o.z);
}

__device__
Vec3 Vec3::operator+(Vec3 o) const {
	return Vec3(x + o.x, y + o.y, z + o.z);
}

__device__
Vec3 Vec3::operator*(float o) const {
	return Vec3(x * o, y * o, z * o);
}
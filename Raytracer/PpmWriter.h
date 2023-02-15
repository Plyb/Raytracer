#pragma once
#include <string>

typedef unsigned char byte;

class PpmWriter {
public:
	PpmWriter(std::string path);

	void write(const byte* buffer, int height, int width, bool p6);
private:
	std::string path;
};
#include "PpmWriter.h"

#include <iostream>
#include <fstream>

PpmWriter::PpmWriter(std::string path) : path(path) {}

void PpmWriter::write(const byte* buffer, int height, int width, bool p6) {
	std::ofstream outFile(path, std::ios_base::binary);

	outFile << (p6 ? "P6 " : "P3 ");
	outFile << width << ' ' << height << " 255 ";
	if (p6) {
		outFile.write((char*)buffer, height * width * 3);
	}
	else {
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				int pixelIndex = j * 3 * width + i * 3;
				byte r = buffer[pixelIndex + 0];
				byte g = buffer[pixelIndex + 1];
				byte b = buffer[pixelIndex + 2];

				outFile << int(r) << ' ' << int(g) << ' ' << int(b) << "\n";
			}
		}
	}
}
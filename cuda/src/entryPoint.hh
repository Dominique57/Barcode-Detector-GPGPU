#pragma once

#include <string>

void executeAlgorithm(const std::string &path);

void handleImage(const std::string &imagePath);
void handleVideo(const std::string &videoPath);
void handleCamera(unsigned cameraId = 0);
void generatePredictedRgb(const std::string &imagePath);
void generateLbpOutFile(const std::vector<std::string> &imagePaths);
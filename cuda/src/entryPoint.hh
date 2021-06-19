#pragma once

#include <string>

void executeAlgorithm(const std::string &path);

void handleImage(const std::string &imagePath);
void handleVideo(const std::string &videoPath);
void handleCamera();
void generatePredictedRgb(const std::string &image);
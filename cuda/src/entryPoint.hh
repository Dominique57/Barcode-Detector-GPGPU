#pragma once

#include <string>

void executeAlgorithm(const std::string& databasePath, const std::string &path);

void handleImage(const std::string& databasePath, const std::string &imagePath);
void handleVideo(const std::string& databasePath, const std::string &videoPath);
void handleCamera(const std::string& databasePath, unsigned cameraId = 0);
void generatePredictedRgb(const std::string& databasePath, const std::string &imagePath);
void generateLbpOutFile(const std::string& databasePath, const std::vector<std::string> &imagePaths);

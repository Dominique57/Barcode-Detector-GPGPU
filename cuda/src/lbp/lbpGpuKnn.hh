#pragma once

#include <cuda/mycuda.hh>
#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpGpuKnn {
public:
    LbpGpuKnn(unsigned width, unsigned height, const std::string &path);

    ~LbpGpuKnn();

    void run(cv::Mat_<uchar> &grayImage, std::vector<uchar> &labels);

private:
    unsigned width_;
    unsigned height_;
    Matrix<uchar> cudaImage_;
    Matrix<float> cudaCentroids_;
};
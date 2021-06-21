#pragma once

#include <cuda/mycuda.hh>
#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpGpu {
public:
    LbpGpu(unsigned width, unsigned height);

    ~LbpGpu();

    void run(cv::Mat_<uchar> &grayImage);

    cv::Mat_<uchar> &getFeatures();
    Matrix<uchar> &getCudaFeatures() { return cudaFeatures_; }
    unsigned numberOfPatches() const {
        return (width_ / SLICE_SIZE) * (height_ / SLICE_SIZE);
    }

private:
    unsigned width_;
    unsigned height_;
    cv::Mat_<uchar> features_;
    Matrix<uchar> cudaImage_;
    Matrix<uchar> cudaFeatures_;
};
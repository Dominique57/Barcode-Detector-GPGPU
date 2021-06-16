#pragma once

#include <cuda/mycuda.hh>
#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpGpu : public LbpAlgorithm {
public:
    LbpGpu(unsigned width, unsigned height);

    ~LbpGpu();

    void run(cv::Mat_<uchar> &grayImage) override;

    cv::Mat_<float> &getFeatures() override;
    Matrix<> &getCudaFeatures() { return cudaFeatures_; }

private:
    Matrix<uchar> cudaImage_;
    Matrix<> cudaFeatures_;
};
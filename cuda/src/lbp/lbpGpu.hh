#pragma once

#include <cuda/mycuda.hh>
#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpGpu : public LbpAlgorithm {
public:
    LbpGpu(unsigned width, unsigned height);

    ~LbpGpu();

    void run(Matrix<> &grayImage) override;

    Matrix<> &getFeatures() override;
    Matrix<> &getCudaFeatures();

private:
    Matrix<> cudaImage_;
    Matrix<> cudaFeatures_;
};
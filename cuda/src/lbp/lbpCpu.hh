#pragma once

#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpCpu : public LbpAlgorithm {
public:
    LbpCpu(unsigned width, unsigned height);

    void run(Matrix<> &grayImage) override;

    Matrix<> &getFeatures() override;

protected:

    void addLocalPatterns(Matrix<> &image);

    void addSliceTextons(const Splice &splice, unsigned sliceIndex);

private:
};
#pragma once

#include <image/matrix.hh>
#include <image/splice.hh>
#include "lbpAlgorithm.hh"

class LbpCpu : public LbpAlgorithm {
public:
    LbpCpu(unsigned width, unsigned height);

    void run(cv::Mat_<uchar> &grayImage) override;

protected:

    void addLocalPatterns(cv::Mat_<uchar> &image);

    void addSliceTextons(const cv::Mat_<uchar> &image, int x, int y, unsigned sliceIndex);

private:
};
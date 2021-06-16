#pragma once

#define SLICE_SIZE 16
#define NEIGHS_COUNT 8

#include "my_opencv/wrapper.hh"

class LbpAlgorithm {
public:
    LbpAlgorithm(unsigned width, unsigned height)
            : width_(width), height_(height),
              features_((width / SLICE_SIZE) * (height / SLICE_SIZE),1 << NEIGHS_COUNT, 0.)
    {}

    virtual void run(cv::Mat_<uchar> &grayImage) {
        if ((unsigned)grayImage.cols != width_ || (unsigned )grayImage.rows != height_)
            throw std::invalid_argument("Image has incorrect format !");
    }

    virtual cv::Mat_<float> &getFeatures() { return features_; };

    unsigned numberOfPatches() const { return (width_ / SLICE_SIZE) * (height_ / SLICE_SIZE); }

protected:
    unsigned width_;
    unsigned height_;
    cv::Mat_<float> features_;
};
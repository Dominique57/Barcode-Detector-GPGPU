#pragma once

#define SLICE_SIZE 16
#define NEIGHS_COUNT 8

class LbpAlgorithm {
public:
    LbpAlgorithm(unsigned width, unsigned height)
            : width_(width), height_(height),
              features_(1 << NEIGHS_COUNT,
                        (width / SLICE_SIZE) * (height / SLICE_SIZE))
    {
        if (width % SLICE_SIZE != 0 || height % SLICE_SIZE != 0)
            throw std::invalid_argument("Given format is not slice-zable !");
    }

    virtual void run(Matrix<> &grayImage) {
        if (grayImage.width() != width_ || grayImage.height() != height_)
            throw std::invalid_argument("Image has incorrect format !");
    }

    virtual Matrix<> &getFeatures() = 0;

    unsigned numberOfPatches() const { return (width_ / SLICE_SIZE) * (height_ / SLICE_SIZE); }

protected:
    unsigned width_;
    unsigned height_;
    Matrix<> features_;
};
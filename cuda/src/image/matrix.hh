#pragma once

#include <fstream>
#include <cstdlib>
#include <string>
#include <cuda/mycuda.hh>

class Splice;

template <typename T = float>
class Matrix {
public:
    friend Splice;
    using data_t = T;

    CUDA_HOST Matrix(unsigned width, unsigned height)
        : buffer_(static_cast<data_t*>(std::calloc(height, width * sizeof(data_t)))),
          width_(width), height_(height), allocatedBuffer_(true)
    {}

    CUDA_HOST Matrix(unsigned width, unsigned height, data_t* buffer)
            : buffer_(buffer), width_(width), height_(height), allocatedBuffer_(false)
    {}

    CUDA_HOSTDEV ~Matrix() {
        if (allocatedBuffer_ && buffer_ != nullptr)
            delete buffer_;
    }

    CUDA_HOSTDEV data_t *operator[](unsigned i) const { return buffer_ + (i * width_); }

    CUDA_HOSTDEV unsigned width() const { return width_; }
    CUDA_HOSTDEV unsigned height() const { return height_; }

    CUDA_HOST static void readMatrix(const std::string &path, Matrix<float> &matrix);
private:
    data_t* buffer_;
    unsigned width_;
    unsigned height_;
    bool allocatedBuffer_;
};
#pragma once

#include <fstream>
#include <cstdlib>
#include <string>
#include <cuda/mycuda.hh>
#include "my_opencv/wrapper.hh"

class Splice;

template<typename T = float>
class Matrix {
public:
    friend Splice;
    using data_t = T;


    CUDA_HOST Matrix(unsigned width, unsigned height, unsigned stride)
        : buffer_(static_cast<data_t *>(std::calloc(height, stride * sizeof(data_t)))),
          width_(width), height_(height), stride_(stride), allocatedBuffer_(true) {}

    CUDA_HOSTDEV Matrix(unsigned width, unsigned height, unsigned stride, data_t *buffer)
        : buffer_(buffer), width_(width), height_(height), stride_(stride), allocatedBuffer_(false) {}

    CUDA_HOST Matrix(unsigned width, unsigned height)
        : Matrix(width, height, width) {}

    CUDA_HOSTDEV Matrix(unsigned width, unsigned height, data_t *buffer)
        : Matrix(width, height, width, buffer) {}

    CUDA_HOSTDEV ~Matrix() {
        if (allocatedBuffer_ && buffer_ != nullptr)
            delete buffer_;
    }

    CUDA_HOSTDEV data_t *operator[](unsigned i) const { return buffer_ + (i * stride_); }

    CUDA_HOSTDEV data_t *&getData() { return buffer_; }

    CUDA_HOSTDEV unsigned width() const { return width_; }

    CUDA_HOSTDEV unsigned height() const { return height_; }

    CUDA_HOSTDEV size_t &getStride() { return stride_; }

    CUDA_HOST static void readMatrix(const std::string &path, Matrix<float> &matrix);

private:
    data_t *buffer_;
    unsigned width_;
    unsigned height_;
    size_t stride_;
    bool allocatedBuffer_;
};
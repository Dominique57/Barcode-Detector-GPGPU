#pragma once

#include <fstream>
#include <cstdlib>
#include <string>

class Splice;

template <typename T = float>
class Matrix {
public:
    friend Splice;
    using data_t = T;

    Matrix(unsigned width, unsigned height)
        : buffer_(static_cast<data_t*>(std::calloc(height, width * sizeof(data_t)))),
          width_(width), height_(height)
    {}

    ~Matrix() { delete buffer_; }

    data_t *operator[](unsigned i) const { return buffer_ + (i * width_); }

    unsigned width() const { return width_; }
    unsigned height() const { return height_; }

    static void readMatrix(const std::string &path, Matrix<float> &matrix);
private:
    data_t* buffer_;
    unsigned width_;
    unsigned height_;
};
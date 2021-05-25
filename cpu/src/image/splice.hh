#pragma once

#include "matrix.hh"

class Splice {
public:
    Splice(Matrix& matrix, unsigned x_start, unsigned y_start, unsigned width, unsigned height)
        : matrix_(matrix),
          x_start_(x_start),
          y_start_(y_start),
          width_(width),
          height_(height) {}

    Matrix::data_t *operator[](unsigned i) const {
        // get ith line After y_start and offset to x_start
        return matrix_.buffer_ + (matrix_.width_ * (y_start_ + i)) + x_start_;
    }

    bool isCoordsValid(int y, int x) const {
        return 0 <= y && y < static_cast<int>(height_) &&
               0 <= x && x < static_cast<int>(width_);
    }

    unsigned width() const { return width_; }
    unsigned height() const { return height_; }
private:
    Matrix& matrix_;
    unsigned x_start_;
    unsigned y_start_;
    unsigned width_;
    unsigned height_;
};
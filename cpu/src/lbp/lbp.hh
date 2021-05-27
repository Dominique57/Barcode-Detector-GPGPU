#pragma once

#include <stdexcept>
#include <iostream>
#include "lbpSplices.hh"

class Lbp {
public:
    Lbp(Matrix<>& matrix)
        : matrix_(matrix)
    {
        if (matrix_.width() % slicesSize != 0 || matrix_.height() % slicesSize != 0)
            throw std::invalid_argument("Matrix image input dimensions are not slices divisible !");
    }

    unsigned textonSize() const { return 1 << pixelsNeighs.size(); }
    unsigned patchCount() const { return matrix_.width() / slicesSize * matrix_.height() / slicesSize; }

    Matrix<> run() {
        if (pixelsNeighs.size() > 32)
            throw std::invalid_argument("Cannot represent a texton bigger than 32 bit !");
        auto resFeatures = Matrix<>(textonSize(),patchCount());

        auto splices = LbpSplices(matrix_, slicesSize);
        splices.addLocalPatterns(resFeatures, pixelsNeighs);

        return resFeatures;
        // Normalize histograms (opt)
        // Run k-neighbours
        // Post-process
        // Get codebar coordinates
        // Show using sdl or some other shit
    }

private:
    Matrix<>& matrix_;
    unsigned slicesSize = 16;
    const std::vector<std::pair<int, int>> pixelsNeighs {
            {-1, -1}, {0, -1}, {1, -1},
            {-1, 0},  {1, 0},
            {-1, 1}, {0, 1}, {1, 1},
    };
};
#pragma once

#include <stdexcept>
#include <iostream>
#include "lbpSplices.hh"

class Lbp {
public:
    Lbp(Matrix& matrix)
        : matrix_(matrix)
    {}

    void run() {
        unsigned slicesSize = 16;
        if (matrix_.width() % slicesSize != 0 || matrix_.height() % slicesSize != 0)
            throw std::invalid_argument("Matrix image input dimensions are not slices divisible !");

        auto pixelsNeighs = std::vector<std::pair<int, int>>{
                {-1, -1}, {0, -1}, {1, -1},
                {-1, 0},  {1, 0},
                {-1, 1}, {0, 1}, {1, 1},
        };
        if (pixelsNeighs.size() > 32)
            std::invalid_argument("Cannot represent a texton bigger than 32 bit !");
        auto resFeatures = Matrix(1 << pixelsNeighs.size(),
                                  matrix_.width() / slicesSize * matrix_.height() / slicesSize);

        auto splices = LbpSplices(matrix_, slicesSize);
        splices.addLocalPatterns(resFeatures, pixelsNeighs);

        // Normalize histograms (opt)
        // Run k-neighbours
        // Post-process
        // Get codebar coordinates
        // Show using sdl or some other shit
    }

private:
    Matrix& matrix_;
};
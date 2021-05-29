#pragma once

#include <stdexcept>
#include <iostream>
#include "lbpSplices.hh"

class Lbp {
public:
    Lbp(Matrix<>& matrix);

    unsigned textonSize() const { return 1 << pixelsNeighs.size(); }
    unsigned patchCount() const { return matrix_.width() / slicesSize * matrix_.height() / slicesSize; }

    Matrix<> run(bool gpu = false);

private:
    Matrix<>& matrix_;
    unsigned slicesSize = 16;
    const std::vector<std::pair<int, int>> pixelsNeighs {
            {-1, -1}, {0, -1}, {1, -1},
            {-1, 0},  {1, 0},
            {-1, 1}, {0, 1}, {1, 1},
    };
};
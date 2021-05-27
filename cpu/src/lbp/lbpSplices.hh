#pragma once

#include <vector>
#include "../image/splice.hh"

class LbpSplices {
public:
    LbpSplices(Matrix<> &image, unsigned slicesSize);

    void addLocalPatterns(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs);

protected:
    void addSliceTextons(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs,
                         const Splice &splice, unsigned sliceIndex);

private:
    static std::vector<std::pair<int, int>> textonsOffset;

    Matrix<> &image_;
    unsigned slicesSize_;

};
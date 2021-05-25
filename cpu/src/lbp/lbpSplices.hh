#pragma once

#include <vector>
#include "../image/splice.hh"

class LbpSplices {
public:
    LbpSplices(Matrix &image, unsigned slicesSize);

    void addLocalPattern(Matrix &resFeatures, const std::vector<std::pair<int, int>> &vector);

protected:
    void addSliceTexton(Matrix &resFeatures, const std::vector<std::pair<int, int>> &neighs,
                        const Splice &splice, unsigned sliceindex);

private:
    static std::vector<std::pair<int, int>> textonsOffset;

    Matrix &image_;
    unsigned slicesSize_;

};
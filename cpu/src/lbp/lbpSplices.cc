#include <iostream>
#include "lbpSplices.hh"

LbpSplices::LbpSplices(Matrix<> &image, unsigned int slicesSize)
        : image_(image), slicesSize_(slicesSize)
{}

void LbpSplices::addLocalPatterns(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs) {
    unsigned sliceIndex = 0;
    for (auto y = 0U; y < image_.height(); y += slicesSize_) {
        for (auto x = 0U; x < image_.width(); x += slicesSize_) {
            auto slice = Splice(image_, x, y, slicesSize_, slicesSize_);
            addSliceTextons(resFeatures, neighs, slice, sliceIndex);
            sliceIndex += 1;
        }
    }
}

void LbpSplices::addSliceTextons(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs,
                                 const Splice &splice, unsigned sliceIndex) {
    for (auto y = 0U; y < splice.height(); ++y) {
        for (auto x = 0U; x < splice.width(); ++x) {

            unsigned textonIndex = 0;
            unsigned texton = 0;
            for (auto [x_off, y_off] : neighs) {
                y_off += y;
                x_off += x;

                if (splice.isCoordsValid(y_off, x_off) && splice[y_off][x_off] >= splice[y][x])
                    texton += (1 << textonIndex);

                textonIndex += 1;
            }

            resFeatures[sliceIndex][texton] += 1;
        }
    }
}
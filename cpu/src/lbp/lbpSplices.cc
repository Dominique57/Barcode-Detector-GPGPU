#include "lbpSplices.hh"

LbpSplices::LbpSplices(Matrix &image, unsigned int slicesSize)
        : image_(image), slicesSize_(slicesSize)
{}

void LbpSplices::addLocalPattern(Matrix &resFeatures, const std::vector<std::pair<int, int>> &neighs) {
    unsigned sliceIndex = 0;
    for (auto y = 0U; y + slicesSize_ < image_.height(); y += slicesSize_) {
        for (auto x = 0U; x + slicesSize_ < image_.width(); x += slicesSize_) {
            auto slice = Splice(image_, x, y, slicesSize_, slicesSize_);
            addSliceTexton(resFeatures, neighs, slice, sliceIndex);
            sliceIndex += 1;
        }
    }
}

void LbpSplices::addSliceTexton(Matrix &resFeatures, const std::vector<std::pair<int, int>> &neighs,
                                const Splice &splice, unsigned sliceIndex) {
    for (auto y = 0U; y < splice.height(); ++y) {
        for (auto x = 0U; x < splice.width(); ++x) {

            unsigned textonIndex = 0;
            unsigned texton = 0;
            for (auto [y_off, x_off] : neighs) {
                y_off += y;
                x_off += x;

                if (splice.isCoordsValid(y_off, x_off) && splice[y_off][x_off] >= splice[y][x])
                    texton += (1 << textonIndex);

                textonIndex += 1;
            }

            resFeatures[sliceIndex][y * splice.width() + x] = texton;
        }
    }
}
#include "lbpCpu.hh"

LbpCpu::LbpCpu(unsigned int width, unsigned int height)
        : LbpAlgorithm(width, height)
{}

void LbpCpu::addLocalPatterns(Matrix<> &image) {
    unsigned sliceIndex = 0;
    for (auto y = 0U; y < image.height(); y += SLICE_SIZE) {
        for (auto x = 0U; x < image.width(); x += SLICE_SIZE) {
            auto slice = Splice(image, x, y, SLICE_SIZE, SLICE_SIZE);
            addSliceTextons(slice, sliceIndex);
            sliceIndex += 1;
        }
    }
}

void LbpCpu::addSliceTextons(const Splice &splice, unsigned sliceIndex) {
    // Neighbors offset
    static int pixelsNeighs[8][2]{
            {-1, -1}, {0,  -1}, {1,  -1},
            {-1, 0}, {1,  0},
            {-1, 1}, {0,  1}, {1,  1},
    };

    for (auto y = 0U; y < splice.height(); ++y) {
        for (auto x = 0U; x < splice.width(); ++x) {

            unsigned texton = 0;
            for (auto i = 0U; i < 8U; ++i) {
                int x_off = static_cast<int>(x) + pixelsNeighs[i][0];
                int y_off = static_cast<int>(y) + pixelsNeighs[i][1];

                if (splice.isCoordsValid(y_off, x_off) && splice[y_off][x_off] >= splice[y][x])
                    texton += (1 << i);
            }

            features_[sliceIndex][texton] += 1;
        }
    }
}

void LbpCpu::run(Matrix<> &grayImage) {
    LbpAlgorithm::run(grayImage);
    addLocalPatterns(grayImage);
}

Matrix<> &LbpCpu::getFeatures() {
    return features_;
}

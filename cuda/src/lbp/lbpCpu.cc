#include "lbpCpu.hh"

LbpCpu::LbpCpu(unsigned int width, unsigned int height)
        : LbpAlgorithm(width, height)
{}

void LbpCpu::addLocalPatterns(cv::Mat_<uchar> &image) {
    unsigned sliceIndex = 0;
    for (auto y = 0; y < image.rows; y += SLICE_SIZE) {
        for (auto x = 0; x < image.cols; x += SLICE_SIZE) {
            addSliceTextons(image, x, y, sliceIndex);
            sliceIndex += 1;
        }
    }
}

void LbpCpu::addSliceTextons(const cv::Mat_<uchar> &image, int x_s, int y_s, unsigned sliceIndex) {
    // Neighbors offset
    static int pixelsNeighs[8][2]{
        {-1, -1}, {0,  -1}, {1,  -1},
        {-1, 0}, {1,  0},
        {-1, 1}, {0,  1}, {1,  1},
    };

    for (auto y = y_s; y < y_s + SLICE_SIZE; ++y) {
        for (auto x = x_s; x < x_s + SLICE_SIZE; ++x) {

            unsigned texton = 0;
            for (auto i = 0U; i < 8U; ++i) {
                int x_off = x + pixelsNeighs[i][0];
                int y_off = y + pixelsNeighs[i][1];

                // invalid coordinates (outside splice)
                if (x_off < x_s || x_off >= x_s + SLICE_SIZE || y_off < y_s || y_off >= y_s + SLICE_SIZE)
                    continue;
                if (image[y_off][x_off] >= image[y][x])
                    texton += (1 << i);
            }

            features_[sliceIndex][texton] += 1;
        }
    }
}

void LbpCpu::run(cv::Mat_<uchar> &grayImage) {
    LbpAlgorithm::run(grayImage);
    addLocalPatterns(grayImage);
}
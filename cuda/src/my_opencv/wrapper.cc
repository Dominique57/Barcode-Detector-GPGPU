#include "wrapper.hh"

namespace my_cv {
    static inline bool isCorrectClass(unsigned char class_) {
        return class_ == 1;
    }
    cv::Mat_<uchar> rebuildImageFromVector(
        const std::vector<uchar> &labels, unsigned width) {
        auto resMat = cv::Mat_<uchar>(labels.size() / width, width);

        for (auto y = 0; y < resMat.rows; y++) {
            for (auto x = 0; x < resMat.cols; ++x) {
                unsigned char class_ = labels[y * resMat.cols + x];
                resMat[y][x] = (isCorrectClass(class_))? 255: 0;
            }
        }

        return resMat;
    }
}
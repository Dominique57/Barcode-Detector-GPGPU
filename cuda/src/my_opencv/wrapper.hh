#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace my_cv {
    cv::Mat_<uchar> rebuildImageFromVector(const std::vector<uchar> &labels,
                                           unsigned width);
}

cv::Rect get_position_barcode(cv::Mat image);

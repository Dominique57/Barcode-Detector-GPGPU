#include <chrono>
#include <image/matrix.hh>
#include <lbp/lbpCpu.hh>
#include <lbp/lbpGpu.hh>
#include <iostream>
#include <kmeans/kmeansTransformGpu.hh>
#include <kmeans/KmeansTransform.hh>
#include "entryPoint.hh"

#include <opencv2/highgui.hpp>

void executeAlgorithm(const std::string &path) {
    auto image = Matrix<>(4032U, 3024U);
    Matrix<>::readMatrix(path, image);

    // lbp on cpu
    auto lbpCpu = LbpCpu(image.width(), image.height());
    auto kmeansCpu = KmeansTransform("kmeans.database", 16, 256);
    auto labelsCpu = Matrix<unsigned char>(1, lbpCpu.numberOfPatches());

    std::cout << "Running CPU (1core|1thread)" << std::endl;
    auto start = std::chrono::system_clock::now();

    lbpCpu.run(image);
    kmeansCpu.transform(lbpCpu.getFeatures(), labelsCpu);

    auto end = std::chrono::system_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << elapsedMs.count() << "ms " << elapsedS.count() << "seconds" << std::endl;


    // lbp on Gpu
    auto lbpGpu = LbpGpu(image.width(), image.height());
    auto kmeanGpu = KmeansTransformGpu("kmeans.database", 16, 256);
    auto labelsGpu = Matrix<unsigned char>(1, lbpGpu.numberOfPatches());

    std::cout << "Running GPU (1050ti)" << std::endl;
    auto start2 = std::chrono::system_clock::now();

    lbpGpu.run(image);
    kmeanGpu.transform(lbpGpu.getCudaFeatures(), labelsGpu);

    auto end2 = std::chrono::system_clock::now();
    auto elapsedMs2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    auto elapsedS2 = std::chrono::duration_cast<std::chrono::seconds>(end2 - start2);
    std::cout << elapsedMs2.count() << "ms " << elapsedS2.count() << "seconds" << std::endl;

    // Check features
    std::cout << "Checking if cpu and gpu histogram matrix are the same" << std::endl;
    auto& cpuFeatures = lbpCpu.getFeatures();
    auto& gpuFeatures = lbpGpu.getFeatures();
    for (auto y = 0U; y < cpuFeatures.height(); ++y) {
        for (auto x = 0U; x < cpuFeatures.width(); ++x) {
            if (cpuFeatures[y][x] != gpuFeatures[y][x]) {
                std::cerr << "y:" << y << " x:" << x << " => " << cpuFeatures[y][x] << " <> " << gpuFeatures[y][x] << std::endl;
                throw std::logic_error("Program failed: histogram matrix's are different !");
            }
        }
    }

    // Check labels
    std::cout << "Checking if cpu and gpu predicted labels are the same" << std::endl;
    for (auto i = 0U; i < labelsCpu.height(); ++i) {
        if (labelsCpu[i][0] != labelsGpu[i][0]) {
            std::cerr << "i:" << i << " => " << (int)labelsCpu[i][0] << " <> " << (int)labelsGpu[i][0] << std::endl;
            throw std::logic_error("Program failed: predicted labels are different!");
        }
    }

    // Show image
    std::cout << "Showing predicted matrix as an image" << std::endl;
    {
        auto mat = cv::Mat(labelsGpu.height() / (image.width() / 16), image.width() / 16, CV_8UC3, cv::Scalar(0, 0, 0));
        cv::Vec3b random_lut[16] = {
                {84,  0,   255},
                {255, 0,   23},
                {0,   116, 255},
                {255, 100, 0},
                {184, 0,   255},
                {255, 200, 0},
                {255, 0,   124},
                {0,   15,  255},
                {255, 0,   0},
                {108, 255, 0},
                {0,   255, 192},
                {0,   255, 92},
                {255, 0,   224},
                {7,   255, 0},
                {208, 255, 0},
                {0,   216, 255}
        };
        for (auto y = 0; y < mat.rows; y++) {
            for (auto x = 0; x < mat.cols; ++x) {
                cv::Vec3b color = random_lut[labelsGpu[y * mat.cols + x][0]];
                mat.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);
            }
        }
        cv::imshow("test", mat * 16);
        cv::waitKey(0);
    }

    // Show specific class
    {
        auto mat = cv::Mat(labelsGpu.height() / (image.width() / 16), image.width() / 16, CV_8U);
        for (auto y = 0; y < mat.rows; y++) {
            for (auto x = 0; x < mat.cols; ++x) {
                if (labelsGpu[y * mat.cols + x][0] == 15) {
                    mat.at<unsigned char>(y, x) = 255;
                } else {
                    mat.at<unsigned char>(y, x) = 0;
                }
            }
        }
        cv::imshow("test", mat * 16);
        cv::waitKey(0);
    }
}
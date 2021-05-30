#include <chrono>
#include <image/matrix.hh>
#include <lbp/lbpCpu.hh>
#include <lbp/lbpGpu.hh>
#include <iostream>
#include "entryPoint.hh"

void executeAlgorithm(const std::string &path) {
    auto image = Matrix<>(4032U, 3024U);
    Matrix<>::readMatrix(path, image);

    // Execute lbp_
    auto lbpCpu = LbpCpu(image.width(), image.height());
    std::cout << "Running CPU (1core|1thread)" << std::endl;
    auto start = std::chrono::system_clock::now();

    lbpCpu.run(image);

    auto end = std::chrono::system_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << elapsedMs.count() << "ms " << elapsedS.count() << "seconds" << std::endl;



    auto lbpGpu = LbpGpu(image.width(), image.height());
    std::cout << "Running GPU (1050ti)" << std::endl;
    auto start2 = std::chrono::system_clock::now();

    lbpGpu.run(image);
    lbpGpu.getFeatures();

    auto end2 = std::chrono::system_clock::now();
    auto elapsedMs2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    auto elapsedS2 = std::chrono::duration_cast<std::chrono::seconds>(end2 - start2);
    std::cout << elapsedMs2.count() << "ms " << elapsedS2.count() << "seconds" << std::endl;

    // Check
    auto& cpuFeatures = lbpCpu.getFeatures();
    auto& gpuFeatures = lbpGpu.getFeatures();
    for (auto y = 0U; y < cpuFeatures.height(); ++y) {
        for (auto x = 0U; x < cpuFeatures.width(); ++x) {
            if (cpuFeatures[y][x] != gpuFeatures[y][x]) {
                std::cerr << "y:" << y << " x:" << x << " => " << cpuFeatures[y][x] << " <> " << gpuFeatures[y][x] << std::endl;
                throw std::logic_error("Program failed!");
            }
        }
    }

//     Execute kmeans
//    auto labels = Matrix<>(1, lbp.patchCount());
//    KmeansTransform kmeans("kmeans.database", 16, lbp.textonSize());
}
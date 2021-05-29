#include <chrono>
#include <image/matrix.hh>
#include <kmeans/KmeansTransform.hh>
#include <lbp/lbp.hh>
#include "entryPoint.hh"

void executeAlgorithm(const std::string &path) {
    auto image = Matrix<>(4032U, 3024U);
    Matrix<>::readMatrix(path, image);

    // Execute lbp
    auto lbp = Lbp(image);
    std::cout << "Running CPU (1core|1thread)" << std::endl;
    auto start = std::chrono::system_clock::now();
    auto features = lbp.run(false);
    auto end = std::chrono::system_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto elapsedS = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    std::cout << elapsedMs.count() << "ms " << elapsedS.count() << "seconds" << std::endl;

    auto lbp2 = Lbp(image);
    std::cout << "Running GPU (1050ti)" << std::endl;
    auto start2 = std::chrono::system_clock::now();
    auto features2 = lbp2.run(true);
    auto end2 = std::chrono::system_clock::now();
    auto elapsedMs2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    auto elapsedS2 = std::chrono::duration_cast<std::chrono::seconds>(end2 - start2);
    std::cout << elapsedMs2.count() << "ms " << elapsedS2.count() << "seconds" << std::endl;

    // Check
    for (auto y = 0U; y < features.height(); ++y) {
        for (auto x = 0U; x < features.width(); ++x) {
            if (features[y][x] != features2[y][x]) {
                std::cerr << "y:" << y << " x:" << x << " => " << features[y][x] << " <> " << features2[y][x] << std::endl;
                throw std::logic_error("Program failed!");
            }
        }
    }


    // Execute kmeans
    auto labels = Matrix<>(1, lbp.patchCount());
    KmeansTransform kmeans("kmeans.database", 16, lbp.textonSize());

}

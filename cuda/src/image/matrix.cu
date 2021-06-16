#include <stdexcept>
#include <iostream>
#include "matrix.hh"

template<>
CUDA_HOST void Matrix<float>::readMatrix(const std::string &path, Matrix<float> &matrix) {
    auto fin = std::ifstream(path);
    float current;
    for (auto y = 0U; y < matrix.height_; ++y) {
        for (auto x = 0U; x < matrix.width(); ++x) {
            fin >> current;
            if (fin.fail()) {
                std::stringstream ss{};
                ss << "Error during image read, could not read! (y:" << y << ";x:" << x << ")\n";
                throw std::invalid_argument(ss.str());
            }
            matrix[y][x] = current;
        }
        if (fin.peek() != '\n') {
            throw std::invalid_argument(std::string("Error during image read, eol not reached !"));
        }
    }
    std::string next;
    fin >> next;
    if (!next.empty() && !fin.eof()) {
        throw std::invalid_argument(std::string("Error during image read, eof not reached: ") + next);
    }
}
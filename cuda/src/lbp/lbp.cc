#include "lbp.hh"

Lbp::Lbp(Matrix<> &matrix)
    : matrix_(matrix)
{
    if (matrix_.width() % slicesSize != 0 || matrix_.height() % slicesSize != 0)
        throw std::invalid_argument("Matrix image input dimensions are not slices divisible !");
}

Matrix<> Lbp::run(bool gpu) {
    if (pixelsNeighs.size() > 32)
        throw std::invalid_argument("Cannot represent a texton bigger than 32 bit !");
    auto resFeatures = Matrix<>(textonSize(),patchCount());

    auto splices = LbpSplices(matrix_, slicesSize);
    if (gpu) {
        splices.addLocalPatternsGpu(resFeatures, pixelsNeighs.size());
    } else {
        splices.addLocalPatterns(resFeatures, pixelsNeighs);
    }

    return resFeatures;
    // Normalize histograms (opt)
}

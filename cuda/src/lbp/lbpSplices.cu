#include <iostream>
#include "lbpSplices.hh"

LbpSplices::LbpSplices(Matrix<> &image, unsigned int slicesSize)
        : image_(image), slicesSize_(slicesSize)
{}

void LbpSplices::addLocalPatterns(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs) {
    unsigned sliceIndex = 0;
    for (auto y = 0U; y < image_.height(); y += slicesSize_) {
        for (auto x = 0U; x < image_.width(); x += slicesSize_) {
            auto slice = Splice(image_, x, y, slicesSize_, slicesSize_);
            addSliceTextons(resFeatures, neighs, slice, sliceIndex);
            sliceIndex += 1;
        }
    }
}

void LbpSplices::addSliceTextons(Matrix<> &resFeatures, const std::vector<std::pair<int, int>> &neighs,
                                 const Splice &splice, unsigned sliceIndex) {
    for (auto y = 0U; y < splice.height(); ++y) {
        for (auto x = 0U; x < splice.width(); ++x) {

            unsigned textonIndex = 0;
            unsigned texton = 0;
            for (auto [x_off, y_off] : neighs) {
                y_off += y;
                x_off += x;

                if (splice.isCoordsValid(y_off, x_off) && splice[y_off][x_off] >= splice[y][x])
                    texton += (1 << textonIndex);

                textonIndex += 1;
            }

            resFeatures[sliceIndex][texton] += 1;
        }
    }
}

CUDA_GLOBAL void executeAddLocalPatternsGpu(Matrix<> resFeatures, Matrix<> image) {
    unsigned sliceIndex = blockIdx.y * gridDim.x + blockIdx.x;
    // Slice x/y pixel start
    unsigned y_start = blockIdx.y * 16;
    unsigned x_start = blockIdx.x * 16;
    auto slice = Splice(image, x_start, y_start, 16, 16);

    // Current pixel in splice
    auto x = threadIdx.x;
    auto y = threadIdx.y;

    // Neighbors offset
    CUDA_DEV static int pixelsNeighs[8][2] {
            {-1, -1}, {0, -1}, {1, -1},
            {-1, 0},  {1, 0},
            {-1, 1}, {0, 1}, {1, 1},
    };

    // Compute texton reading neighbors
    unsigned texton = 0;
    for (auto i = 0U; i < 8U; ++i) {
        int x_off = static_cast<int>(x) + pixelsNeighs[i][0];
        int y_off = static_cast<int>(y) + pixelsNeighs[i][1];

        if (slice.isCoordsValid(y_off, x_off) && slice[y_off][x_off] >= slice[y][x])
            texton += (1 << i);
    }

    // Increment texton presence
    __syncthreads();
    atomicAdd(resFeatures[sliceIndex] + texton, 1);
}

void LbpSplices::addLocalPatternsGpu(Matrix<> &resFeatures, unsigned neighCount) {
    // Allocate and copy image to device
    float *imgPtr;
    unsigned imgSize = image_.height() * image_.width() * sizeof(Matrix<>::data_t);
    cudaMalloc(&imgPtr, imgSize);
    cudaMemcpy(imgPtr, image_[0], imgSize, cudaMemcpyHostToDevice);
    auto cudaImage = Matrix<>(image_.width(), image_.height(), imgPtr);

    // Kernel dimensions
    auto dimGrid = dim3(image_.width() / 16, image_.height() / 16);
    auto dimBlock = dim3(16, 16);

    // Allocate result buffer
    float *resultBuffer;
    unsigned featureSize = (1 << neighCount) * (dimGrid.x * dimGrid.y) * sizeof(Matrix<>::data_t);
    cudaMalloc(&resultBuffer, featureSize);
    cudaMemset(resultBuffer, 0, featureSize);
    auto cudaFeatures = Matrix<>(1 << neighCount, (dimGrid.x * dimGrid.y), resultBuffer);

    // Execute kernel
    executeAddLocalPatternsGpu<<<dimGrid, dimBlock>>>(cudaFeatures, cudaImage);

    // Copy back result
    cudaMemcpy(resFeatures[0], resultBuffer, featureSize, cudaMemcpyDeviceToHost);

    // Free ressources
    cudaFree(imgPtr);
    cudaFree(resultBuffer);
}
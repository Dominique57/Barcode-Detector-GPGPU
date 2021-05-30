#include "lbpGpu.hh"

LbpGpu::LbpGpu(unsigned int width, unsigned int height)
        : LbpAlgorithm(width, height),
          cudaImage_(width, height, nullptr),
          cudaFeatures_(1 << NEIGHS_COUNT,
                        (width / SLICE_SIZE) * (height / SLICE_SIZE),
                        nullptr) {
    unsigned imgSize = height * width * sizeof(Matrix<>::data_t);
    cudaMalloc(&(cudaImage_.getData()), imgSize);
    unsigned featureSize = cudaFeatures_.height() * cudaFeatures_.width() * sizeof(Matrix<>::data_t);
    cudaMalloc(&(cudaFeatures_.getData()), featureSize);
}

LbpGpu::~LbpGpu() {
    cudaFree(cudaImage_.getData());
    cudaFree(cudaFeatures_.getData());
}

static CUDA_GLOBAL void executeAddLocalPatternsGpu(Matrix<> resFeatures, Matrix<> image) {
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


void LbpGpu::run(Matrix<> &grayImage) {
    LbpAlgorithm::run(grayImage);
    // Create kernel dimensions
    auto dimGrid = dim3(width_ / SLICE_SIZE, height_ / SLICE_SIZE);
    auto dimBlock = dim3(SLICE_SIZE, SLICE_SIZE);

    // Copy image
    unsigned imgSize = height_ * width_ * sizeof(Matrix<>::data_t);
    cudaMemcpy(cudaImage_.getData(), grayImage.getData(), imgSize, cudaMemcpyHostToDevice);

    // Reset features histogram
    unsigned featureSize = (1 << NEIGHS_COUNT) * (dimGrid.x * dimGrid.y) * sizeof(Matrix<>::data_t);
    cudaMemset(cudaFeatures_.getData(), 0, featureSize);

    // Execute kernel
    executeAddLocalPatternsGpu<<<dimGrid, dimBlock>>>(cudaFeatures_, cudaImage_);
}

Matrix<> &LbpGpu::getFeatures() {
    unsigned featureSize = (1 << NEIGHS_COUNT) * (width_ / SLICE_SIZE * height_ / SLICE_SIZE) * sizeof(Matrix<>::data_t);
    cudaMemcpy(features_.getData(), cudaFeatures_.getData(), featureSize, cudaMemcpyDeviceToHost);
    return features_;
}

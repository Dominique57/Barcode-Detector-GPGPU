#include "lbpGpu.hh"

LbpGpu::LbpGpu(unsigned int width, unsigned int height)
        : LbpAlgorithm(width, height),
          cudaImage_(width, height, nullptr),
          cudaFeatures_(1 << NEIGHS_COUNT,
                        (width / SLICE_SIZE) * (height / SLICE_SIZE),
                        nullptr) {
    unsigned imgLineSize = cudaImage_.width() * sizeof(uchar);
    cudaMallocPitch(&(cudaImage_.getData()), &(cudaImage_.getStride()),
                    imgLineSize, cudaImage_.height());
    unsigned featureLineSize = cudaFeatures_.width() * sizeof(float);
    cudaMallocPitch(&(cudaFeatures_.getData()), &(cudaFeatures_.getStride()),
                    featureLineSize, cudaFeatures_.height());
}

LbpGpu::~LbpGpu() {
    cudaFree(cudaImage_.getData());
    cudaFree(cudaFeatures_.getData());
}

static CUDA_GLOBAL void executeAddLocalPatternsGpu(Matrix<float> resFeatures, Matrix<uchar> image) {
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

void LbpGpu::run(cv::Mat_<uchar> &grayImage) {
    LbpAlgorithm::run(grayImage);

    // Kernel dimensions (width * height) / (slice * slice)
    auto dimGrid = dim3(width_ / SLICE_SIZE, height_ / SLICE_SIZE);
    auto dimBlock = dim3(SLICE_SIZE, SLICE_SIZE);

    // Copy image
    cudaMemcpy2D(
        cudaImage_.getData(), cudaImage_.getStride(), grayImage.data, grayImage.step.p[0],
        cudaImage_.width() * sizeof(uchar), cudaImage_.height(), cudaMemcpyHostToDevice
    );
    // Reset histogram memory
    cudaMemset2D(cudaFeatures_.getData(), cudaFeatures_.getStride(), 0,
                 cudaFeatures_.width() * sizeof(float), cudaFeatures_.height());

    executeAddLocalPatternsGpu<<<dimGrid, dimBlock>>>(cudaFeatures_, cudaImage_);
}

cv::Mat_<float> &LbpGpu::getFeatures() {
    cudaMemcpy2D(
        features_.data, features_.step.p[0], // src
        cudaFeatures_.getData(), cudaFeatures_.getStride(), // dst
        cudaFeatures_.width() * sizeof(float ), cudaFeatures_.height(), // width(bytes), lines
        cudaMemcpyDeviceToHost
    );
    return features_;
}
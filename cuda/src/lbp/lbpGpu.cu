#include "lbpGpu.hh"

LbpGpu::LbpGpu(unsigned int width, unsigned int height)
    : width_(width), height_(height),
      features_((width / SLICE_SIZE) * (height / SLICE_SIZE),1 << NEIGHS_COUNT, 0.),
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

static CUDA_GLOBAL void executeAddLocalPatternsGpu(Matrix<uchar> resFeatures, Matrix<uchar> image) {
    unsigned sliceIndex = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned tid = threadIdx.y * blockDim.y + threadIdx.x;
    // Slice x/y pixel start
    unsigned y_start = blockIdx.y * 16;
    unsigned x_start = blockIdx.x * 16;
    auto slice = Splice(image, x_start, y_start, 16, 16);

    // Current pixel in splice
    auto x = threadIdx.x;
    auto y = threadIdx.y;

    __shared__ uchar pixels[16][16];
    pixels[y][x] = slice[y][x];
    __shared__ unsigned histo[256];
    histo[tid] = 0;
    __syncthreads();
    /*
    __shared__ uchar histo[256];
    histo[tid] = 0;
    __syncthreads();
    */

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

        // if (slice.isCoordsValid(y_off, x_off) && slice[y_off][x_off] >= slice[y][x])
        if (slice.isCoordsValid(y_off, x_off) && pixels[y_off][x_off] >= pixels[y][x])
            texton += (1 << i);
    }

    // Increment texton presence
    /*
    unsigned elt = 1;
    atomicAdd((unsigned*)(histo + (texton - texton % 4)), elt << (8 * (texton % 4)));
    resFeatures[sliceIndex][tid] = (uchar)histo[tid];
    */
    atomicAdd(histo + texton, 1);
    __syncthreads();
    resFeatures[sliceIndex][tid] = (uchar)histo[tid];
}

void LbpGpu::run(cv::Mat_<uchar> &grayImage) {
    if ((unsigned)grayImage.cols != width_ || (unsigned)grayImage.rows != height_)
        throw std::invalid_argument("Image has incorrect format !");

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
                 cudaFeatures_.width() * sizeof(uchar), cudaFeatures_.height());

    executeAddLocalPatternsGpu<<<dimGrid, dimBlock>>>(cudaFeatures_, cudaImage_);
}

cv::Mat_<uchar> &LbpGpu::getFeatures() {
    cudaMemcpy2D(
        features_.data, features_.step.p[0], // src
        cudaFeatures_.getData(), cudaFeatures_.getStride(), // dst
        cudaFeatures_.width() * sizeof(uchar), cudaFeatures_.height(), // width(bytes), lines
        cudaMemcpyDeviceToHost
    );
    return features_;
}
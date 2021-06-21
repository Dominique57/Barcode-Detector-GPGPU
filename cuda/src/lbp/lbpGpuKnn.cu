#include "lbpGpuKnn.hh"

LbpGpuKnn::LbpGpuKnn(unsigned int width, unsigned int height, const std::string &path)
        : width_(width), height_(height),
          cudaImage_(width, height, nullptr),
          cudaCentroids_(256, 16, nullptr) {
    unsigned imgLineSize = cudaImage_.width() * sizeof(uchar);
    cudaMallocPitch(&(cudaImage_.getData()), &(cudaImage_.getStride()),
                    imgLineSize, cudaImage_.height());

    Matrix<float> centroids(256, 16);
    Matrix<>::readMatrix(path, centroids);
    unsigned centroidSize = 16 * 256 * sizeof(float);
    cudaMalloc(&(cudaCentroids_.getData()), centroidSize);
    cudaMemcpy(cudaCentroids_.getData(), centroids.getData(), centroidSize, cudaMemcpyHostToDevice);
}

LbpGpuKnn::~LbpGpuKnn() {
    cudaFree(cudaImage_.getData());
}

CUDA_GLOBAL void executeLbpKnn(Matrix<uchar> image, Matrix<float> centroids, Matrix<uchar> labels) {
    unsigned sliceIndex = blockIdx.y * gridDim.x + blockIdx.x;
    unsigned tid = threadIdx.y * blockDim.y + threadIdx.x;
    __shared__ unsigned histo[256];
    histo[tid] = 0;
    __shared__ float sum[16];
    __shared__ float bufSum[256];
    if (tid < 16) sum[tid] = 0;
    __shared__ uchar min[8];

    // Neighbors offset
    CUDA_DEV static int pixelsNeighs[8][2] {
            {-1, -1}, {0, -1}, {1, -1},
            {-1, 0},  {1, 0},
            {-1, 1}, {0, 1}, {1, 1},
    };

    { // fill histogram of local feature
        // Slice x/y pixel start
        unsigned y_start = blockIdx.y * 16;
        unsigned x_start = blockIdx.x * 16;
        auto slice = Splice(image, x_start, y_start, 16, 16);

        // Current pixel in splice
        auto x = threadIdx.x;
        auto y = threadIdx.y;
        // Compute texton reading neighbors
        unsigned texton = 0;
        for (auto i = 0U; i < 8U; ++i) {
            int x_off = static_cast<int>(x) + pixelsNeighs[i][0];
            int y_off = static_cast<int>(y) + pixelsNeighs[i][1];

            if (slice.isCoordsValid(y_off, x_off) && slice[y_off][x_off] >= slice[y][x])
                texton += (1 << i);
        }

        // Increment texton presence
        atomicAdd(histo + texton, 1);
    }
    __syncthreads();
    { // store distance in shared memory
        for (auto centroidId = 0U; centroidId < centroids.height(); ++centroidId) {
            float centroidFeature = centroids[centroidId][tid];
            float diff = histo[tid] - centroidFeature;
            // atomicAdd(sum + centroidId, diff * diff);
            bufSum[tid] = diff * diff;
            for (unsigned size = 256 / 2; size > 0; size /= 2) {
                __syncthreads();
                if (tid < size)
                    bufSum[tid] += bufSum[tid+size];
            }
            if (tid == 0)
                sum[centroidId] = bufSum[0];
        }
        __syncthreads();
        if (tid < 16)
            sum[tid] = sqrtf(sum[tid]);
    }
    __syncthreads();
    { // reduce pattern to get minimum of 16 float distances
        if (tid < 8)
            min[tid] = (sum[tid*2] < sum[tid*2+1])? tid*2 : tid*2+1;
        if (tid < 4)
            min[tid] = (sum[min[tid*2]] < sum[min[tid*2+1]])? min[tid*2] : min[tid*2+1];
        if (tid < 2)
            min[tid] = (sum[min[tid*2]] < sum[min[tid*2+1]])? min[tid*2] : min[tid*2+1];
        if (tid == 0)
        labels[sliceIndex][0] = (sum[min[tid*2]] < sum[min[tid*2+1]])? min[tid*2] : min[tid*2+1];
    }
}

void LbpGpuKnn::run(cv::Mat_<uchar> &grayImage, std::vector<uchar> &labels) {
    // Kernel dimensions (width * height) / (slice * slice)
    auto dimGrid = dim3(width_ / SLICE_SIZE, height_ / SLICE_SIZE);
    auto dimBlock = dim3(SLICE_SIZE, SLICE_SIZE);
    // Check labels has enough reserved data
    if (dimGrid.x * dimGrid.y > labels.size())
        throw std::invalid_argument("Invalid label buffer: to small!");

    // Copy image
    cudaMemcpy2D(
        cudaImage_.getData(), cudaImage_.getStride(), grayImage.data, grayImage.step.p[0],
        cudaImage_.width() * sizeof(uchar), cudaImage_.height(), cudaMemcpyHostToDevice
    );

    // Create cuda label buffer
    Matrix<unsigned char> cudaLabels(1, dimGrid.x * dimGrid.y, nullptr);
    unsigned labelSize = cudaLabels.width() * cudaLabels.height() * sizeof(uchar);
    cudaMalloc(&(cudaLabels.getData()), labelSize);

    executeLbpKnn<<<dimGrid, dimBlock>>>(cudaImage_, cudaCentroids_, cudaLabels);

    // Write data back
    cudaMemcpy(labels.data(), cudaLabels.getData(), labelSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaLabels.getData());
}
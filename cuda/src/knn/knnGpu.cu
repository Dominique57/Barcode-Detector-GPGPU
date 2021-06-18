#include "knnGpu.hh"

KnnGpu::KnnGpu(const std::string &path, unsigned int nbClusters, unsigned int clusterDim)
        : centroids_(clusterDim, nbClusters),
          cudaCentroids_(clusterDim, nbClusters, nullptr),
          nbClusters_(nbClusters),
          clusterDim_(clusterDim) {
    Matrix<>::readMatrix(path, centroids_);
    unsigned centroidSize = nbClusters * clusterDim * sizeof(Matrix<>::data_t);
    cudaMalloc(&(cudaCentroids_.getData()), centroidSize);
    cudaMemcpy(cudaCentroids_.getData(), centroids_.getData(), centroidSize, cudaMemcpyHostToDevice);
}

KnnGpu::~KnnGpu() {
    cudaFree(cudaCentroids_.getData());
}

CUDA_DEV float execComputeDistance(const float* clusterCentroid, unsigned centroidWith, const half* feature) {
    float sum = 0;
    for (auto i = 0U; i < centroidWith; ++i) {
        float sub = __half2float(feature[i]) - clusterCentroid[i];
        sum += sub * sub;
    }

    return sqrtf(sum);
}

CUDA_GLOBAL void execTransform(const Matrix<float> cudaFeatures, Matrix<> cudaCentroids, Matrix<unsigned char> cudaLabels) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cudaFeatures.height())
        return;

    // Copy current features in local memory
    half feature[256];
    for (auto i = 0U; i < 256; ++i)
        feature[i] = __float2half(cudaFeatures[index][i]);

    __shared__ float centroid[256];

    // Get smallest euclidian distance cluster
    float dist = INFINITY;
    unsigned char cluster = 0;
    for (auto j = 0U; j < cudaCentroids.height(); ++j) {
        centroid[threadIdx.x] = cudaCentroids[j][threadIdx.x]; //  coalescing
        // centroid[threadIdx.x] = constCentroids_[j * 256 + threadIdx.x]; //  coalescing
        __syncthreads();
        // float curDist = execComputeDistance(centroids + (j * cudaCentroids.width()), cudaCentroids.width(), cudaFeatures[index]);
        float curDist = execComputeDistance(
            // cudaCentroids.getData() + (j * cudaCentroids.width()),
            centroid,
            cudaCentroids.width(),
            feature
        );
        if (curDist < dist) {
            dist = curDist;
            cluster = j;
        }
    }
    cudaLabels[index][0] = cluster;
}

void KnnGpu::transform(const Matrix<float> &cudaFeatures, std::vector<uchar> &labels) {
    if (cudaFeatures.height() > labels.size())
        throw std::invalid_argument("Invalid label buffer: to small!");

    // Create cuda label buffer
    Matrix<unsigned char> cudaLabels(1, cudaFeatures.height(), nullptr);
    unsigned labelSize = cudaLabels.width() * cudaLabels.height() * sizeof(uchar);
    cudaMalloc(&(cudaLabels.getData()), labelSize);

    // Compute Kernel dimensions
    unsigned blockWidth = 256;
    unsigned gridWidth = cudaFeatures.height() / blockWidth;
    if (gridWidth % blockWidth != 0)
        gridWidth += 1;

    // Execute kernel
    execTransform<<<gridWidth, blockWidth>>>(cudaFeatures, cudaCentroids_, cudaLabels);

    // Copy result in memory
    cudaMemcpy(labels.data(), cudaLabels.getData(), labelSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaLabels.getData());
}
#include "kmeansTransformGpu.hh"

KmeansTransformGpu::KmeansTransformGpu(const std::string &path, unsigned int nbClusters, unsigned int clusterDim)
        : centroids_(clusterDim, nbClusters),
          cudaCentroids_(clusterDim, nbClusters, nullptr),
          nbClusters_(nbClusters),
          clusterDim_(clusterDim) {
    Matrix<>::readMatrix(path, centroids_);
    unsigned centroidSize = nbClusters * clusterDim * sizeof(Matrix<>::data_t);
    cudaMalloc(&(cudaCentroids_.getData()), centroidSize);
    cudaMemcpy(cudaCentroids_.getData(), centroids_.getData(), centroidSize, cudaMemcpyHostToDevice);
}

KmeansTransformGpu::~KmeansTransformGpu() {
    cudaFree(cudaCentroids_.getData());
}

CUDA_DEV float execComputeDistance(const float* clusterCentroid, unsigned centroidWith, const float* feature) {
    float sum = 0;
    for (auto i = 0U; i < centroidWith; ++i) {
        float sub = feature[i] - clusterCentroid[i];
        sum += sub * sub;
    }

    return sqrtf(sum);
}

CUDA_GLOBAL void execTransform(const Matrix<float> cudaFeatures, Matrix<> cudaCentroids, Matrix<unsigned char> cudaLabels) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cudaFeatures.height())
        return;

    // Copy current features in local memory
    float feature[256];
    for (auto i = 0U; i < 256; ++i)
        feature[i] = cudaFeatures[index][i];

    // Copy centroids in shared memory
    extern __shared__ float centroids[];
    for (auto i = threadIdx.x; i < cudaCentroids.width() * cudaCentroids.height(); i += blockDim.x)
        centroids[i] = cudaCentroids.getData()[i];

    // Get smallest euclidian distance cluster
    float dist = INFINITY;
    unsigned char cluster = 0;
    for (auto j = 0U; j < cudaCentroids.height(); ++j) {
        // float curDist = execComputeDistance(centroids + (j * cudaCentroids.width()), cudaCentroids.width(), cudaFeatures[index]);
        float curDist = execComputeDistance(centroids + (j * cudaCentroids.width()), cudaCentroids.width(), feature);
        if (curDist < dist) {
            dist = curDist;
            cluster = j;
        }
    }
    cudaLabels[index][0] = cluster;
}

void KmeansTransformGpu::transform(const Matrix<float> &cudaFeatures, Matrix<unsigned char> &labels) {
    // Create cuda label buffer
    Matrix<unsigned char> cudaLabels(1, cudaFeatures.height(), nullptr);
    unsigned labelSize = cudaLabels.width() * cudaLabels.height() * sizeof(unsigned char);
    cudaMalloc(&(cudaLabels.getData()), labelSize);

    if (cudaFeatures.height() > labels.height())
        throw std::invalid_argument("FIXME: need to resize cuda buffer!");

    // Compute Kernel dimensions
    unsigned blockWidth = 256;
    unsigned gridWidth = cudaFeatures.height() / blockWidth;
    if (gridWidth % blockWidth != 0)
        gridWidth += 1;

    // Execute kernel
    unsigned centroidSize = centroids_.width() * centroids_.height() * sizeof(Matrix<>::data_t);
    execTransform<<<gridWidth, blockWidth, centroidSize>>>(cudaFeatures, cudaCentroids_, cudaLabels);

    // Copy result in memory
    cudaMemcpy(labels.getData(), cudaLabels.getData(), labelSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaLabels.getData());
}

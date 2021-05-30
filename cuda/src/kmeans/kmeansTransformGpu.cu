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

CUDA_DEV float execComputeDistance(Matrix<> &cudaCentroids, unsigned clusterIndex, const float* feature) {
    float sum = 0;
    for (auto i = 0U; i < cudaCentroids.width(); ++i) {
        float sub = feature[i] - cudaCentroids[clusterIndex][i];
        sum += sub * sub;
    }

    return sqrtf(sum);
}

CUDA_GLOBAL void execTransform(const Matrix<float> cudaFeatures, Matrix<> cudaCentroids, Matrix<unsigned char> cudaLabels) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cudaFeatures.height())
        return;

    // Get smallest euclidian distance cluster
    float dist = INFINITY;
    unsigned char cluster = 0;
    for (auto j = 0U; j < cudaCentroids.height(); ++j) {
        float curDist = execComputeDistance(cudaCentroids, j, cudaFeatures[index]);
        if (curDist < dist) {
            dist = curDist;
            cluster = j;
        }
    }
    cudaLabels[index][0] = cluster;
}

void KmeansTransformGpu::transform(const Matrix<float> &cudaFeatures, Matrix<unsigned char> &labels) const {
    // Create cuda label buffer
    Matrix<unsigned char> cudaLabels(1, cudaFeatures.height(), nullptr);
    unsigned labelSize = cudaLabels.width() * cudaLabels.height() * sizeof(unsigned char);
    cudaMalloc(&(cudaLabels.getData()), labelSize);

    // Compute Kernel dimensions
    unsigned blockWidth = 1024;
    unsigned gridWidth = cudaFeatures.height() / 1024;
    if (gridWidth % 1024 != 0)
        gridWidth += 1;

    // Execute kernel
    execTransform<<<gridWidth, blockWidth>>>(cudaFeatures, cudaCentroids_, cudaLabels);

    // Copy result in memory
    cudaMemcpy(labels.getData(), cudaLabels.getData(), labelSize, cudaMemcpyDeviceToHost);
}

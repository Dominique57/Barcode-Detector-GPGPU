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

CUDA_DEV float execComputeDistance(const float* clusterCentroid, unsigned centroidWith, const uchar* feature) {
    float sum = 0;
    for (auto i = 0U; i < centroidWith; ++i) {
        float sub = feature[i] - clusterCentroid[i];
        sum += sub * sub;
    }

    return sqrtf(sum);
}

CUDA_GLOBAL void execTransform(const Matrix<uchar> cudaFeatures, Matrix<float> cudaCentroids, Matrix<unsigned char> cudaLabels) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cudaFeatures.height())
        return;

    // Copy current features in local memory
    uchar feature[256];
    for (auto i = 0U; i < 256; ++i)
        feature[i] = cudaFeatures[index][i];

    __shared__ float centroid[2][256];

    // Get smallest euclidian distance cluster
    float dist = INFINITY;
    unsigned char cluster = 0;
    for (auto j = 0U; j < cudaCentroids.height(); ++j) {
        centroid[j%2][threadIdx.x] = cudaCentroids[j][threadIdx.x]; //  coalescing
        // centroid[threadIdx.x] = constCentroids_[j * 256 + threadIdx.x]; //  coalescing
        __syncthreads();
        // float curDist = execComputeDistance(centroids + (j * cudaCentroids.width()), cudaCentroids.width(), cudaFeatures[index]);
        float curDist = execComputeDistance(
            // cudaCentroids.getData() + (j * cudaCentroids.width()),
            centroid[j%2],
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

void KnnGpu::transform(const Matrix<uchar> &cudaFeatures, std::vector<uchar> &labels) {
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

CUDA_GLOBAL void executeDistanceComputation(const Matrix<uchar> cudaFeatures,
    Matrix<float> cudaCentroids, Matrix<float> cudaDistances) {
    auto bid = blockIdx.x;
    auto tid = threadIdx.x;
    auto featureId = bid / 16;
    auto centroidId = bid % 16;

    __shared__ float distDiff[256];
    distDiff[tid] = cudaFeatures[featureId][tid] - cudaCentroids[centroidId][tid];
    distDiff[tid] = distDiff[tid] * distDiff[tid];
    distDiff[tid+128] = cudaFeatures[featureId][tid+128] - cudaCentroids[centroidId][tid+128];
    distDiff[tid+128] = distDiff[tid+128] * distDiff[tid+128];

    // execute total sum with worker freeing
    for (unsigned size = 256 / 2; size > 0; size /= 2) {
        if (tid >= size)
            return; // free threads (thus hopefully warps)
        __syncthreads();
        distDiff[tid] += distDiff[tid+size];
    }
    if (tid == 0) // store result
        cudaDistances[featureId][centroidId] = sqrtf(distDiff[0]);
}

CUDA_GLOBAL void executeDistanceComputationOther(const Matrix<uchar> cudaFeatures,
                                            Matrix<float> cudaCentroids, Matrix<float> cudaDistances) {
    auto bid = blockIdx.x;
    auto tid = threadIdx.x;
    auto centroidId = tid / 32;
    auto featureId = tid % 32;
    // 32 bit for each feature euclidean distance
    __shared__ half sum[16][256];
    if (tid < 256) {
        float val = cudaFeatures[bid][tid];
        for (auto i = 0; i < 16; ++i)
            sum[i][tid] = __float2half(val);
    }
    __syncthreads();
    for (auto i = featureId; i < 256; i += 32) {
        float diff = __half2float(sum[centroidId][i]) - cudaCentroids[centroidId][i];
        sum[centroidId][i] = __float2half(diff * diff);
    }
    // 128 -> 64 -> 32 // here inner loop doesnt do anything really // -> 16 -> 8 -> 4 -> 2 -> 1
    for (auto middle = 128U; middle > 0; middle/=2) {
        for (auto i = featureId; i < middle; i += 32) {
            sum[centroidId][i] = __float2half(
                __half2float(sum[centroidId][i]) + __half2float(sum[centroidId][i + middle]));
        }
        __syncthreads();
    }
    if (tid < 16)
        cudaDistances[bid][tid] = sqrt(__half2float(sum[tid][0]));
}

CUDA_GLOBAL void executeDistanceMinOutPriv(Matrix<float> distances, Matrix<unsigned char> labels) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= distances.height())
        return;

    auto minValue = distances[index][0];
    auto minLabel = 0;
    for (auto i = 1; i < 16; ++i) {
        auto curValue = distances[index][i];
        if (minValue > curValue) {
            minValue = curValue;
            minLabel = i;
        }
    }

    labels[index][0] = minLabel;
}

void KnnGpu::transformMultiStep(const Matrix<uchar> &cudaFeatures,
                                std::vector<uchar> &labels) {
    if (cudaFeatures.height() > labels.size())
        throw std::invalid_argument("Invalid label buffer: to small!");

    // Create cuda distance buffer
    Matrix<float> cudaDistances(nbClusters_, cudaFeatures.height(), nullptr);
    unsigned distanceLineSize = sizeof(float) * cudaDistances.width();
    cudaMallocPitch(&(cudaDistances.getData()), &(cudaDistances.getStride()),
                    distanceLineSize, cudaDistances.height());
    cudaDistances.getStride() = cudaDistances.getStride() / sizeof(float);

    /*
    // Run
    auto distBlock = clusterDim_ / 2;
    auto distGrid = cudaFeatures.height() * nbClusters_;
    executeDistanceComputation<<<distGrid, distBlock>>>(
        cudaFeatures, cudaCentroids_, cudaDistances);
    gpuErrchk(cudaDeviceSynchronize());
     */
    auto distBlock = 512;
    auto distGrid = cudaFeatures.height();
    executeDistanceComputationOther<<<distGrid, distBlock>>>(
        cudaFeatures, cudaCentroids_, cudaDistances);
    gpuErrchk(cudaDeviceSynchronize());

    // Create cuda label buffer
    Matrix<unsigned char> cudaLabels(1, cudaFeatures.height(), nullptr);
    unsigned labelSize = cudaLabels.width() * cudaLabels.height() * sizeof(uchar);
    cudaMalloc(&(cudaLabels.getData()), labelSize);

    /*
    // Compute Kernel dimensions
    auto minDistBlock = 32;
    auto minDistGrid = cudaLabels.height() / (32 / 8);
    // Reduce to get minLabel
    executeDistanceMin<<<minDistGrid, minDistBlock>>>(cudaDistances, cudaLabels);
    */
    executeDistanceMinOutPriv<<<(cudaFeatures.height() / 256) + 1, 256>>>(cudaDistances, cudaLabels);

    // Copy result in memory
    cudaMemcpy(labels.data(), cudaLabels.getData(), labelSize, cudaMemcpyDeviceToHost);
    cudaFree(cudaDistances.getData());
    cudaFree(cudaLabels.getData());
}

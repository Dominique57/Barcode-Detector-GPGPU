#pragma once

#include <string>
#include <image/matrix.hh>
#include <cuda/mycuda.hh>

class KnnGpu {

public:
    KnnGpu(const std::string &path, unsigned nbClusters, unsigned clusterDim);

    ~KnnGpu();

    KnnGpu(KnnGpu &kmeans) = delete;
    KnnGpu(KnnGpu &&kmeans) = delete;
    void operator=(KnnGpu &kmeans) = delete;

    void transform(const Matrix<uchar> &features, std::vector<uchar> &labels);


private:
    Matrix<float> centroids_;
    Matrix<float> cudaCentroids_;
    unsigned nbClusters_;
    unsigned clusterDim_;
};
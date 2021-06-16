#pragma once

#include <string>
#include <image/matrix.hh>
#include <cuda/mycuda.hh>

class KmeansTransformGpu {

public:
    KmeansTransformGpu(const std::string &path, unsigned nbClusters, unsigned clusterDim);

    ~KmeansTransformGpu();

    KmeansTransformGpu(KmeansTransformGpu &kmeans) = delete;
    KmeansTransformGpu(KmeansTransformGpu &&kmeans) = delete;
    void operator=(KmeansTransformGpu &kmeans) = delete;

    void transform(const Matrix<float> &features, std::vector<uchar> &labels);


private:
    Matrix<float> centroids_;
    Matrix<float> cudaCentroids_;
    unsigned nbClusters_;
    unsigned clusterDim_;
};
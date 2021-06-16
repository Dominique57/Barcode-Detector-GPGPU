#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <image/matrix.hh>
#include "my_opencv/wrapper.hh"

class KnnCpu {
public:
    KnnCpu(const std::string &path, unsigned nbClusters, unsigned clusterDim)
            : centroids_(clusterDim, nbClusters),
              nbClusters_(nbClusters),
              clusterDim_(clusterDim) {
        Matrix<>::readMatrix(path, centroids_);
    }

    KnnCpu(const KnnCpu &kmeans) = delete;
    KnnCpu(const KnnCpu &&kmeans) = delete;
    void operator=(const KnnCpu &kmeans) = delete;

    void transform(const cv::Mat_<float> &features, std::vector<uchar> &labels) const;

protected:
    float computeDistance(unsigned clusterIndex, const float* feature) const;

private:
    Matrix<float> centroids_;
    unsigned nbClusters_;
    unsigned clusterDim_;
};
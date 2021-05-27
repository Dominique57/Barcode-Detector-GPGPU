#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <image/matrix.hh>

class KmeansTransform {
public:
    KmeansTransform(const std::string &path, unsigned nbClusters, unsigned clusterDim)
            : centroids_(clusterDim, nbClusters),
              nbClusters_(nbClusters),
              clusterDim_(clusterDim) {
        loadCentroids(path);
    }

    KmeansTransform(KmeansTransform &kmeans) = delete;
    KmeansTransform(KmeansTransform &&kmeans) = delete;
    void operator=(KmeansTransform &kmeans) = delete;

    void transform(const Matrix<float> &features, Matrix<>& labels) const;

protected:
    void loadCentroids(const std::string &path);
    float computeDistance(unsigned clusterIndex, const float* feature) const;

private:
    Matrix<float> centroids_;
    unsigned nbClusters_;
    unsigned clusterDim_;
};
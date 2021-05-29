#include <iostream>
#include "KmeansTransform.hh"

void KmeansTransform::loadCentroids(const std::string &path) {
    std::ifstream fin(path);

    auto i = 0U;
    for (; i < nbClusters_ && fin.good(); ++i) {
        std::string line;
        std::getline(fin, line);
        std::stringstream lineStream(line);

        // Could make function of this chunk for code cleanness
        auto j = 0U;
        for (; j < clusterDim_ && lineStream.good(); ++j) {
            float centroids_feature;
            fin >> centroids_feature;
            centroids_[i][j] = centroids_feature;
        }
        // Safety checks
        if (j != clusterDim_) {
            if (lineStream.eof())
                throw std::invalid_argument(std::to_string(i) + "th line has not enough scalars");
            throw std::invalid_argument("I/O error !");
        }
    }

    // Safety checks
    if (i != nbClusters_) {
        if (fin.eof())
            throw std::invalid_argument("Number of line is insufficient for kmeans model !");
        throw std::invalid_argument("I/O error !");
    }
}

void KmeansTransform::transform(const Matrix<float> &features, Matrix<> &labels) const {
    if (features.width() != clusterDim_)
        throw std::invalid_argument("Features have incorrect dimension !");
    if (features.height() > labels.height())
        throw std::invalid_argument("Result vector is not compatible with feature count !");

    for (auto i = 0U; i < features.height(); ++i) {
        // Get smallest euclidian distance cluster
        float dist = INFINITY;
        unsigned char cluster = 0;
        for (auto j = 0U; j < nbClusters_; ++j) {
            float curDist = computeDistance(j, features[i]);
            if (curDist < dist) {
                dist = curDist;
                cluster = j;
            }
        }

        labels[i][0] = cluster;
    }
}

float KmeansTransform::computeDistance(unsigned clusterIndex, const float* feature) const {
    float sum = 0;
    for (auto i = 0U; i < clusterDim_; ++i)
        sum += feature[i] * centroids_[clusterIndex][i];

    return sqrtf(sum);
}
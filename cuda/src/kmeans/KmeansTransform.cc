#include <iostream>
#include "KmeansTransform.hh"

void KmeansTransform::transform(const Matrix<> &features, Matrix<unsigned char> &labels) const {
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
    for (auto i = 0U; i < clusterDim_; ++i) {
        float sub = feature[i] - centroids_[clusterIndex][i];
        sum += sub * sub;
    }

    return sqrtf(sum);
}
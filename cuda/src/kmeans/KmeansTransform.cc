#include <iostream>
#include "KmeansTransform.hh"

void KmeansTransform::transform(const cv::Mat_<float> &features, std::vector<uchar> &labels) const {
    if (features.cols != (int)clusterDim_)
        throw std::invalid_argument("Features have incorrect dimension !");
    if (features.rows > (int)labels.size())
        throw std::invalid_argument("Result vector is not compatible with feature count !");

    for (auto i = 0; i < features.rows; ++i) {
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

        labels[i] = cluster;
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
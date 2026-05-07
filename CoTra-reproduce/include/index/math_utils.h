// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "diskann/include/common_includes.h"
#include "diskann/include/utils.h"

namespace math_utils {


struct KmeansCenter
{
    KmeansCenter() = default;

    KmeansCenter(size_t pivo_id, float weight_dist, float dist) : 
        piv_id{pivo_id}, piv_weight_dist{weight_dist}, piv_dist{dist}
    {
    }

    bool operator<(const KmeansCenter &p) const
    {
        return p.piv_weight_dist < piv_weight_dist;
    }

    bool operator>(const KmeansCenter &p) const
    {
        return p.piv_weight_dist > piv_weight_dist;
    }

    size_t piv_id;
    float piv_dist;
    float piv_weight_dist;
};
    

struct ClusterInfo {
    size_t count;         // number of vector in cluster
    float totalDist;      // total distance to the medoid in cluster
    float maxDist;        // max distance the medoid in cluster
};


class ClusterBalancer {
private:
    float m_prevLambda;
    float m_decayRate;
    float m_minLambda;
    float m_maxLambda;

public:
    ClusterBalancer(float init=0.1, float decay=0.98, float min=0.0, float max=1.0)
        : m_prevLambda(init), m_decayRate(decay), m_minLambda(min), m_maxLambda(max) {}

    // Compute lambda based on cluster msg
    float ComputeLambda(const std::vector<ClusterInfo>& clusters) {
        size_t totalPoints = 0;
        float maxVar = 0, maxAvgDist = 0;
        size_t maxClusterIdx = 0;
        
        for (size_t i=0; i<clusters.size(); ++i) {
            totalPoints += clusters[i].count;
            if (clusters[i].count > clusters[maxClusterIdx].count) {
                maxClusterIdx = i;
            }
        }

        const auto& maxCluster = clusters[maxClusterIdx];
        float avgDist = (maxCluster.count == 0) ? 0.0 : maxCluster.totalDist / maxCluster.count ;
        float maxDist = maxCluster.maxDist;
        
        // 2. compute base lambda
        float baseLambda = (totalPoints == 0) ? 0.0 : (maxDist - avgDist) / totalPoints;

        printf("<<>> maxDist: %.6f avgDist: %.6f totalPoints: %llu baseLambda: %.6f\n", 
            maxDist, avgDist, totalPoints, baseLambda);
        
        float stdDev = ComputeStdDev(clusters);
        float stdPenalty = (maxCluster.count == 0) ? 0.0 : stdDev / avgDist;
        
        // float newLambda = baseLambda * (1.0 + stdPenalty);
        float newLambda = baseLambda * (1.0 + stdDev);

        newLambda = std::clamp(newLambda * m_decayRate, m_minLambda, m_maxLambda);

        printf("<<>> stdDev: %.6f newLambda: %.6f\n", 
            stdDev, newLambda);
        
        m_prevLambda = newLambda;
        return newLambda;
    }

private:
    float ComputeStdDev(const std::vector<ClusterInfo>& clusters) {
        float mean = 0, sumSq = 0;
        for (const auto& c : clusters) {
            mean += c.count;
        }
        float total = mean;
        mean /= clusters.size();
        
        for (const auto& c : clusters) {
            sumSq += total ==0 ? 0.0 :((double)c.count/total - mean/total) 
                * ((double)c.count/total - mean/total);
        }
        return std::sqrt(sumSq / clusters.size());
    }
};


class LambdaController {
 private:
  float alpha;       // weight between distance and cluster size (0~1)
  float max_lambda;  // λ upperbound
  float min_lambda;  // λ lowerbound
  float decay_rate;  // λ decay rate
  float cur_lambda;  // current λ

 public:
  LambdaController(
      float a = 0.5, float max = 1.0, float min = 0.01, float dr = 0.95)
      : alpha(a), max_lambda(max), min_lambda(min), decay_rate(dr) {}

  float update(
      const std::vector<size_t> &cluster_sizes, const float *distances,
      size_t num_points) {
    // compute balance ratio
    float balance_ratio = compute_balance_ratio(cluster_sizes);

    // compute distance variance
    float dist_var =
        compute_distance_variance(distances, num_points, cluster_sizes.size());

    // compute new lambda
    float new_lambda = alpha * balance_ratio + (1 - alpha) * sqrt(dist_var);
    new_lambda = std::clamp(new_lambda * decay_rate, min_lambda, max_lambda);
    cur_lambda = new_lambda;
    // printf("new lambda : %.6f\r", new_lambda);
    return new_lambda;
  }

  float get_lambda() { return cur_lambda; }

  // compute balance ratio
  float compute_balance_ratio(const std::vector<size_t> &cluster_sizes) {
    auto [min_it, max_it] =
        minmax_element(cluster_sizes.begin(), cluster_sizes.end());
    return (*max_it) / float(*min_it + 1e-6);
  }

  float compute_distance_variance(
      const float *distances, size_t num_points, size_t num_centers) {
    std::vector<float> avg_dist(num_centers, 0);
    // compute average distance
    for (size_t j = 0; j < num_centers; ++j) {
      for (size_t i = 0; i < num_points; ++i) {
        avg_dist[j] += distances[i * num_centers + j];
      }
      avg_dist[j] /= num_points;
    }
    // compute variance
    float mean =
        accumulate(avg_dist.begin(), avg_dist.end(), 0.0f) / num_centers;
    float var = 0;
    for (auto d : avg_dist) var += (d - mean) * (d - mean);
    return var;
  }
};

float calc_distance(float *vec_1, float *vec_2, size_t dim);

// compute l2-squared norms of data stored in row major num_points * dim,
// needs
// to be pre-allocated
void compute_vecs_l2sq(
    float *vecs_l2sq, float *data, const size_t num_points, const size_t dim);

void rotate_data_randomly(
    float *data, size_t num_points, size_t dim, float *rot_mat, float *&new_mat,
    bool transpose_rot = false);

// calculate closest center to data of num_points * dim (row major)
// centers is num_centers * dim (row major)
// data_l2sq has pre-computed squared norms of data
// centers_l2sq has pre-computed squared norms of centers
// pre-allocated center_index will contain id of k nearest centers
// pre-allocated dist_matrix shound be num_points * num_centers and contain
// squared distances

// Ideally used only by compute_closest_centers
void compute_closest_centers_in_block(
    const float *const data, const size_t num_points, const size_t dim,
    const float *const centers, const size_t num_centers,
    const float *const docs_l2sq, const float *const centers_l2sq,
    uint32_t *center_index, float *const dist_matrix, size_t k = 1);

// Given data in num_points * new_dim row major
// Pivots stored in full_pivot_data as k * new_dim row major
// Calculate the closest pivot for each point and store it in vector
// closest_centers_ivf (which needs to be allocated outside)
// Additionally, if inverted index is not null (and pre-allocated), it will
// return inverted index for each center Additionally, if pts_norms_squared is
// not null, then it will assume that point norms are pre-computed and use
// those
// values

void compute_closest_centers(
    float *data, size_t num_points, size_t dim, float *pivot_data,
    size_t num_centers, size_t k, uint32_t *closest_centers_ivf,
    std::vector<size_t> *inverted_index = NULL,
    float *pts_norms_squared = NULL);

/**
 * Compute the closest center with the balance factor.
 */
void compute_balance_closest_centers_in_block(
    const float *const data, const size_t num_points, const size_t dim,
    const float *const centers, const size_t num_centers,
    const float *const docs_l2sq, const float *const centers_l2sq,
    uint32_t *center_index, float *const dist_matrix, size_t k,
    std::vector<ClusterInfo> &cluster_stats, float lambda);

/**
 * Compute the balanced topk kmeans partition (with replica).
 * lambda: control the balance factor.
 */
void compute_balance_closest_centers(
    float *data, size_t num_points, size_t dim, float *pivot_data,
    size_t num_centers, size_t k, uint32_t *closest_centers_ivf,
    ClusterBalancer &balancer, std::vector<ClusterInfo> &cluster_stats,
    std::vector<size_t> *inverted_index = NULL,
    float *pts_norms_squared = NULL);



void compute_balance_norp_rp_closest_centers_in_block(
    const float *const data, const size_t num_points, const size_t dim,
    const float *const centers, const size_t num_centers,
    const float *const docs_l2sq, const float *const centers_l2sq,
    uint32_t *center_index, float *const dist_matrix, size_t k,
    std::vector<ClusterInfo> &norp_cluster_stats,
    std::vector<ClusterInfo> &rp_cluster_stats, 
    float norp_lambda , float rp_lambda, std::string data_type);

void compute_balance_norp_rp_closest_centers(
    float *data, size_t num_points, size_t dim, float *pivot_data,
    size_t num_centers, size_t k, uint32_t *closest_centers_ivf,
    ClusterBalancer &balancer, std::vector<ClusterInfo> &norp_cluster_stats, 
    std::vector<ClusterInfo> &rp_cluster_stats, std::string data_type,
    float *pts_norms_squared = NULL);

// if to_subtract is 1, will subtract nearest center from each row. Else will
// add. Output will be in data_load iself.
// Nearest centers need to be provided in closst_centers.

void process_residuals(
    float *data_load, size_t num_points, size_t dim, float *cur_pivot_data,
    size_t num_centers, uint32_t *closest_centers, bool to_subtract);

}  // namespace math_utils

namespace kmeans {

// run Lloyds one iteration
// Given data in row major num_points * dim, and centers in row major
// num_centers * dim
// And squared lengths of data points, output the closest center to each data
// point, update centers, and also return inverted index.
// If closest_centers == NULL, will allocate memory and return.
// Similarly, if closest_docs == NULL, will allocate memory and return.

float lloyds_iter(
    float *data, size_t num_points, size_t dim, float *centers,
    size_t num_centers, float *docs_l2sq, std::vector<size_t> *closest_docs,
    uint32_t *&closest_center);

// Run Lloyds until max_reps or stopping criterion
// If you pass NULL for closest_docs and closest_center, it will NOT return
// the results, else it will assume appriate allocation as closest_docs = new
// vector<size_t> [num_centers], and closest_center = new size_t[num_points]
// Final centers are output in centers as row major num_centers * dim
//
float run_lloyds(
    float *data, size_t num_points, size_t dim, float *centers,
    const size_t num_centers, const size_t max_reps,
    std::vector<size_t> *closest_docs, uint32_t *closest_center);

// assumes already memory allocated for pivot_data as new
// float[num_centers*dim] and select randomly num_centers points as pivots
void selecting_pivots(
    float *data, size_t num_points, size_t dim, float *pivot_data,
    size_t num_centers);

void kmeanspp_selecting_pivots(
    float *data, size_t num_points, size_t dim, float *pivot_data,
    size_t num_centers);

}  // namespace kmeans

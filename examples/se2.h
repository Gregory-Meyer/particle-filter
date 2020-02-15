#ifndef SE2_H
#define SE2_H

#include <stddef.h>

#define SE2_STATE_LENGTH ((size_t)3)

struct se2_mean_covariance {
  double mean[SE2_STATE_LENGTH];
  double covariance[SE2_STATE_LENGTH][SE2_STATE_LENGTH];
};

void se2_compute_weighted_mean_and_covariance(
    void *restrict arg, size_t num_particles, size_t state_length,
    const double states[restrict num_particles][SE2_STATE_LENGTH],
    const double weights[restrict num_particles]);

#endif

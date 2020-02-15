#include "se2.h"

#include "util.h"

#include <assert.h>
#include <complex.h>

void se2_compute_weighted_mean_and_covariance(
    void *restrict arg, size_t num_particles, size_t state_length,
    const double states[restrict num_particles][SE2_STATE_LENGTH],
    const double weights[restrict num_particles]) {
  assert(arg);
  assert(num_particles > 0);
  assert(state_length == SE2_STATE_LENGTH);
  assert(states);
  assert(weights);

  struct se2_mean_covariance *const restrict mean_covariance = arg;

  double x_mean_accumulator = 0.0;
  double y_mean_accumulator = 0.0;
  double complex angle_mean_accumulator = 0.0;

  for (size_t i = 0; i < num_particles; ++i) {
    const double this_weight = weights[i];

    x_mean_accumulator += this_weight * states[i][0];
    y_mean_accumulator += this_weight * states[i][1];
    angle_mean_accumulator += this_weight * cexp(I * states[i][2]);
  }

  const double angle_mean = carg(angle_mean_accumulator);

  mean_covariance->mean[0] = x_mean_accumulator;
  mean_covariance->mean[1] = y_mean_accumulator;
  mean_covariance->mean[2] = angle_mean;

  double covariance_accumulator[SE2_STATE_LENGTH][SE2_STATE_LENGTH] = {0};
  double squared_weights_sum_accumulator = 0.0;

  for (size_t particle_index = 0; particle_index < num_particles;
       ++particle_index) {
    const double this_weight = weights[particle_index];
    squared_weights_sum_accumulator += this_weight * this_weight;

    const double offsets[SE2_STATE_LENGTH] = {
        states[particle_index][0] - x_mean_accumulator,
        states[particle_index][1] - y_mean_accumulator,
        wrap_to_pi(states[particle_index][2] - angle_mean),
    };

    for (size_t i = 0; i < SE2_STATE_LENGTH; ++i) {
      const double partial_product = this_weight * offsets[i];

      for (size_t j = 0; j < SE2_STATE_LENGTH; ++j) {
        covariance_accumulator[i][j] += partial_product * offsets[j];
      }
    }
  }

  const double covariance_normalizer = 1.0 - squared_weights_sum_accumulator;

  for (size_t i = 0; i < SE2_STATE_LENGTH; ++i) {
    for (size_t j = 0; j < SE2_STATE_LENGTH; ++j) {
      covariance_accumulator[i][j] /= covariance_normalizer;
    }
  }

  for (size_t i = 0; i < SE2_STATE_LENGTH; ++i) {
    for (size_t j = 0; j < SE2_STATE_LENGTH; ++j) {
      mean_covariance->covariance[i][j] = covariance_accumulator[i][j];
    }
  }
}

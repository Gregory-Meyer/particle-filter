#ifndef V2DP_H
#define V2DP_H

#include "se2.h"

#include <stddef.h>

#define V2DP_ACTION_LENGTH ((size_t)3)
#define V2DP_NOISE_LENGTH ((size_t)3)

struct v2dp_state {
  double sampling_period;
  double alpha[6];
};

int v2dp_model(
    void *restrict arg, size_t action_length,
    const double action[restrict static V2DP_ACTION_LENGTH], size_t num_states,
    size_t state_length,
    const double states[restrict num_states][SE2_STATE_LENGTH],
    size_t noise_length,
    const double noise_vectors[restrict num_states][V2DP_NOISE_LENGTH],
    double next_states[restrict num_states][SE2_STATE_LENGTH]);
int v2dp_noise_covariance(
    void *restrict arg, size_t action_length,
    const double action[restrict static V2DP_ACTION_LENGTH],
    size_t noise_length,
    double process_noise_covariance[restrict static V2DP_NOISE_LENGTH]
                                   [V2DP_NOISE_LENGTH]);

void v2dp_motion_model(
    const struct v2dp_state *restrict state,
    const double action[restrict static V2DP_ACTION_LENGTH],
    const double current_state[restrict static SE2_STATE_LENGTH],
    double next_state[restrict static SE2_STATE_LENGTH]);
void v2dp_noise_variance(
    const struct v2dp_state *restrict state,
    const double action[restrict static V2DP_ACTION_LENGTH],
    double variance[restrict static V2DP_NOISE_LENGTH]);

#endif

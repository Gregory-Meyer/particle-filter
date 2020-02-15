#include "v2dp.h"

#include "util.h"

#include <assert.h>
#include <math.h>

int v2dp_model(
    void *restrict arg, size_t action_length,
    const double action[restrict static V2DP_ACTION_LENGTH], size_t num_states,
    size_t state_length,
    const double states[restrict num_states][SE2_STATE_LENGTH],
    size_t noise_length,
    const double noise_vectors[restrict num_states][V2DP_NOISE_LENGTH],
    double next_states[restrict num_states][SE2_STATE_LENGTH]) {
  assert(arg != NULL);
  assert(action_length == V2DP_ACTION_LENGTH);
  assert(action != NULL);
  assert(num_states > 0);
  assert(state_length == SE2_STATE_LENGTH);
  assert(states != NULL);
  assert(noise_length == V2DP_NOISE_LENGTH);
  assert(noise_vectors != NULL);
  assert(next_states != NULL);

  const struct v2dp_state *const restrict state = arg;

  const double linear_velocity = action[0];  // v
  const double angular_velocity = action[1]; // omega
  const double turn_velocity = action[2];    // gamma

  for (size_t i = 0; i < num_states; ++i) {
    const double noisy_action[V2DP_ACTION_LENGTH] = {
        linear_velocity + noise_vectors[i][0],
        angular_velocity + noise_vectors[i][1],
        turn_velocity + noise_vectors[i][2]};

    v2dp_motion_model(state, states[i], noisy_action, next_states[i]);
  }

  return 0;
}

int v2dp_noise_covariance(
    void *restrict arg, size_t action_length,
    const double action[restrict static V2DP_ACTION_LENGTH],
    size_t noise_length,
    double process_noise_covariance[restrict static V2DP_NOISE_LENGTH]
                                   [V2DP_NOISE_LENGTH]) {
  assert(arg);
  assert(action_length == V2DP_ACTION_LENGTH);
  assert(action);
  assert(noise_length == V2DP_NOISE_LENGTH);
  assert(process_noise_covariance);

  const struct v2dp_state *const restrict state = arg;

  const double linear_velocity = action[0];
  const double angular_velocity = action[1];

  double noise_variance[V2DP_NOISE_LENGTH];
  v2dp_noise_variance(state, action, noise_variance);

  process_noise_covariance[0][0] = noise_variance[0];
  process_noise_covariance[0][1] = 0.0;
  process_noise_covariance[0][2] = 0.0;

  process_noise_covariance[1][0] = 0.0;
  process_noise_covariance[1][1] = noise_variance[1];
  process_noise_covariance[1][2] = 0.0;

  process_noise_covariance[2][0] = 0.0;
  process_noise_covariance[2][1] = 0.0;
  process_noise_covariance[2][2] = noise_variance[2];

  return 0;
}

void v2dp_motion_model(
    const struct v2dp_state *restrict state,
    const double action[restrict static V2DP_ACTION_LENGTH],
    const double current_state[restrict static SE2_STATE_LENGTH],
    double next_state[restrict static SE2_STATE_LENGTH]) {
  assert(state);
  assert(action);
  assert(current_state);
  assert(next_state);
  assert(state->sampling_period > 0.0);

  const double timestep = state->sampling_period;

  const double x_position = current_state[0];          // x
  const double y_position = current_state[1];          // y
  const double heading = wrap_to_pi(current_state[2]); // theta

  const double linear_velocity = action[0];  // v
  const double angular_velocity = action[1]; // omega
  const double turn_velocity = action[2];    // gamma

  const double arc_radius = linear_velocity / angular_velocity;
  const double arc_heading = wrap_to_pi(heading + angular_velocity * timestep);

  next_state[0] =
      x_position - arc_radius * sin(heading) + arc_radius * sin(arc_heading);
  next_state[1] =
      y_position + arc_radius * cos(heading) - arc_radius * cos(arc_heading);
  next_state[2] = wrap_to_pi(heading + angular_velocity * timestep +
                             turn_velocity * timestep);
}

void v2dp_noise_variance(
    const struct v2dp_state *restrict state,
    const double action[restrict static V2DP_ACTION_LENGTH],
    double variance[restrict static V2DP_NOISE_LENGTH]) {
  assert(state);
  assert(action);
  assert(variance);

#ifndef NDEBUG
  for (size_t i = 0; i < sizeof(state->alpha) / sizeof(state->alpha[0]); ++i) {
    assert(state->alpha[i] > 0);
  }
#endif

  const double linear_velocity = action[0];  // v
  const double angular_velocity = action[1]; // omega

  const double squared_linear_velocity = linear_velocity * linear_velocity;
  const double squared_angular_velocity = angular_velocity * angular_velocity;

  const double *const restrict alpha = state->alpha;

  variance[0] =
      alpha[0] * squared_linear_velocity + alpha[1] * angular_velocity;
  variance[1] =
      alpha[2] * squared_linear_velocity + alpha[3] * angular_velocity;
  variance[2] =
      alpha[4] * squared_linear_velocity + alpha[5] * angular_velocity;
}

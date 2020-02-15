#include "l2dm.h"

#include "trig.h"

#include <assert.h>
#include <math.h>

/// 2-D single landmark measurement model probability density function
int l2dm_pdf(void *restrict arg, size_t measurement_length,
             const double measurement[restrict static L2DM_MEASUREMENT_LENGTH],
             size_t num_states, size_t state_length,
             const double states[restrict num_states][SE2_STATE_LENGTH],
             double likelihoods[restrict num_states]) {
  assert(arg);
  assert(measurement_length == L2DM_MEASUREMENT_LENGTH);
  assert(measurement);
  assert(num_states > 0);
  assert(state_length == SE2_STATE_LENGTH);
  assert(states);
  assert(likelihoods);

  const struct l2dm_state *const restrict state = arg;

  const double covariance_determinant =
      state->bearing_variance * state->range_variance;

  const double likelihood_normalizer =
      1.0 / (2.0 * PI * sqrt(covariance_determinant));

  const double bearing_offset_scale =
      state->range_variance / covariance_determinant;
  const double range_offset_scale =
      state->bearing_variance / covariance_determinant;

  const double landmark_x = state->landmark_x;
  const double landmark_y = state->landmark_y;

  const double bearing = wrap_to_pi(measurement[0]);
  const double range = measurement[1];

  for (size_t i = 0; i < num_states; ++i) {
    const double x_position = states[i][0];
    const double y_position = states[i][1];
    const double heading = wrap_to_pi(states[i][2]);

    const double x_offset = landmark_x - x_position;
    const double y_offset = landmark_y - y_position;

    const double expected_bearing =
        wrap_to_pi(atan2(y_offset, x_offset) - heading);

    const double expected_range = hypot(x_offset, y_offset);

    const double bearing_offset = wrap_to_pi(expected_bearing - bearing);
    const double range_offset = expected_range - range;

    const double exponent =
        -0.5 * (bearing_offset_scale * bearing_offset * bearing_offset) +
        (range_offset_scale * range_offset * range_offset);

    const double likelihood = likelihood_normalizer * exp(exponent);
    likelihoods[i] = likelihood;
  }

  return 0;
}

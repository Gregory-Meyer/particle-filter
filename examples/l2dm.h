#ifndef L2DM_H
#define L2DM_H

#include "se2.h"

#define L2DM_MEASUREMENT_LENGTH ((size_t)2)

/// 2-D single landmark measurement model state
struct l2dm_state {
  double bearing_variance;
  double range_variance;
  double landmark_x;
  double landmark_y;
};

/// 2-D single landmark measurement model probability density function
int l2dm_pdf(void *restrict arg, size_t measurement_length,
             const double measurement[restrict static L2DM_MEASUREMENT_LENGTH],
             size_t num_states, size_t state_length,
             const double states[restrict num_states][SE2_STATE_LENGTH],
             double likelihoods[restrict num_states]);

#endif

#include "alloc.h"
#include "l2dm.h"
#include "se2.h"
#include "trig.h"
#include "v2dp.h"

#include <pf.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pcg_variants.h>

#define SQUARE(X) ((X) * (X))
#define RADIANS(X) ((X)*PI / 180.0)

#define BEARING_STDEV RADIANS(5.0)
#define RANGE_STDEV 25.0

struct double_pair {
  double first;
  double second;
};

static uint32_t rng(void *restrict arg);
static struct double_pair standard_normal_random(pcg32_random_t *restrict rng);

int main(int argc, const char *const argv[restrict argc]) {
  if (argc < 2) {
    fputs("error: missing argument NUM_TIMESTEPS\n", stderr);

    return EXIT_FAILURE;
  }

  size_t num_timesteps;

  if (sscanf(argv[1], "%zu", &num_timesteps) != 1) {
    fprintf(stderr,
            "error: couldn't parse '%s' as a positive integer (size_t)\n",
            argv[1]);

    return EXIT_FAILURE;
  }

  struct v2dp_state process_model_state = {
      .sampling_period = 0.01,
      .alpha = {SQUARE(0.00025), SQUARE(0.00005), SQUARE(0.0025),
                SQUARE(0.0005), SQUARE(0.0025), SQUARE(0.0005)}};
  struct l2dm_state measurement_model_state = {
      .bearing_variance = SQUARE(BEARING_STDEV),
      .range_variance = SQUARE(RANGE_STDEV),
      .landmark_x = 0.0,
      .landmark_y = 0.0};

  struct pf_particle_filter filter = {
      .process_model =
          {
              .model_fn = &v2dp_model,
              .noise_covariance_fn = &v2dp_noise_covariance,
              .action_length = (int)V2DP_ACTION_LENGTH,
              .noise_length = (int)V2DP_NOISE_LENGTH,
              .arg = &process_model_state,
          },

      .measurement_model =
          {
              .pdf = &l2dm_pdf,
              .measurement_length = (int)L2DM_MEASUREMENT_LENGTH,
              .arg = &measurement_model_state,
          },

      .state_length = (int)SE2_STATE_LENGTH,
      .num_particles = 512,
  };

  pcg32_random_t pcg_rng = PCG32_INITIALIZER;
  const struct pf_random_number_generator random_number_generator = {
      .rng_fn = &rng,
      .arg = &pcg_rng,
  };

  const struct pf_allocator system_allocator = {
      .alloc = &allocator, .dealloc = &deallocator, .arg = NULL};

  double true_state[SE2_STATE_LENGTH] = {180.0, 50.0, 0.0};
  const double initial_belief_covariance[SE2_STATE_LENGTH][SE2_STATE_LENGTH] = {
      {0.001, 0.0, 0.0},
      {0.0, 0.001, 0.0},
      {0.0, 0.0, 0.001},
  };
  enum pf_status result = pf_new(&filter, true_state, initial_belief_covariance,
                                 &random_number_generator, &system_allocator);

  if (result != PF_OK) {
    fprintf(stderr, "error: pf_new() failed: %s (%d)\n",
            pf_error_description(result).data, (int)result);

    return EXIT_FAILURE;
  }

  int retc = EXIT_SUCCESS;

  const double action[V2DP_ACTION_LENGTH] = {1.0, 1.0, 0.0};

  double noise_variance[V2DP_NOISE_LENGTH];
  v2dp_noise_variance(&process_model_state, action, noise_variance);

  const double linear_velocity_stdev = sqrt(noise_variance[0]);
  const double angular_velocity_stdev = sqrt(noise_variance[1]);
  const double turn_velocity_stdev = sqrt(noise_variance[2]);

  for (size_t i = 0; i < num_timesteps; ++i) {
    const double t = process_model_state.sampling_period * (double)i;

    const struct double_pair noise0 = standard_normal_random(&pcg_rng);
    const struct double_pair noise1 = standard_normal_random(&pcg_rng);

    const double noisy_action[V2DP_ACTION_LENGTH] = {
        action[0] + noise0.first * linear_velocity_stdev,
        action[1] + noise0.second * angular_velocity_stdev,
        action[2] + noise1.first * turn_velocity_stdev};

    double next_state[SE2_STATE_LENGTH];
    v2dp_motion_model(&process_model_state, true_state, action, next_state);
    memcpy(true_state, next_state, sizeof(true_state));

    const struct double_pair noise2 = standard_normal_random(&pcg_rng);

    const double landmark_x_offset =
        measurement_model_state.landmark_x - true_state[0];
    const double landmark_y_offset =
        measurement_model_state.landmark_y - true_state[1];

    const double landmark_bearing =
        wrap_to_pi(atan2(landmark_y_offset, landmark_x_offset) - true_state[2]);
    const double landmark_range = hypot(landmark_x_offset, landmark_y_offset);

    const double measurement[L2DM_MEASUREMENT_LENGTH] = {
        wrap_to_pi(landmark_bearing + noise1.second * BEARING_STDEV),
        landmark_range + noise0.second * BEARING_STDEV};

    result = pf_predict(&filter, action, &random_number_generator);

    if (result != PF_OK) {
      fprintf(stderr, "error: pf_predict() failed: %s (%d)\n",
              pf_error_description(result).data, (int)result);
      retc = EXIT_FAILURE;

      goto cleanup;
    }

    result = pf_correct(&filter, measurement, &random_number_generator);

    if (result != PF_OK) {
      fprintf(stderr, "error: pf_correct() failed: %s (%d)\n",
              pf_error_description(result).data, (int)result);
      retc = EXIT_FAILURE;

      goto cleanup;
    }

    struct se2_mean_covariance mean_covariance;
    result = pf_particles_function(
        &filter, &se2_compute_weighted_mean_and_covariance, &mean_covariance);

    printf("- time: %f\n"
           "  true mean: [%f, %f, %f]\n"
           "  estimated mean: [%f, %f, %f]\n"
           "  estimated covariance:\n"
           "  - [%f, %f, %f]\n"
           "  - [%f, %f, %f]\n"
           "  - [%f, %f, %f]\n",
           t, true_state[0], true_state[1], true_state[2],
           mean_covariance.mean[0], mean_covariance.mean[1],
           mean_covariance.mean[2], mean_covariance.covariance[0][0],
           mean_covariance.covariance[0][1], mean_covariance.covariance[0][2],
           mean_covariance.covariance[1][0], mean_covariance.covariance[1][1],
           mean_covariance.covariance[1][2], mean_covariance.covariance[2][0],
           mean_covariance.covariance[2][1], mean_covariance.covariance[2][2]);
  }

cleanup:
  result = pf_drop(&filter, &system_allocator);

  if (result != PF_OK) {
    fprintf(stderr, "error: pf_drop() failed: %s (%d)\n",
            pf_error_description(result).data, (int)result);
    retc = EXIT_FAILURE;
  }

  return retc;
}

static uint32_t rng(void *restrict arg) {
  assert(arg);

  return pcg32_random_r((pcg32_random_t *)arg);
}

static double random_in_01(pcg32_random_t *restrict rng);

static struct double_pair standard_normal_random(pcg32_random_t *restrict rng) {
  assert(rng);

  const double u1 = random_in_01(rng);
  const double u2 = random_in_01(rng);

  const double magnitude = sqrt(-2.0 * log(1.0 - u1));
  const double angle = 2.0 * PI * u2;

  const double z0 = magnitude * cos(angle);
  const double z1 = magnitude * sin(angle);

  return (struct double_pair){.first = z0, .second = z1};
}

static double random_in_01(pcg32_random_t *restrict rng) {
  static_assert(FLT_RADIX == 2,
                "double must be a binary floating point number");
  static_assert(DBL_MANT_DIG <= 64,
                "double must have 64 or fewer mantissa digits!");

  assert(rng);

  const uint32_t upper_bits = pcg32_random_r(rng);
  const uint32_t lower_bits = pcg32_random_r(rng);

  const uint64_t bits = ((uint64_t)upper_bits << 32) | (uint64_t)lower_bits;
  const uint64_t mantissa = bits >> (64 - DBL_MANT_DIG);

  return (double)mantissa / (double)((uint64_t)1 << DBL_MANT_DIG);
}

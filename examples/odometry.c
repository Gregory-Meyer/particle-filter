#include <pf.h>

#include <assert.h>
#include <complex.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <pcg_variants.h>

#define ACTION_LENGTH ((size_t)3)
#define STATE_LENGTH ((size_t)3)
#define PROCESS_NOISE_LENGTH ((size_t)3)
#define MEASUREMENT_LENGTH ((size_t)2)
#define PI 3.14159265358979323846264338327950288

#define SQUARE(X) ((X) * (X))
#define RADIANS(X) ((X)*PI / 180.0)

#define BEARING_STDEV RADIANS(5.0)
#define RANGE_STDEV 25.0

struct process_model_state {
  double sampling_period;
  double alpha[6];
};

struct measurement_model_state {
  double bearing_variance;
  double range_variance;
  double landmark_x;
  double landmark_y;
};

struct double_pair {
  double first;
  double second;
};

struct mean_covariance {
  double mean[STATE_LENGTH];
  double covariance[STATE_LENGTH][STATE_LENGTH];
};

static int process_model(
    void *restrict arg, size_t action_length,
    const double action[restrict static ACTION_LENGTH], size_t num_states,
    size_t state_length, const double states[restrict num_states][STATE_LENGTH],
    size_t noise_length,
    const double noise_vectors[restrict num_states][PROCESS_NOISE_LENGTH],
    double next_states[restrict num_states][STATE_LENGTH]);
static int process_noise_covariance(
    void *restrict arg, size_t action_length,
    const double action[restrict static ACTION_LENGTH], size_t noise_length,
    double process_noise_covariance[restrict static PROCESS_NOISE_LENGTH]
                                   [PROCESS_NOISE_LENGTH]);
static uint32_t rng(void *restrict arg);
static int measurement_model_probability_density(
    void *restrict arg, size_t measurement_length,
    const double measurement[restrict static MEASUREMENT_LENGTH],
    size_t num_states, size_t state_length,
    const double states[restrict num_states][STATE_LENGTH],
    double likelihoods[restrict num_states]);
static void *allocator(void *restrict arg, size_t size, size_t align);
static void deallocator(void *restrict arg, void *restrict ptr, size_t size,
                        size_t align);
static void motion_model(double timestep,
                         const double state[restrict static STATE_LENGTH],
                         const double action[restrict ACTION_LENGTH],
                         double next_state[restrict static STATE_LENGTH]);
static struct double_pair standard_normal_random(pcg32_random_t *restrict rng);
static double wrap_to_pi(double angle);
static void
process_noise_variance(const struct process_model_state *restrict state,
                       double linear_velocity, double angular_velocity,
                       double variance[restrict static PROCESS_NOISE_LENGTH]);
static void compute_mean_covariance(
    void *restrict arg, size_t num_particles, size_t state_length,
    const double states[restrict num_particles][STATE_LENGTH],
    const double weights[restrict num_particles]);

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

  struct process_model_state process_model_state = {
      .sampling_period = 0.01,
      .alpha = {SQUARE(0.00025), SQUARE(0.00005), SQUARE(0.0025),
                SQUARE(0.0005), SQUARE(0.0025), SQUARE(0.0005)}};
  struct measurement_model_state measurement_model_state = {
      .bearing_variance = SQUARE(BEARING_STDEV),
      .range_variance = SQUARE(RANGE_STDEV),
      .landmark_x = 0.0,
      .landmark_y = 0.0};

  struct pf_particle_filter filter = {
      .process_model =
          {
              .model_fn = &process_model,
              .noise_covariance_fn = &process_noise_covariance,
              .action_length = (int)ACTION_LENGTH,
              .noise_length = (int)PROCESS_NOISE_LENGTH,
              .arg = &process_model_state,
          },

      .measurement_model =
          {
              .pdf = &measurement_model_probability_density,
              .measurement_length = (int)MEASUREMENT_LENGTH,
              .arg = &measurement_model_state,
          },

      .state_length = (int)STATE_LENGTH,
      .num_particles = 512,
  };

  pcg32_random_t pcg_rng = PCG32_INITIALIZER;
  const struct pf_random_number_generator random_number_generator = {
      .rng_fn = &rng,
      .arg = &pcg_rng,
  };

  const struct pf_allocator system_allocator = {
      .alloc = &allocator, .dealloc = &deallocator, .arg = NULL};

  double true_state[STATE_LENGTH] = {180.0, 50.0, 0.0};
  const double initial_belief_covariance[STATE_LENGTH][STATE_LENGTH] = {
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

  const double linear_velocity = 1.0;
  const double angular_velocity = 1.0;

  double noise_variance[PROCESS_NOISE_LENGTH];
  process_noise_variance(&process_model_state, linear_velocity,
                         angular_velocity, noise_variance);

  const double linear_velocity_stdev = sqrt(noise_variance[0]);
  const double angular_velocity_stdev = sqrt(noise_variance[1]);
  const double turn_velocity_stdev = sqrt(noise_variance[2]);

  for (size_t i = 0; i < num_timesteps; ++i) {
    const double t = process_model_state.sampling_period * (double)i;

    const struct double_pair noise0 = standard_normal_random(&pcg_rng);
    const struct double_pair noise1 = standard_normal_random(&pcg_rng);

    const double noisy_linear_velocity =
        linear_velocity + noise0.first * linear_velocity_stdev;
    const double noisy_angular_velocity =
        angular_velocity + noise0.second * angular_velocity_stdev;
    const double noisy_turn_velocity = noise1.first * turn_velocity_stdev;

    const double action[ACTION_LENGTH] = {
        noisy_linear_velocity, noisy_angular_velocity, noisy_turn_velocity};

    double next_state[STATE_LENGTH];
    motion_model(process_model_state.sampling_period, true_state, action,
                 next_state);
    memcpy(true_state, next_state, sizeof(true_state));

    const struct double_pair noise2 = standard_normal_random(&pcg_rng);

    const double landmark_x_offset =
        measurement_model_state.landmark_x - true_state[0];
    const double landmark_y_offset =
        measurement_model_state.landmark_y - true_state[1];

    const double landmark_bearing =
        wrap_to_pi(atan2(landmark_y_offset, landmark_x_offset) - true_state[2]);
    const double noisy_landmark_bearing =
        landmark_bearing + noise1.second * BEARING_STDEV;

    const double landmark_range = hypot(landmark_x_offset, landmark_y_offset);
    const double noisy_landmark_range =
        landmark_range + noise2.first * RANGE_STDEV;

    const double measurement[MEASUREMENT_LENGTH] = {noisy_landmark_bearing,
                                                    noisy_landmark_range};

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

    struct mean_covariance mean_covariance;
    result = pf_particles_function(&filter, &compute_mean_covariance,
                                   &mean_covariance);

    printf("- time: %f\n  true mean: [%f, %f, %f]\n  estimated mean: [%f, %f, "
           "%f]\n  estimated covariance:\n  - [%f, %f, %f]\n  - [%f, %f, %f]\n"
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

static int process_model(
    void *restrict arg, size_t action_length,
    const double action[restrict static ACTION_LENGTH], size_t num_states,
    size_t state_length, const double states[restrict num_states][STATE_LENGTH],
    size_t noise_length,
    const double noise_vectors[restrict num_states][PROCESS_NOISE_LENGTH],
    double next_states[restrict num_states][STATE_LENGTH]) {
  assert(arg != NULL);
  assert(action_length == ACTION_LENGTH);
  assert(action != NULL);
  assert(num_states > 0);
  assert(state_length == STATE_LENGTH);
  assert(states != NULL);
  assert(noise_length == PROCESS_NOISE_LENGTH);
  assert(noise_vectors != NULL);
  assert(next_states != NULL);

  const struct process_model_state *const restrict state =
      (struct process_model_state *)arg;

  const double linear_velocity = action[0];  // v
  const double angular_velocity = action[1]; // omega
  const double turn_velocity = action[2];    // gamma

  for (size_t i = 0; i < num_states; ++i) {
    const double noisy_linear_velocity = linear_velocity + noise_vectors[i][0];
    const double noisy_angular_velocity =
        angular_velocity + noise_vectors[i][1];
    const double noisy_turn_velocity = turn_velocity + noise_vectors[i][2];

    const double noisy_action[ACTION_LENGTH] = {
        noisy_linear_velocity, noisy_angular_velocity, noisy_turn_velocity};

    motion_model(state->sampling_period, states[i], noisy_action,
                 next_states[i]);
  }

  return 0;
}

static double wrap_to_pi(double angle) {
  while (angle >= PI) {
    angle -= 2.0 * PI;
  }

  while (angle < -PI) {
    angle += 2.0 * PI;
  }

  return angle;
}

static void motion_model(double timestep,
                         const double state[restrict static STATE_LENGTH],
                         const double action[restrict ACTION_LENGTH],
                         double next_state[restrict static STATE_LENGTH]) {
  assert(timestep > 0.0);
  assert(state != NULL);
  assert(action != NULL);
  assert(next_state != NULL);

  const double x_position = state[0];          // x
  const double y_position = state[1];          // y
  const double heading = wrap_to_pi(state[2]); // theta

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

static int process_noise_covariance(
    void *restrict arg, size_t action_length,
    const double action[restrict static ACTION_LENGTH], size_t noise_length,
    double process_noise_covariance[restrict static PROCESS_NOISE_LENGTH]
                                   [PROCESS_NOISE_LENGTH]) {
  assert(arg);
  assert(action_length == ACTION_LENGTH);
  assert(action);
  assert(noise_length == PROCESS_NOISE_LENGTH);
  assert(process_noise_covariance);

  const struct process_model_state *const restrict state =
      (struct process_model_state *)arg;

  const double linear_velocity = action[0];
  const double angular_velocity = action[1];

  double noise_variance[PROCESS_NOISE_LENGTH];
  process_noise_variance(state, linear_velocity, angular_velocity,
                         noise_variance);

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

static void
process_noise_variance(const struct process_model_state *restrict state,
                       double linear_velocity, double angular_velocity,
                       double variance[restrict static PROCESS_NOISE_LENGTH]) {
  assert(state);
  assert(variance);

  const double v2 = linear_velocity * linear_velocity;
  const double omega2 = angular_velocity * angular_velocity;

  const double *const restrict alpha = state->alpha;

  variance[0] = alpha[0] * v2 + alpha[1] * omega2;
  variance[1] = alpha[2] * v2 + alpha[3] * omega2;
  variance[2] = alpha[4] * v2 + alpha[5] * omega2;
}

static uint32_t rng(void *restrict arg) {
  assert(arg);

  return pcg32_random_r((pcg32_random_t *)arg);
}

static int measurement_model_probability_density(
    void *restrict arg, size_t measurement_length,
    const double measurement[restrict static MEASUREMENT_LENGTH],
    size_t num_states, size_t state_length,
    const double states[restrict num_states][STATE_LENGTH],
    double likelihoods[restrict num_states]) {
  assert(arg);
  assert(measurement_length == MEASUREMENT_LENGTH);
  assert(measurement);
  assert(num_states > 0);
  assert(state_length == STATE_LENGTH);
  assert(states);
  assert(likelihoods);

  const struct measurement_model_state *const restrict state = arg;

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

static bool is_power_of_two(size_t x);

static void *allocator(void *restrict arg, size_t size, size_t align) {
  (void)arg;
  assert(size > 0);
  assert(is_power_of_two(align));

  return aligned_alloc(align, size);
}

static bool is_power_of_two(size_t x) {
  return (x != 0) && ((x & (x - 1)) == 0);
}

static void deallocator(void *restrict arg, void *restrict ptr, size_t size,
                        size_t align) {
  (void)arg;
  assert(ptr);
  assert(size > 0);
  assert(is_power_of_two(align));

  free(ptr);
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

static void compute_mean_covariance(
    void *restrict arg, size_t num_particles, size_t state_length,
    const double states[restrict num_particles][STATE_LENGTH],
    const double weights[restrict num_particles]) {
  assert(arg);
  assert(num_particles > 0);
  assert(state_length == STATE_LENGTH);
  assert(states);
  assert(weights);

  struct mean_covariance *const restrict mean_covariance = arg;

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

  double covariance_accumulator[STATE_LENGTH][STATE_LENGTH] = {0};
  double squared_weights_sum_accumulator = 0.0;

  for (size_t particle_index = 0; particle_index < num_particles;
       ++particle_index) {
    const double this_weight = weights[particle_index];
    squared_weights_sum_accumulator += this_weight * this_weight;

    const double offsets[STATE_LENGTH] = {
        states[particle_index][0] - x_mean_accumulator,
        states[particle_index][1] - y_mean_accumulator,
        wrap_to_pi(states[particle_index][2] - angle_mean),
    };

    for (size_t i = 0; i < STATE_LENGTH; ++i) {
      const double partial_product = this_weight * offsets[i];

      for (size_t j = 0; j < STATE_LENGTH; ++j) {
        covariance_accumulator[i][j] += partial_product * offsets[j];
      }
    }
  }

  const double covariance_normalizer = 1.0 - squared_weights_sum_accumulator;

  for (size_t i = 0; i < STATE_LENGTH; ++i) {
    for (size_t j = 0; j < STATE_LENGTH; ++j) {
      covariance_accumulator[i][j] /= covariance_normalizer;
    }
  }

  for (size_t i = 0; i < STATE_LENGTH; ++i) {
    for (size_t j = 0; j < STATE_LENGTH; ++j) {
      mean_covariance->covariance[i][j] = covariance_accumulator[i][j];
    }
  }
}

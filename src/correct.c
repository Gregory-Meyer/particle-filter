// Copyright (C) 2020 Gregory Meyer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "internal/random.h"
#include "internal/util.h"

#include <pf.h>

#include <assert.h>
#include <string.h>

static void resample(struct pf_particle_filter *restrict filter,
                     const struct pf_random_number_generator *restrict rng);

/// Update particle weights using the measurement model PDF and measurement.
///
/// If the filter appears to be suffering from sample impoverishment, it is
/// resampled using the low-variance resampling algorithm described in
/// Probabilistic Robotics.
///
/// @param filter must be non-null
/// @param measurement \f$\mathbf{z}_k\f$. Must be non-null
///
/// @returns PF_OK if the filter updated all particle weights
enum pf_status pf_correct(
    struct pf_particle_filter *restrict filter,
    const double
        measurement[restrict filter->measurement_model.measurement_length],
    const struct pf_random_number_generator *restrict rng) {
  if (!filter) {
    return PF_NULL_FILTER;
  } else if (!measurement) {
    return PF_NULL_MEASUREMENT;
  } else if (!rng) {
    return PF_NULL_RNG;
  } else if (!check_invariants(filter)) {
    return PF_INVALID_FILTER_STATE;
  }

  const size_t num_particles = (size_t)filter->num_particles;
  double inverse_effective_sample_size;

  {
    const size_t state_length = (size_t)filter->state_length;
    const double *restrict states;

    if (filter->use_first_particle_states) {
      states = filter->particle_states;
    } else {
      states = filter->particle_states + (num_particles * state_length);
    }

    const double *restrict weights;
    double *restrict next_weights_or_likelihoods;

    if (filter->use_first_particle_weights) {
      weights = filter->particle_weights;
      next_weights_or_likelihoods = filter->particle_weights + num_particles;
    } else {
      weights = filter->particle_weights + num_particles;
      next_weights_or_likelihoods = filter->particle_weights;
    }

    const size_t measurement_length =
        (size_t)filter->measurement_model.measurement_length;
    if (filter->measurement_model.pdf(
            filter->measurement_model.arg, measurement_length, measurement,
            num_particles, state_length, (const double(*)[])states,
            next_weights_or_likelihoods) != 0) {
      return PF_MEASUREMENT_PDF_FAILED;
    }

    double unnormalized_next_weights_sum = 0.0;
    for (size_t i = 0; i < num_particles; ++i) {
      const double next_weight_unnormalized =
          weights[i] * next_weights_or_likelihoods[i];

      unnormalized_next_weights_sum += next_weight_unnormalized;
      next_weights_or_likelihoods[i] = next_weight_unnormalized;
    }

    inverse_effective_sample_size = 0.0;
    for (size_t i = 0; i < num_particles; ++i) {
      const double next_weight =
          next_weights_or_likelihoods[i] / unnormalized_next_weights_sum;

      inverse_effective_sample_size += next_weight * next_weight;
      next_weights_or_likelihoods[i] = next_weight;
    }

    filter->use_first_particle_weights = !filter->use_first_particle_weights;
  }

  const double effective_sample_size = 1.0 / inverse_effective_sample_size;
  const double resampling_threshold = (double)num_particles / 3.0;

  if (effective_sample_size < resampling_threshold) {
    resample(filter, rng);
  }

  return PF_OK;
}

static void resample(struct pf_particle_filter *restrict filter,
                     const struct pf_random_number_generator *restrict rng) {
  assert(filter);
  assert(rng);
  assert(check_invariants(filter));
  assert(rng->rng_fn);

  const size_t num_particles = (size_t)filter->num_particles;
  const size_t state_length = (size_t)filter->state_length;
  const double *restrict states;
  double *restrict next_states;

  if (filter->use_first_particle_states) {
    states = filter->particle_states;
    next_states = filter->particle_states + (num_particles * state_length);
  } else {
    states = filter->particle_states + (num_particles * state_length);
    next_states = filter->particle_states;
  }

  const double *restrict weights;
  double *restrict next_weights;

  if (filter->use_first_particle_weights) {
    weights = filter->particle_weights;
    next_weights = filter->particle_weights + num_particles;
  } else {
    weights = filter->particle_weights + num_particles;
    next_weights = filter->particle_weights;
  }

  const double n_inverse = 1.0 / (double)num_particles;
  const double r = random_in_01(rng) * n_inverse;
  size_t j = 0;
  double weights_cumulative_sum = weights[0];

  for (size_t i = 0; i < num_particles; ++i) {
    const double u = r + (double)i * n_inverse;

    while (u > weights_cumulative_sum) {
      ++j;
      weights_cumulative_sum += weights[j];
    }

    memcpy(&next_states[i * state_length], &states[j * state_length],
           sizeof(double) * state_length);
    next_weights[i] = n_inverse;
  }

  filter->use_first_particle_states = !filter->use_first_particle_states;
  filter->use_first_particle_weights = !filter->use_first_particle_weights;
}

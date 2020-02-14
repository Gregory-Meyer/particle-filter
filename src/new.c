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

#include "internal/memory.h"
#include "internal/random.h"

#include <pf.h>

#include <string.h>

#include <openblas/cblas.h>

/// Initialize a particle filter, making all necessary memory preallocations.
///
/// @param filter must be non-null. Unmodified if anything other than PF_OK is
///        returned
/// @param belief_mean must be non-null. Initial state for all particles
/// @param allocator must be non-null
///
/// @returns PF_OK if the filter was successfully initialized
enum pf_status
pf_new(struct pf_particle_filter *restrict filter,
       const double belief_mean[restrict filter->state_length],
       const double belief_covariance[restrict filter->state_length]
                                     [filter->state_length],
       const struct pf_random_number_generator *restrict rng,
       const struct pf_allocator *restrict allocator) {
  if (filter == NULL) {
    return PF_NULL_FILTER;
  } else if (belief_mean == NULL) {
    return PF_NULL_BELIEF_MEAN;
  } else if (belief_covariance == NULL) {
    return PF_NULL_BELIEF_COVARIANCE;
  } else if (rng == NULL) {
    return PF_NULL_RNG;
  } else if (allocator == NULL) {
    return PF_NULL_ALLOCATOR;
  } else if (filter->process_model.model_fn == NULL) {
    return PF_NULL_PROCESS_MODEL_FN;
  } else if (filter->process_model.noise_covariance_fn == NULL) {
    return PF_NULL_PROCESS_MODEL_NOISE_COVARIANCE_FN;
  } else if (filter->process_model.action_length <= 0) {
    return PF_NONPOSITIVE_ACTION_LENGTH;
  } else if (filter->process_model.noise_length <= 0) {
    return PF_NONPOSITIVE_PROCESS_NOISE_LENGTH;
  } else if (filter->measurement_model.pdf == NULL) {
    return PF_NULL_MEASUREMENT_MODEL_PROBABILITY_DENSITY_FN;
  } else if (filter->measurement_model.measurement_length <= 0) {
    return PF_NONPOSITIVE_MEASUREMENT_LENGTH;
  } else if (filter->state_length <= 0) {
    return PF_NONPOSITIVE_STATE_LENGTH;
  } else if (filter->num_particles <= 0) {
    return PF_NONPOSITIVE_NUM_PARTICLES;
  } else if (rng->rng_fn == NULL) {
    return PF_NULL_RNG_FN;
  } else if (allocator->alloc == NULL) {
    return PF_NULL_ALLOC_FN;
  } else if (allocator->dealloc == NULL) {
    return PF_NULL_DEALLOC_FN;
  }

  const size_t process_model_noise_length =
      (size_t)filter->process_model.noise_length;
  double *const process_noise_covariance_or_sqrt = alloc_matrix(
      allocator, process_model_noise_length, process_model_noise_length);

  if (process_noise_covariance_or_sqrt == NULL) {
    return PF_OUT_OF_MEMORY;
  }

  const size_t state_length = (size_t)filter->state_length;
  const size_t num_particles = (size_t)filter->num_particles;
  double *const particle_states =
      alloc_matrix(allocator, 2 * num_particles, state_length);

  if (particle_states == NULL) {
    goto err_process_noise_covariance_or_sqrt;
  }

  double *const process_noise_vectors =
      alloc_matrix(allocator, num_particles, process_model_noise_length);

  if (process_noise_vectors == NULL) {
    goto err_particle_states;
  }

  double *const particle_weights = alloc_array(allocator, 2 * num_particles);

  if (!particle_weights) {
    goto err_process_noise_vectors;
  }

  double *const belief_covariance_sqrt =
      alloc_matrix(allocator, state_length, state_length);

  if (!belief_covariance_sqrt) {
    return PF_OUT_OF_MEMORY;
  }

  const double initial_weight = 1.0 / (double)num_particles;
  for (size_t i = 0; i < num_particles; ++i) {
    particle_weights[i] = initial_weight;
  }

  memcpy(belief_covariance_sqrt, belief_covariance,
         sizeof(double) * state_length * state_length);

  const bool belief_covariance_not_positive_definite =
      zero_mean_multivariate_gaussian(rng, num_particles, state_length,
                                      (double(*)[])belief_covariance_sqrt,
                                      (double(*)[])particle_states);

  dealloc_matrix(allocator, belief_covariance_sqrt, state_length, state_length);

  if (belief_covariance_not_positive_definite) {
    goto err_particle_weights;
  }

  for (size_t i = 0; i < num_particles; ++i) {
    cblas_daxpy((blasint)state_length, 1.0, belief_mean, 1,
                &particle_states[i * state_length], 1);
  }

  filter->process_noise_covariance_or_sqrt = process_noise_covariance_or_sqrt;
  filter->particle_states = particle_states;
  filter->process_noise_vectors = process_noise_vectors;
  filter->particle_weights = particle_weights;

  filter->use_first_particle_weights = true;
  filter->use_first_particle_states = true;

  return PF_OK;

err_particle_weights:
  dealloc_array(allocator, particle_weights, 2 * num_particles);

err_process_noise_vectors:
  dealloc_matrix(allocator, process_noise_vectors, num_particles,
                 process_model_noise_length);

err_particle_states:
  dealloc_matrix(allocator, particle_states, 2 * num_particles, state_length);

err_process_noise_covariance_or_sqrt:
  dealloc_matrix(allocator, process_noise_covariance_or_sqrt,
                 process_model_noise_length, process_model_noise_length);

  return PF_OUT_OF_MEMORY;
}

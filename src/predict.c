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

/// Predict new particle states using the process model and the action
/// taken.
///
/// @param filter must be non-null
/// @param action \f$\mathbf{u}_k\f$. Must be non-null
/// @param rng must be non-null
///
/// @returns PF_OK if the filter advanced the state of all particles
enum pf_status
pf_predict(struct pf_particle_filter *restrict filter,
           const double action[restrict filter->process_model.action_length],
           const struct pf_random_number_generator *restrict rng) {
  if (!filter) {
    return PF_NULL_FILTER;
  } else if (!action) {
    return PF_NULL_ACTION;
  } else if (!rng) {
    return PF_NULL_RNG;
  } else if (!check_invariants(filter)) {
    return PF_INVALID_FILTER_STATE;
  } else if (!rng->rng_fn) {
    return PF_NULL_RNG_FN;
  }

  const size_t action_length = (size_t)filter->process_model.action_length;
  const size_t noise_length = (size_t)filter->process_model.noise_length;
  if (filter->process_model.noise_covariance_fn(
          filter->process_model.arg, action_length, action, noise_length,
          (double(*)[])filter->process_noise_covariance_or_sqrt) != 0) {
    return PF_PROCESS_NOISE_COVARIANCE_FN_FAILED;
  }

  const size_t num_particles = (size_t)filter->num_particles;
  const size_t state_length = (size_t)filter->state_length;
  if (zero_mean_multivariate_gaussian(
          rng, num_particles, state_length,
          (double(*)[])filter->process_noise_covariance_or_sqrt,
          (double(*)[])filter->process_noise_vectors)) {
    return PF_PROCESS_NOISE_COVARIANCE_NOT_POSITIVE_DEFINITE;
  }

  const double *restrict states;
  double *restrict next_states;

  if (filter->use_first_particle_states) {
    states = filter->particle_states;
    next_states = filter->particle_states + (num_particles * state_length);
  } else {
    states = filter->particle_states + (num_particles * state_length);
    next_states = filter->particle_states;
  }

  if (filter->process_model.model_fn(
          filter->process_model.arg, action_length, action, num_particles,
          state_length, (const double(*)[])states, noise_length,
          (const double(*)[])filter->process_noise_vectors,
          (double(*)[])next_states) != 0) {
    return PF_PROCESS_MODEL_FN_FAILED;
  }

  filter->use_first_particle_states = !filter->use_first_particle_states;

  return PF_OK;
}

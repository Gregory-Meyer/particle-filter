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

#include "internal/util.h"

#include <pf.h>

/// Run a user-provided function on the particle states and weights.
///
/// @param filter must be non-null
/// @param f must be non-null
/// @param arg user-provided data pointer
///
/// @returns PF_OK if f completed successfully. This will always happen unless
///          some arguments are invalid or the filter's is in an invalid state.
enum pf_status pf_particles_function(
    const struct pf_particle_filter *restrict filter,
    void (*f)(void *restrict arg, size_t num_particles, size_t state_length,
              const double states[restrict num_particles][state_length],
              const double weights[restrict num_particles]),
    void *restrict arg) {
  if (filter == NULL) {
    return PF_NULL_FILTER;
  } else if (f == NULL) {
    return PF_NULL_F;
  } else if (!check_invariants(filter)) {
    return PF_INVALID_FILTER_STATE;
  }

  const size_t num_particles = (size_t)filter->num_particles;
  const size_t state_length = (size_t)filter->state_length;
  const double *restrict states;

  if (filter->use_first_particle_states) {
    states = filter->particle_states;
  } else {
    states = filter->particle_states + (num_particles * state_length);
  }

  const double *restrict weights;

  if (filter->use_first_particle_weights) {
    weights = filter->particle_weights;
  } else {
    weights = filter->particle_weights + num_particles;
  }

  f(arg, num_particles, state_length, (const double(*)[])states, weights);

  return PF_OK;
}

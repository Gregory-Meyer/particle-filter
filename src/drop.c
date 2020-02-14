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
#include "internal/util.h"

#include <pf.h>

/// Deinitialize a particle filter, deallocating all owned memory.
///
/// @param filter must be non-null
/// @param allocator must be non-null
///
/// @returns PF_OK if the filter was successfully deinitialized
enum pf_status pf_drop(struct pf_particle_filter *restrict filter,
                       const struct pf_allocator *restrict allocator) {
  if (filter == NULL) {
    return PF_NULL_FILTER;
  } else if (allocator == NULL) {
    return PF_NULL_ALLOCATOR;
  } else if (!check_invariants(filter)) {
    return PF_INVALID_FILTER_STATE;
  } else if (allocator->dealloc == NULL) {
    return PF_NULL_DEALLOC_FN;
  }

  const size_t num_particles = (size_t)filter->num_particles;
  dealloc_array(allocator, filter->particle_weights, 2 * num_particles);
  filter->particle_weights = NULL;

  const size_t process_model_noise_length =
      (size_t)filter->process_model.noise_length;
  dealloc_matrix(allocator, filter->process_noise_vectors, num_particles,
                 process_model_noise_length);
  filter->process_noise_vectors = NULL;

  const size_t state_length = (size_t)filter->state_length;
  dealloc_matrix(allocator, filter->particle_states, 2 * num_particles,
                 state_length);
  filter->particle_states = NULL;

  dealloc_matrix(allocator, filter->process_noise_covariance_or_sqrt,
                 process_model_noise_length, process_model_noise_length);
  filter->process_noise_covariance_or_sqrt = NULL;

  return PF_OK;
}

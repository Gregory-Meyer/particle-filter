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

#include "util.h"

#include <pf.h>

#include <assert.h>

bool check_invariants(const struct pf_particle_filter *restrict filter) {
  assert(filter);

  return filter->process_model.model_fn != NULL &&
         filter->process_model.noise_covariance_fn != NULL &&
         filter->process_model.action_length > 0 &&
         filter->process_model.noise_length > 0 &&
         filter->measurement_model.pdf != NULL &&
         filter->measurement_model.measurement_length > 0 &&
         filter->state_length > 0 && filter->num_particles > 0 &&
         filter->process_noise_covariance_or_sqrt != NULL &&
         filter->particle_states != NULL &&
         filter->process_noise_vectors != NULL &&
         filter->particle_weights != NULL;
}

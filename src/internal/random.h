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

#ifndef PF_INTERNAL_RANDOM_H
#define PF_INTERNAL_RANDOM_H

struct pf_random_number_generator;

#include <stddef.h>

bool zero_mean_multivariate_gaussian(
    const struct pf_random_number_generator *restrict rng, size_t num_samples,
    size_t sample_length,
    double covariance_then_sqrt[restrict sample_length][sample_length],
    double samples[restrict num_samples][sample_length]);

double random_in_01(const struct pf_random_number_generator *restrict rng);

#endif

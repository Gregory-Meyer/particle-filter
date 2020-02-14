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

#include "memory.h"

#include <pf.h>

#include <assert.h>
#include <stdalign.h>

double *alloc_matrix(const struct pf_allocator *restrict allocator,
                     size_t num_rows, size_t num_cols) {
  assert(allocator);
  assert(allocator->alloc);
  assert(num_rows > 0);
  assert(num_cols > 0);

  return alloc_array(allocator, num_rows * num_cols);
}

double *alloc_array(const struct pf_allocator *restrict allocator,
                    size_t length) {
  assert(allocator);
  assert(allocator->alloc);
  assert(length > 0);

  return allocator->alloc(allocator->arg, sizeof(double) * length,
                          alignof(double));
}

void dealloc_matrix(const struct pf_allocator *restrict allocator,
                    double *restrict ptr, size_t num_rows, size_t num_cols) {
  assert(allocator);
  assert(allocator->dealloc);
  assert(ptr);
  assert(num_rows > 0);
  assert(num_cols > 0);

  dealloc_array(allocator, ptr, num_rows * num_cols);
}

void dealloc_array(const struct pf_allocator *restrict allocator,
                   double *restrict ptr, size_t length) {
  assert(allocator);
  assert(allocator->dealloc);
  assert(length > 0);

  allocator->dealloc(allocator->arg, ptr, sizeof(double) * length,
                     alignof(double));
}

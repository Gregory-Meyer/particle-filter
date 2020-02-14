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

#include <pf.h>

#define MAKE_STRING_SLICE(STR)                                                 \
  ((struct pf_string_slice){.data = (STR), .length = sizeof((STR)) - 1})

/// @returns a character string describing the nature of the error
struct pf_string_slice pf_error_description(enum pf_status status) {
  switch (status) {
  case PF_OK:
    return MAKE_STRING_SLICE("ok");
  case PF_NULL_FILTER:
    return MAKE_STRING_SLICE("null filter");
  case PF_NULL_BELIEF_MEAN:
    return MAKE_STRING_SLICE("null belief_mean");
  case PF_NULL_BELIEF_COVARIANCE:
    return MAKE_STRING_SLICE("null belief_covariance");
  case PF_NULL_RNG:
    return MAKE_STRING_SLICE("null rng");
  case PF_NULL_ALLOCATOR:
    return MAKE_STRING_SLICE("null allocator");
  case PF_NULL_PROCESS_MODEL_FN:
    return MAKE_STRING_SLICE("null filter->process_model.model_fn");
  case PF_NULL_PROCESS_MODEL_NOISE_COVARIANCE_FN:
    return MAKE_STRING_SLICE("null filter->process_model.noise_covariance_fn");
  case PF_NONPOSITIVE_ACTION_LENGTH:
    return MAKE_STRING_SLICE("nonpositive filter->process_model.action_length");
  case PF_NONPOSITIVE_PROCESS_NOISE_LENGTH:
    return MAKE_STRING_SLICE("nonpositive filter->process_model.noise_length");
  case PF_NULL_MEASUREMENT_MODEL_PROBABILITY_DENSITY_FN:
    return MAKE_STRING_SLICE("null filter->measurement_model.pdf");
  case PF_NONPOSITIVE_MEASUREMENT_LENGTH:
    return MAKE_STRING_SLICE(
        "nonpositive filter->measurement_model.measurement_length");
  case PF_NONPOSITIVE_STATE_LENGTH:
    return MAKE_STRING_SLICE("nonpositive filter->state_length");
  case PF_NONPOSITIVE_NUM_PARTICLES:
    return MAKE_STRING_SLICE("nonpositive filter->num_particles");
  case PF_NULL_RNG_FN:
    return MAKE_STRING_SLICE("null rng->rng_fn");
  case PF_NULL_ALLOC_FN:
    return MAKE_STRING_SLICE("null allocator->alloc");
  case PF_NULL_DEALLOC_FN:
    return MAKE_STRING_SLICE("null allocator->dealloc");
  case PF_BELIEF_COVARIANCE_NOT_POSITIVE_DEFINITE:
    return MAKE_STRING_SLICE(
        "belief_covariance was a non positive definite matrix");
  case PF_INVALID_FILTER_STATE:
    return MAKE_STRING_SLICE("*filter is in an invalid state");
  case PF_NULL_ACTION:
    return MAKE_STRING_SLICE("null action");
  case PF_PROCESS_NOISE_COVARIANCE_FN_FAILED:
    return MAKE_STRING_SLICE(
        "filter->process_model.noise_covariance_fn returned nonzero");
  case PF_PROCESS_NOISE_COVARIANCE_NOT_POSITIVE_DEFINITE:
    return MAKE_STRING_SLICE("filter->process_model.noise_covariance_fn "
                             "returned a non positive definite matrix");
  case PF_PROCESS_MODEL_FN_FAILED:
    return MAKE_STRING_SLICE("filter->process_model.model_fn returned nonzero");
  case PF_NULL_MEASUREMENT:
    return MAKE_STRING_SLICE("null measurement");
  case PF_MEASUREMENT_PDF_FAILED:
    return MAKE_STRING_SLICE("filter->measurement_model.pdf returned nonzero");
  case PF_NULL_F:
    return MAKE_STRING_SLICE("null f");
  case PF_OUT_OF_MEMORY:
    return MAKE_STRING_SLICE("allocator->alloc returned NULL");
  }

  return MAKE_STRING_SLICE("unrecognized error code");
}

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

#include "random.h"

#include <pf.h>

#include <assert.h>
#include <float.h>
#include <math.h>

#include <openblas/cblas.h>
#include <openblas/lapacke.h>

static void fill_with_standard_normal_random(
    const struct pf_random_number_generator *restrict rng, size_t n,
    double to_fill[restrict n]);

bool zero_mean_multivariate_gaussian(
    const struct pf_random_number_generator *restrict rng, size_t num_samples,
    size_t sample_length,
    double covariance_then_sqrt[restrict sample_length][sample_length],
    double samples[restrict num_samples][sample_length]) {
  assert(rng);
  assert(num_samples > 0);
  assert(sample_length > 0);
  assert(covariance_then_sqrt);
  assert(samples);

  fill_with_standard_normal_random(rng, num_samples * sample_length,
                                   (double *)samples);

  const lapack_int covariance_order = (lapack_int)sample_length;
  const lapack_int covariance_leading_dimension = (lapack_int)sample_length;
  const int cholesky_result = LAPACKE_dpotrf_work(
      LAPACK_ROW_MAJOR, 'U', covariance_order, (double *)covariance_then_sqrt,
      covariance_leading_dimension);

  if (cholesky_result > 0) {
    return true;
  }

  assert(cholesky_result == 0); // negative values are logical errors on my part

  const blasint samples_num_rows = (blasint)num_samples;
  const blasint samples_num_cols = (blasint)sample_length;
  const blasint samples_leading_dimension = (blasint)sample_length;
  cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
              samples_num_rows, samples_num_cols, 1.0,
              (const double *)covariance_then_sqrt,
              covariance_leading_dimension, (double *)samples,
              samples_leading_dimension);

  return false;
}

struct double_pair {
  double first;
  double second;
};

static struct double_pair
standard_normal_random(const struct pf_random_number_generator *restrict rng);

static void fill_with_standard_normal_random(
    const struct pf_random_number_generator *restrict rng, size_t n,
    double to_fill[restrict n]) {
  assert(rng);
  assert(n > 0);
  assert(to_fill);

  for (; n >= 2; n -= 2, to_fill += 2) {
    const struct double_pair z = standard_normal_random(rng);
    to_fill[0] = z.first;
    to_fill[1] = z.second;
  }

  if (n > 0) {
    const struct double_pair z = standard_normal_random(rng);
    to_fill[0] = z.first;
  }
}

static struct double_pair
standard_normal_random(const struct pf_random_number_generator *restrict rng) {
  static const double PI = 3.14159265358979323846264338327950288;

  assert(rng);

  const double u1 = random_in_01(rng);
  const double u2 = random_in_01(rng);

  const double magnitude = sqrt(-2.0 * log(1.0 - u1));
  const double angle = 2.0 * PI * u2;

  const double z0 = magnitude * cos(angle);
  const double z1 = magnitude * sin(angle);

  return (struct double_pair){.first = z0, .second = z1};
}

double random_in_01(const struct pf_random_number_generator *restrict rng) {
  static_assert(FLT_RADIX == 2,
                "double must be a binary floating point number");
  static_assert(DBL_MANT_DIG <= 64,
                "double must have 64 or fewer mantissa digits!");

  assert(rng);

  const uint32_t upper_bits = rng->rng_fn(rng->arg);
  const uint32_t lower_bits = rng->rng_fn(rng->arg);

  const uint64_t bits = ((uint64_t)upper_bits << 32) | (uint64_t)lower_bits;
  const uint64_t mantissa = bits >> (64 - DBL_MANT_DIG);

  return (double)mantissa / (double)((uint64_t)1 << DBL_MANT_DIG);
}

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

#ifndef PF_H
#define PF_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/// Process model function.
///
/// Used to update particles by sampling from the motion model.
///
/// \f[
///     \mathbf{x}_k = f(\mathbf{u}_k, \mathbf{x}_{k - 1}, \mathbf{w}_k)
/// \f]
///
/// \f[
///     \mathbf{X}_k = \begin{bmatrix}
///         \mathbf{x}^{(1)}_k & \cdots & \mathbf{x}^{(n)}_k
///     \end{bmatrix}
/// \f]
///
/// \f[
///     \mathbf{W}_k = \begin{bmatrix}
///         \mathbf{w}^{(1)}_k & \cdots & \mathbf{w}^{(n)}_k
///     \end{bmatrix}
/// \f]
///
/// @param arg user-provided data pointer
/// @param action_length Must be nonzero
/// @param action \f$\mathbf{u}_k\f$. Must be non-null
/// @param num_states \f$n\f$. Must be nonzero
/// @param state_length must be nonzero
/// @param states \f$\mathbf{X}_{k - 1}\f$. Must be non-null
/// @param noise_length must be nonzero
/// @param noise_vectors \f$\mathbf{W}_k\f$. Must be non-null
/// @param next_states \f$\mathbf{X}_k\f$. Must be non-null
///
/// @returns nonzero if the next state could not be computed
typedef int(pf_process_model_fn_t)(
    void *restrict arg, size_t action_length,
    const double action[restrict action_length], size_t num_states,
    size_t state_length, const double states[restrict num_states][state_length],
    size_t noise_length,
    const double noise_vectors[restrict num_states][noise_length],
    double next_states[restrict num_states][state_length]);

/// Process model noise covariance function.
///
/// @param arg user-provided data pointer
/// @param action_length \f$m\f$. Must be nonzero
/// @param action \f$\mathbf{u}_k\f$. Must be non-null
/// @param noise_length \f$p\f$. Must be nonzero
/// @param process_noise_covariance \f$Cov[\mathbf{w}_k]\f$. Must be positive
///        definite and non-null.
///
/// @returns nonzero if the noise covariance could not be computed
typedef int(pf_process_noise_covariance_fn_t)(
    void *restrict arg, size_t action_length,
    const double action[restrict action_length], size_t noise_length,
    double process_noise_covariance[restrict noise_length][noise_length]);

/// Process model.
struct pf_process_model {
  pf_process_model_fn_t *model_fn;
  pf_process_noise_covariance_fn_t *noise_covariance_fn;
  int action_length;
  int noise_length;
  void *arg;
};

/// Measurement model probability density function.
///
/// \f[
///     \mathbf{z}_k = h(\mathbf{x}_k, \mathbf{v}_k)
/// \f]
///
/// \f[
///     \mathbf{X}_k = \begin{bmatrix}
///         \mathbf{x}^{(1)}_k & \cdots & \mathbf{x}^{(l)}_k
///     \end{bmatrix}
/// \f]
///
/// \f[
///     \mathbf{P}_k = \begin{bmatrix}
///         p(\mathbf{z}_k | \mathbf{x}^{(1)}_k}) & \cdots
///             & p(\mathbf{z}_k | \mathbf{x}^{(n)}_k})
///     \end{bmatrix}
/// \f]
///
/// @param arg user-provided data pointer
/// @param measurement_length must be nonzero
/// @param measurement \f$\mathbf{z}_k\f$. Must be non-null
/// @param num_states \f$n\f$. Must be nonzero
/// @param state_length must be nonzero
/// @param state \f$\mathbf{X}_k\f$. Must be non-null
/// @param likelihoods \f$\mathbf{P}_k\f$. Must be non-null
///
/// @returns nonzero if the predicted measurement could not be computed
typedef int(pf_measurement_model_probability_density_fn_t)(
    void *restrict arg, size_t measurement_length,
    const double measurement[restrict measurement_length], size_t num_states,
    size_t state_length, const double states[restrict num_states][state_length],
    double likelihoods[restrict num_states]);

/// Measurement model.
struct pf_measurement_model {
  pf_measurement_model_probability_density_fn_t *pdf;
  int measurement_length;
  void *arg;
};

/// Size and alignment aware memory allocation function.
///
/// @param arg user-provided data pointer
/// @param size must be positive
/// @param align must be a power of two
///
/// @returns NULL if memory could not be allocated
typedef void *(pf_allocator_fn_t)(void *restrict arg, size_t size,
                                  size_t align);

/// Size and alignment aware memory deallocation function.
///
/// @param arg user-provided data pointer
/// @param ptr a pointer returned by a pf_allocator_fn_t. Must be non-null
/// @param size the size requested by a call to pf_allocator_fn_t. Must be
///        positive
/// @param align the alignment requested by a call to a pf_allocator_fn_t. Must
///        be a power of two
typedef void(pf_deallocator_fn_t)(void *restrict arg, void *restrict ptr,
                                  size_t size, size_t align);

/// Size and alignment aware memory allocator.
struct pf_allocator {
  pf_allocator_fn_t *alloc;
  pf_deallocator_fn_t *dealloc;
  void *arg;
};

/// Random number generator function.
///
/// Used to generate noise vectors \f$\mathbf{w}_k\f$ for the process model.
///
/// @param arg user-provided data pointer
///
/// @returns an random integer in the range [0, UINT32_MAX]
typedef uint32_t(pf_random_number_generator_fn_t)(void *restrict arg);

/// Random number generator.
struct pf_random_number_generator {
  pf_random_number_generator_fn_t *rng_fn;
  void *arg;
};

/// Particle filter with low-variance resampling.
struct pf_particle_filter {
  struct pf_process_model process_model;
  struct pf_measurement_model measurement_model;

  int state_length;
  int num_particles;

  double *process_noise_covariance_or_sqrt; // [noise_length x noise_length]
  double *particle_states;       // [(2 * num_particles) x state_length]
  double *process_noise_vectors; // [num_particles x noise_length]
  double *particle_weights;      // [2 * num_particles]

  bool use_first_particle_states;
  bool use_first_particle_weights;
};

/// Error status codes returned by particle filter functions.
enum pf_status {
  PF_OK, ///< No error.

  PF_NULL_FILTER,            ///< filter argument was null.
  PF_NULL_BELIEF_MEAN,       ///< belief_mean argument was null.
  PF_NULL_BELIEF_COVARIANCE, ///< belief_covariance argument was null.
  PF_NULL_RNG,               ///< rng argument was null.
  PF_NULL_ALLOCATOR,         ///< allocator argument was null.

  PF_NULL_PROCESS_MODEL_FN, ///< filter->process_model.model_fn was null.
  /// filter->process_model.noise_covariance_fn was null.
  PF_NULL_PROCESS_MODEL_NOISE_COVARIANCE_FN,
  /// filter->process_model.action_length was not positive.
  PF_NONPOSITIVE_ACTION_LENGTH,
  /// filter->process_model.noise_length was not positive.
  PF_NONPOSITIVE_PROCESS_NOISE_LENGTH,

  /// filter->measurement_model.pdf was null.
  PF_NULL_MEASUREMENT_MODEL_PROBABILITY_DENSITY_FN,
  /// filter->measurement_model.measurement_length was not positive.
  PF_NONPOSITIVE_MEASUREMENT_LENGTH,

  PF_NONPOSITIVE_STATE_LENGTH,  ///< filter->state_length was not positive.
  PF_NONPOSITIVE_NUM_PARTICLES, ///< filter->num_particles was not positive.

  PF_NULL_RNG_FN, ///< rng->rng_fn was null.

  PF_NULL_ALLOC_FN,   ///< allocator->alloc was null.
  PF_NULL_DEALLOC_FN, ///< allocator->dealloc was null.

  /// belief_covariance was not positive definite.
  PF_BELIEF_COVARIANCE_NOT_POSITIVE_DEFINITE,

  /// The invariants of a struct pf_particle_filter are not true.
  PF_INVALID_FILTER_STATE,

  PF_NULL_ACTION, ///< action argument was null.

  /// filter->process_model.noise_covariance_fn returned nonzero.
  PF_PROCESS_NOISE_COVARIANCE_FN_FAILED,
  /// \f$Cov[\mathbf{w}_k]\f$ was not positive definite.
  PF_PROCESS_NOISE_COVARIANCE_NOT_POSITIVE_DEFINITE,
  /// filter->process_model.model_fn returned nonzero
  PF_PROCESS_MODEL_FN_FAILED,

  PF_NULL_MEASUREMENT, ///< measurement argument was null.

  /// filter->measurement_model.pdf returned nonzero.
  PF_MEASUREMENT_PDF_FAILED,

  PF_NULL_F, ///< f argument was null.

  PF_OUT_OF_MEMORY, ///< Allocation returned a null pointer.
};

/// Null-terminated string slice with length.
struct pf_string_slice {
  const char *data;
  size_t length;
};

/// Initialize a particle filter, making all necessary memory preallocations.
///
/// @param filter must be non-null. Unmodified if anything other than PF_OK is
///        returned
/// @param belief_mean must be non-null. Particle population mean
/// @param belief_covariance must be non-null. Particle population covariance
/// @param rng must be non-null
/// @param allocator must be non-null
///
/// @returns PF_OK if the filter was successfully initialized
enum pf_status
pf_new(struct pf_particle_filter *restrict filter,
       const double belief_mean[restrict filter->state_length],
       const double belief_covariance[restrict filter->state_length]
                                     [filter->state_length],
       const struct pf_random_number_generator *restrict rng,
       const struct pf_allocator *restrict allocator);

/// Deinitialize a particle filter, deallocating all owned memory.
///
/// @param filter must be non-null
/// @param allocator must be non-null
///
/// @returns PF_OK if the filter was successfully deinitialized
enum pf_status pf_drop(struct pf_particle_filter *restrict filter,
                       const struct pf_allocator *restrict allocator);

/// Predict new particle states using the process model and the action taken.
///
/// @param filter must be non-null
/// @param action \f$\mathbf{u}_k\f$. Must be non-null
/// @param rng must be non-null
///
/// @returns PF_OK if the filter advanced the state of all particles
enum pf_status
pf_predict(struct pf_particle_filter *restrict filter,
           const double action[restrict filter->process_model.action_length],
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
    const struct pf_random_number_generator *restrict rng);

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
    void *restrict arg);

/// @returns a character string describing the nature of the error
struct pf_string_slice pf_error_description(enum pf_status status);

#endif

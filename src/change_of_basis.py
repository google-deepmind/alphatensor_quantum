# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generation of change of basis for AlphaTensor-Quantum."""

import functools

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt


def _sample_triangular_matrix(
    rng: chex.PRNGKey,
    size: int,
    upper_triangular: bool,
    prob_zero_entry: float,
) -> jt.Integer[jt.Array, '{size} {size}']:
  """Returns a random triangular matrix whose diagonal entries are 1."""
  # Sample the off-diagonal elements of the matrix.
  masking_fn = jnp.triu if upper_triangular else jnp.tril
  triangle_part = masking_fn(
      jax.random.bernoulli(rng, p=1.0 - prob_zero_entry, shape=(size, size)),
      k=1 if upper_triangular else -1
  ).astype(jnp.int32)

  # Add the identity matrix. This ensures that the matrix determinant is 1, and
  # therefore it is invertible.
  return jnp.eye(size, dtype=jnp.int32) + triangle_part


@functools.partial(jax.vmap, in_axes=(None, None, 0))
def generate_change_of_basis(
    size: int,
    prob_zero_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, '{size} {size}']:
  """Generates a change of basis matrix.

  Args:
    size: The desired size of the matrix.
    prob_zero_entry: The probability of the sampled entries being zero.
    rng: A Jax random key.

  Returns:
    A change of basis matrix. By construction, the matrix is guaranteed to be
    invertible.
  """
  rng_upper, rng_lower = jax.random.split(rng)
  upper = _sample_triangular_matrix(rng_upper, size, True, prob_zero_entry)
  lower = _sample_triangular_matrix(rng_lower, size, False, prob_zero_entry)
  return jnp.mod(jnp.matmul(upper, lower), 2)


def apply_change_of_basis(
    tensor: jt.Integer[jt.Array, 'size size size'],
    cob_matrix: jt.Integer[jt.Array, 'size size'],
) -> jt.Integer[jt.Array, 'size size size']:
  """Applies a change of basis to a tensor.

  Args:
    tensor: The tensor to be transformed.
    cob_matrix: The change of basis matrix.

  Returns:
    The tensor after applying the change of basis.
  """
  transformed_tensor = jnp.einsum(
      'ia,jb,kc,abc->ijk', cob_matrix, cob_matrix, cob_matrix, tensor
  )
  return jnp.mod(transformed_tensor, 2)

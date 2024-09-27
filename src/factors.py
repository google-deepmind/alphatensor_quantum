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

"""Factor utility methods for AlphaTensor-Quantum."""

import jax.numpy as jnp
import jaxtyping as jt


# The reward saving for completing a gadget:
# - When a Toffoli gadget is completed, we already obtained a reward of -7.0
#   (-1.0 per played action). Since the Toffoli gadget has an equivalent cost of
#   two T gates, we set the saving to 5.0 so that the net reward is -2.0.
# - When a CS gadget is completed, the reward for playing three actions is -3.0,
#   so we set the saving to 1.0 so that the net reward is also -2.0.
TOFFOLI_REWARD_SAVING = 5.0
CS_REWARD_SAVING = 1.0


def action_index_to_factor(
    action: jt.Integer[jt.Scalar, ''], tensor_size: int
) -> jt.Integer[jt.Array, '{tensor_size}']:
  """Converts an action index to a factor.

  Args:
    action: The action index, a scalar in {0, ..., num_actions - 1}, where
      `num_actions = 2 ** tensor_size - 1`.
    tensor_size: The size of the tensor.

  Returns:
    The corresponding factor, containing the action index expressed in base 2 in
    reversed (less significant bit first) order. Note: Since the all-zero factor
    is not allowed, we consider a unit shift, such that action 0 corresponds to
    factor [1, 0, ..., 0].
  """
  action += 1  # Shift by 1.

  factor = jnp.zeros((tensor_size,), dtype=jnp.int32)
  for i in range(tensor_size):
    factor = factor.at[i].set(action % 2)
    action = action // 2
  return factor


def action_factor_to_index(
    factor: jt.Integer[jt.Array, 'tensor_size']
) -> jt.Integer[jt.Scalar, '']:
  """Converts a factor to an action index.

  Args:
    factor: The factor, containing entries in {0, 1}.

  Returns:
    The corresponding action index, i.e., the integer value of the factor when
    considered as a vector in base 2 in reversed (less significant bit first)
    order. Note: Since the all-zero factor is not allowed, we consider a unit
    shift, such that action 0 corresponds to factor [1, 0, ..., 0].
  """
  powers = 2 ** jnp.arange(factor.shape[0])
  return jnp.sum(factor * powers) - 1  # Shift by 1.


def rank_one_update_to_tensor(
    tensor: jt.Integer[jt.Array, 'size size size'],
    factor: jt.Integer[jt.Array, 'size'],
) -> jt.Integer[jt.Array, 'size size size']:
  """Subtracts from the given `tensor` a rank-one tensor formed from `factor`.

  Args:
    tensor: The tensor to update.
    factor: The factor to use for building the rank-one tensor.

  Returns:
    The updated tensor, after subtracting from `tensor` the rank-one tensor
    obtained from the outer product `factor x factor x factor`.
  """
  rank_one_tensor = jnp.einsum('u,v,w->uvw', factor, factor, factor)
  return jnp.mod(tensor - rank_one_tensor, 2)


def factors_are_linearly_independent(
    factor1: jt.Integer[jt.Array, 'size'],
    factor2: jt.Integer[jt.Array, 'size'],
    factor3: jt.Integer[jt.Array, 'size']
) -> jt.Bool[jt.Scalar, '']:
  """Returns whether 3 factors are linearly independent over the field GF(2).

  Args:
    factor1: A factor.
    factor2: A factor.
    factor3: A factor.

  Returns:
    Whether the 3 factors are linearly independent. This function assumes that
    none of the inputs is the all-zero factor (otherwise, the output may be
    incorrect).
  """
  # Because we operate in GF(2), the factors are linearly independent iff they
  # are distinct and `factor3 != factor1 + factor2`.
  distinct = jnp.logical_and(
      jnp.any(factor1 != factor2),
      jnp.logical_and(jnp.any(factor1 != factor3), jnp.any(factor2 != factor3))
  )
  return jnp.logical_and(
      distinct, jnp.any(factor3 != jnp.mod(factor1 + factor2, 2))
  )


def factors_form_toffoli_gadget(
    factors: jt.Integer[jt.Array, '7 size']
) -> jt.Bool[jt.Scalar, '']:
  """Returns whether the 7 input factors form a Toffoli gadget.

  The Toffoli gadget is determined by a list of 7 actions of the form:
    [a, b, c, a+b, a+c, a+b+c, b+c],
  where `a, b, c` are linearly independent vectors (the order is relevant, and
  it has been chosen so that no CS gadget appears inside the list). The T-cost
  of the Toffoli gadget is 2.

  Args:
    factors: The 7 input factors. This function assumes that none of the inputs
    is the all-zero factor (otherwise, the output may be incorrect).

  Returns:
    Whether the 7 input factors form a Toffoli gadget.
  """
  if factors.shape[0] != 7:
    raise ValueError(
        f'The input factors must have shape (7, size). Got: {factors.shape}.'
    )
  a, b, c, ab, ac, abc, bc = factors
  linearly_independent = factors_are_linearly_independent(a, b, c)

  # We check whether the remaining 4 factors are linear combinations of the
  # first three.
  linear_combinations = jnp.logical_and(
      jnp.all(ab == jnp.mod(a + b, 2)),
      jnp.logical_and(
          jnp.all(ac == jnp.mod(a + c, 2)),
          jnp.logical_and(
              jnp.all(abc == jnp.mod(a + b + c, 2)),
              jnp.all(bc == jnp.mod(b + c, 2))
          )
      )
  )
  return jnp.logical_and(linearly_independent, linear_combinations)


def factors_form_cs_gadget(
    factors: jt.Integer[jt.Array, '3 size']
) -> jt.Bool[jt.Scalar, '']:
  """Returns whether the 3 input factors form a CS gadget.

  The CS gadget is determined by a list of 3 actions of the form:
    [a, b, a+b],
  where `a, b` are linearly independent vectors (the order is relevant).

  Args:
    factors: The 3 input factors. This function assumes that none of the inputs
      is the all-zero factor (otherwise, the output may be incorrect).

  Returns:
    Whether the 3 input factors form a CS gadget.
  """
  if factors.shape[0] != 3:
    raise ValueError(
        f'The input factors must have shape (3, size). Got: {factors.shape}.'
    )
  a, b, ab = factors
  # Because we operate in GF(2), `a, b` are linearly independent iff they are
  # distinct.
  linearly_independent = jnp.any(a != b)
  # We check whether the remaining factor is a linear combination of the
  # first two.
  linear_combination = jnp.all(ab == jnp.mod(a + b, 2))
  return jnp.logical_and(linearly_independent, linear_combination)

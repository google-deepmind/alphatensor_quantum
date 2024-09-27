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

"""Generation of synthetic demonstrations for AlphaTensor-Quantum."""

import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import factors as factors_lib


class Demonstration(NamedTuple):
  """A synthetic demonstration (or a batch of them).

  Attributes:
    tensor: The tensor.
    num_factors: The number of factors that form a decomposition of the tensor.
    factors: The sequence of factors. Only the first `num_factors` ones are
      valid, and the rest (up to `max_num_factors`) are zero-padded.
    factors_complete_toffoli_gadget: Whether each factor is the last factor of a
      Toffoli gadget. Only the first `num_factors` entries of
      `factors_complete_toffoli_gadget` are valid.
    factors_complete_cs_gadget: Whether each factor is the last factor of a CS
      gadget. Note that any given factor may be part of at most one gadget. Only
      the first `num_factors` entries of `factors_complete_cs_gadget` are valid.
  """
  tensor: jt.Integer[jt.Array, '*batch tensor_size tensor_size tensor_size']
  num_factors: jt.Integer[jt.Array, '*batch']
  factors: jt.Integer[jt.Array, '*batch max_num_factors tensor_size']
  factors_complete_toffoli_gadget: jt.Bool[jt.Array, '*batch max_num_factors']
  factors_complete_cs_gadget: jt.Bool[jt.Array, '*batch max_num_factors']


class _LoopState(NamedTuple):
  """The state of the loop used in some auxiliary functions.

  Attributes:
    factors: The current factors.
    rng: A Jax random key.
  """
  factors: jt.Integer[jt.Array, 'num_factors size']
  rng: chex.PRNGKey


def _resample_factors(
    state: _LoopState,
    prob_zero_factor_entry: float,
    overwrite_only_zero_factors: bool
) -> _LoopState:
  """Samples a new set of factors to overwrite the input factors.

  This is an auxiliary function to be used within a Jax while loop.

  Args:
    state: The current state of the loop.
    prob_zero_factor_entry: The probability of the entries of the randomly
      generated factors being zero.
    overwrite_only_zero_factors: Whether to overwrite the all-zero factors only,
      or alternatively all the input factors.

  Returns:
    The new state of the loop, where the factors have been overwritten, and the
    random key has been updated.
  """
  rng_used, rng_next = jax.random.split(state.rng)
  new_factors_if_overwrite = jax.random.bernoulli(
      rng_used, p=1.0 - prob_zero_factor_entry, shape=state.factors.shape
  ).astype(jnp.int32)
  overwrite = (
      jnp.all(state.factors == 0, axis=-1, keepdims=True)
      if overwrite_only_zero_factors
      else jnp.ones(state.factors.shape, dtype=jnp.bool_)
  )  # Shape (num_factors, 1 or size).
  return _LoopState(
      rng=rng_next,
      factors=jnp.where(overwrite, new_factors_if_overwrite, state.factors)
  )


def _generate_random_factors(
    num_factors: int,
    size: int,
    prob_zero_factor_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, '{num_factors} {size}']:
  """Generates random factors, ensuring that none of them is all-zero."""
  loop_state = jax.lax.while_loop(
      cond_fun=lambda state: jnp.any(jnp.all(state.factors == 0, axis=-1)),
      body_fun=functools.partial(
          _resample_factors,
          prob_zero_factor_entry=prob_zero_factor_entry,
          overwrite_only_zero_factors=True
      ),
      init_val=_LoopState(
          rng=rng, factors=jnp.zeros((num_factors, size), dtype=jnp.int32)
      )
  )
  return loop_state.factors


def _generate_three_linearly_independent_factors(
    size: int,
    prob_zero_factor_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, 'num_factors=3 {size}']:
  """Generates three linearly independent factors."""

  def _cond_fun(state: _LoopState) -> jt.Bool[jt.Scalar, '']:
    """Returns whether the input factors are linearly dependent or all-zero."""
    any_factor_is_zero = jnp.any(jnp.all(state.factors == 0, axis=-1))
    factors_are_dependent = jnp.logical_not(
        factors_lib.factors_are_linearly_independent(*state.factors)
    )
    return jnp.logical_or(any_factor_is_zero, factors_are_dependent)

  loop_state = jax.lax.while_loop(
      cond_fun=_cond_fun,
      body_fun=functools.partial(
          _resample_factors,
          prob_zero_factor_entry=prob_zero_factor_entry,
          overwrite_only_zero_factors=False
      ),
      init_val=_LoopState(
          rng=rng, factors=jnp.zeros((3, size), dtype=jnp.int32)
      )
  )
  return loop_state.factors


def _generate_two_linearly_independent_factors(
    size: int,
    prob_zero_factor_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, 'num_factors=2 {size}']:
  """Generates two linearly independent factors."""

  def _cond_fun(state: _LoopState) -> jt.Bool[jt.Scalar, '']:
    """Returns whether the input factors are linearly dependent or all-zero."""
    any_factor_is_zero = jnp.any(jnp.all(state.factors == 0, axis=-1))
    # In GF(2), two non-zero factors are linearly dependent iff they are equal.
    factors_are_dependent = jnp.all(state.factors[0] == state.factors[1])
    return jnp.logical_or(any_factor_is_zero, factors_are_dependent)

  loop_state = jax.lax.while_loop(
      cond_fun=_cond_fun,
      body_fun=functools.partial(
          _resample_factors,
          prob_zero_factor_entry=prob_zero_factor_entry,
          overwrite_only_zero_factors=False
      ),
      init_val=_LoopState(
          rng=rng, factors=jnp.zeros((2, size), dtype=jnp.int32)
      )
  )
  return loop_state.factors


def _generate_toffoli_gadget(
    size: int,
    prob_zero_factor_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, 'num_factors=7 {size}']:
  """Generates seven factors that form a Toffoli gadget."""
  a, b, c = _generate_three_linearly_independent_factors(
      size=size,
      prob_zero_factor_entry=prob_zero_factor_entry,
      rng=rng
  )
  ab = jnp.mod(a + b, 2)
  ac = jnp.mod(a + c, 2)
  abc = jnp.mod(a + b + c, 2)
  bc = jnp.mod(b + c, 2)
  # This order is consistent with the definition of the Toffoli gadget in this
  # work (see also `factors.py`), as it ensures that there are no three
  # consecutive factors that form a CS gadget.
  return jnp.stack([a, b, c, ab, ac, abc, bc], axis=0)


def _generate_cs_gadget(
    size: int,
    prob_zero_factor_entry: float,
    rng: chex.PRNGKey
) -> jt.Integer[jt.Array, 'num_factors=3 {size}']:
  """Generates three factors that form a CS gadget."""
  a, b = _generate_two_linearly_independent_factors(
      size=size,
      prob_zero_factor_entry=prob_zero_factor_entry,
      rng=rng
  )
  return jnp.stack([a, b, jnp.mod(a + b, 2)], axis=0)


class _NumGadgetsLoopState(NamedTuple):
  """The loop state used in functions for sampling the number of gadgets.

  Attributes:
    next_num_gadgets: The next value of the total number of gadgets to consider.
    num_toffoli_gadgets: The current number of Toffoli gadgets.
    num_cs_gadgets: The current number of CS gadgets. Note: The total number of
      gadgets is `num_toffoli_gadgets + num_cs_gadgets`, however this quantity
      needs not be consistent with `next_num_gadgets`, because the latter refers
      to the next value of the total number of gadgets to consider (as opposed
      to the current one).
    rng: A Jax random key.
  """
  next_num_gadgets: jt.Integer[jt.Scalar, '']
  num_toffoli_gadgets: jt.Integer[jt.Scalar, '']
  num_cs_gadgets: jt.Integer[jt.Scalar, '']
  rng: chex.PRNGKey


def _sample_num_gadgets_per_type(
    num_gadgets: jt.Integer[jt.Scalar, ''],
    num_factors: jt.Integer[jt.Scalar, ''],
    prob_toffoli_gadget: float,
    rng: chex.PRNGKey
) -> tuple[jt.Integer[jt.Scalar, ''], jt.Integer[jt.Scalar, '']]:
  """Samples the number of gadgets of each type.

  Args:
    num_gadgets: The initial number of gadgets to consider.
    num_factors: The total number of valid factors.
    prob_toffoli_gadget: The probability of a gadget being Toffoli (as opposed
      to CS) for each generated gadget.
    rng: A Jax random key.

  Returns:
    A 2-tuple containing:
    - The number of Toffoli gadgets.
    - The number of CS gadgets.
    This method ensures that the total number of factors taken by the gadgets is
    at most `num_factors`. Therefore, the returned value of gadgets (combining
    both types) may be smaller than `num_gadgets`.
  """

  def _body_fun(state: _NumGadgetsLoopState) -> _NumGadgetsLoopState:
    # Sample the number of gadgets of each type.
    rng_used, rng_next = jax.random.split(state.rng)
    num_toffoli_gadgets = jax.random.binomial(
        key=rng_used,
        n=state.next_num_gadgets,
        p=prob_toffoli_gadget,
    ).astype(jnp.int32)
    num_cs_gadgets = state.next_num_gadgets - num_toffoli_gadgets
    # We decrease `next_num_gadgets` to ensures that, if exceeding the total
    # number of valid factors, in the next iteration of the `while` loop we will
    # sample one fewer gadget.
    return _NumGadgetsLoopState(
        next_num_gadgets=state.next_num_gadgets - 1,
        num_toffoli_gadgets=num_toffoli_gadgets,
        num_cs_gadgets=num_cs_gadgets,
        rng=rng_next,
    )

  def _cond_fun(state: _NumGadgetsLoopState) -> jt.Bool[jt.Scalar, '']:
    # Run the loop while the total number of factors taken by the gadgets is
    # above `num_factors`.
    num_factors_taken = state.num_toffoli_gadgets * 7 + state.num_cs_gadgets * 3
    return num_factors_taken > num_factors

  loop_state = jax.lax.while_loop(
      cond_fun=_cond_fun,
      body_fun=_body_fun,
      init_val=_NumGadgetsLoopState(
          next_num_gadgets=num_gadgets,
          # Use dummy values below to violate stopping criterion.
          num_toffoli_gadgets=num_factors,
          num_cs_gadgets=num_factors,
          rng=rng
      )
  )
  return loop_state.num_toffoli_gadgets, loop_state.num_cs_gadgets


class _GadgetLoopState(NamedTuple):
  """The loop state used in functions for overwriting factors with gadgets.

  Attributes:
    index: The current index pointing to the next factor that could potentially
      be overwritten. That is, `index` is the smallest index where a gadget
      could be inserted, and factors with index smaller than `index` are
      considered fixed and should not be overwritten.
    factors: The current factors. (Only the first `num_factors` entries are
      valid.)
    num_toffoli_gadgets: The remaining number of Toffoli gadgets to be added.
    num_cs_gadgets: The remaining number of CS gadgets to be added.
    factors_complete_toffoli_gadget: Whether each factor is the last one of a
      Toffoli gadget. (Only the first `num_factors` entries are valid.)
    factors_complete_cs_gadget: Whether each factor is the last one of a CS
      gadget. (Only the first `num_factors` entries are valid.)
    rng: A Jax random key.
  """
  index: jt.Integer[jt.Scalar, '']
  factors: jt.Integer[jt.Array, 'max_num_factors size']
  num_toffoli_gadgets: jt.Integer[jt.Scalar, '']
  num_cs_gadgets: jt.Integer[jt.Scalar, '']
  factors_complete_toffoli_gadget: jt.Bool[jt.Array, 'max_num_factors']
  factors_complete_cs_gadget: jt.Bool[jt.Array, 'max_num_factors']
  rng: chex.PRNGKey


def _overwrite_factors_with_gadgets(
    factors: jt.Integer[jt.Array, 'max_num_factors size'],
    num_factors: jt.Integer[jt.Scalar, ''],
    num_toffoli_gadgets: jt.Integer[jt.Scalar, ''],
    num_cs_gadgets: jt.Integer[jt.Scalar, ''],
    config: config_lib.DemonstrationsParams,
    rng: chex.PRNGKey
) -> tuple[jt.Integer[jt.Array, 'max_num_factors size'],
           jt.Bool[jt.Array, 'max_num_factors'],
           jt.Bool[jt.Array, 'max_num_factors']]:
  """Overwrites some factors with gadgets of the given type.

  Args:
    factors: The current factors. (Only the first `num_factors` entries are
      valid.)
    num_factors: The total number of factors.
    num_toffoli_gadgets: The total number of Toffoli gadgets to be added.
    num_cs_gadgets: The total number of CS gadgets to be added.
    config: The hyperparameters describing the synthetic demonstrations.
    rng: A Jax random key.

  Returns:
    A 3-tuple containing:
    - The new factors, after overwriting some with gadgets. There are
      `num_toffoli_gadgets + num_cs_gadgets` gadgets in total.
    - Whether each factor completes a Toffoli gadget.
    - Whether each factor completes a CS gadget.
  """

  def _cond_fun(state: _GadgetLoopState) -> jt.Bool[jt.Scalar, '']:
    """Returns whether there are still gadgets to be added."""
    return jnp.logical_or(
        state.num_toffoli_gadgets > 0, state.num_cs_gadgets > 0
    )

  def _body_fun(state: _GadgetLoopState) -> _GadgetLoopState:
    """Generates one gadget and replaces some factors with it."""
    rng_insert, rng_type, rng_toffoli, rng_cs, rng_next = jax.random.split(
        state.rng, num=5
    )
    num_factors_taken = 7 * state.num_toffoli_gadgets + 3 * state.num_cs_gadgets
    # Sample the index at which the gadget will be inserted. Since gadgets are
    # added in increasing order of `index`, in order to make sure this and all
    # future gadgets fit within the available `num_factors` factors, the sampled
    # index must be in {state.index, ..., num_factors - num_factors_taken}. To
    # meet this condition, and to prevent the position of gadgets from being
    # biased towards either the beginning or the end of the demonstration, we
    # sample `num_factors_taken` indices uniformly at random (and without
    # replacement) in {state.index, ..., num_factors - 1}, and we insert the
    # gadget starting at the smallest sampled index.
    # To avoid variable-length arrays, instead of sampling a set of indices
    # uniformly at random and then taking the minimum, we directly obtain the
    # probability distribution over that minimum, which is:
    #   Prob(i) = (n-i+state.index-1 choose k-1) / (n choose k),
    #             if i \in {state.index, ..., num_factors - num_factors_taken},
    #   Prob(i) = 0, otherwise.
    # where `n=num_factors-state.index` and `k=num_factors_taken`. We construct
    # this probability distribution recursively and obtain a sample from it.
    n = num_factors - state.index
    k = num_factors_taken
    probs_insert_idx = jnp.zeros((config.max_num_factors,))
    probs_insert_idx = probs_insert_idx.at[state.index].set(k / n)
    probs_insert_idx = jax.lax.fori_loop(
        lower=state.index + 1,
        upper=num_factors - num_factors_taken + 1,
        body_fun=lambda i, p: p.at[i].set(
            p[i-1] * (n - k - i + state.index + 1) / (n - i + state.index)
        ),
        init_val=probs_insert_idx
    )
    insert_idx = jax.random.choice(
        rng_insert, config.max_num_factors, p=probs_insert_idx
    )
    # Sample the type of gadget (Toffoli or not, in which case a CS) to add at
    # this iteration.
    insert_toffoli = jax.random.bernoulli(
        rng_type,
        p=state.num_toffoli_gadgets / (
            state.num_toffoli_gadgets + state.num_cs_gadgets
        )
    )
    # Overwrite the factors with the gadget.
    toffoli_factors = _generate_toffoli_gadget(
        factors.shape[-1], config.prob_zero_factor_entry, rng_toffoli
    )
    factors_if_toffoli = jax.lax.fori_loop(
        lower=insert_idx,
        upper=insert_idx + 7,
        body_fun=lambda i, f: f.at[i].set(toffoli_factors[i - insert_idx]),
        init_val=state.factors
    )
    cs_factors = _generate_cs_gadget(
        factors.shape[-1], config.prob_zero_factor_entry, rng_cs
    )
    factors_if_cs = jax.lax.fori_loop(
        lower=insert_idx,
        upper=insert_idx + 3,
        body_fun=lambda i, f: f.at[i].set(cs_factors[i - insert_idx]),
        init_val=state.factors
    )
    # Update the gadget indicators.
    factors_complete_toffoli_gadget_if_toffoli = (
        state.factors_complete_toffoli_gadget.at[insert_idx + 6].set(True)
    )
    factors_complete_cs_gadget_if_cs = (
        state.factors_complete_cs_gadget.at[insert_idx + 2].set(True)
    )
    # Return the new loop state.
    return _GadgetLoopState(
        index=insert_idx + jnp.where(insert_toffoli, 7, 3),
        factors=jnp.where(insert_toffoli, factors_if_toffoli, factors_if_cs),
        num_toffoli_gadgets=state.num_toffoli_gadgets - insert_toffoli,
        num_cs_gadgets=state.num_cs_gadgets - jnp.where(insert_toffoli, 0, 1),
        factors_complete_toffoli_gadget=jnp.where(
            insert_toffoli,
            factors_complete_toffoli_gadget_if_toffoli,
            state.factors_complete_toffoli_gadget
        ),
        factors_complete_cs_gadget=jnp.where(
            insert_toffoli,
            state.factors_complete_cs_gadget,
            factors_complete_cs_gadget_if_cs
        ),
        rng=rng_next,
    )

  # Insert gadgets, one at a time, until all the gadgets are added. Gadgets are
  # added in increasing order of `index`.
  loop_state = jax.lax.while_loop(
      cond_fun=_cond_fun,
      body_fun=_body_fun,
      init_val=_GadgetLoopState(
          index=jnp.zeros((), dtype=jnp.int32),
          factors=factors,
          num_toffoli_gadgets=num_toffoli_gadgets,
          num_cs_gadgets=num_cs_gadgets,
          factors_complete_toffoli_gadget=jnp.zeros(
              (config.max_num_factors,), dtype=jnp.bool_
          ),
          factors_complete_cs_gadget=jnp.zeros(
              (config.max_num_factors,), dtype=jnp.bool_
          ),
          rng=rng
      )
  )
  return (
      loop_state.factors,
      loop_state.factors_complete_toffoli_gadget,
      loop_state.factors_complete_cs_gadget
  )


@functools.partial(jax.vmap, in_axes=(None, None, 0))
def generate_synthetic_demonstrations(
    tensor_size: int,
    config: config_lib.DemonstrationsParams,
    rng: chex.PRNGKey
) -> Demonstration:
  """Generates a synthetic demonstration.

  Args:
    tensor_size: The size of the tensors.
    config: The hyperparameters describing the synthetic demonstrations.
    rng: A Jax random key.

  Returns:
    A randomly generated demonstration.
  """
  rngs = jax.random.split(rng, num=5)

  # Generate `max_num_factors` factors.
  factors = _generate_random_factors(
      num_factors=config.max_num_factors,
      size=tensor_size,
      prob_zero_factor_entry=config.prob_zero_factor_entry,
      rng=rngs[0]
  )

  # Randomly generate the effective number of factors in the demonstration.
  num_factors = jax.random.randint(
      rngs[1],
      shape=(),
      minval=config.min_num_factors,
      maxval=config.max_num_factors + 1,
  )

  # Set the factors beyond `num_factors` to zero to indicate that they are not
  # part of the demonstration, i.e., `factors[num_factors:] = 0`.
  is_padding = jnp.arange(config.max_num_factors) >= num_factors
  padded_factors = jnp.where(jnp.expand_dims(is_padding, axis=-1), 0, factors)

  # Sample the number of gadgets, an integer in {0, ..., max_num_gadgets}.
  prob_zero_gadgets = (1 - config.prob_include_gadget) * jnp.ones((1,))
  probs_nonzero_gadgets = config.prob_include_gadget * jnp.ones(
      (config.max_num_gadgets,)) / config.max_num_gadgets
  probs_num_gadgets = jnp.concatenate(
      [prob_zero_gadgets, probs_nonzero_gadgets]
  )
  num_gadgets = jax.random.choice(
      rngs[2], 1 + config.max_num_gadgets, p=probs_num_gadgets
  )

  # Sample the types of gadgets. It might happen that the sampled gadgets do not
  # fit in `num_factors` factors; if that is the case, we decrease `num_gadgets`
  # and resample the types of gadgets, until they fit.
  num_toffoli_gadgets, num_cs_gadgets = _sample_num_gadgets_per_type(
      num_gadgets, num_factors, config.prob_toffoli_gadget, rngs[3]
  )

  # Overwrite the factors with the type of gadgets that we have sampled.
  (
      factors_with_gadgets,
      factors_complete_toffoli_gadget,
      factors_complete_cs_gadget
  ) = _overwrite_factors_with_gadgets(
      padded_factors,
      num_factors=num_factors,
      num_toffoli_gadgets=num_toffoli_gadgets,
      num_cs_gadgets=num_cs_gadgets,
      config=config,
      rng=rngs[4]
  )

  # Obtain the tensor from the factors. Due to zero-padding, considering all the
  # `max_num_factors` factors leads to the same result as considering only
  # `num_factors` factors.
  tensor = jnp.mod(
      jnp.einsum(
          'ru,rv,rw->uvw',
          factors_with_gadgets,
          factors_with_gadgets,
          factors_with_gadgets
      ),
      2
  )

  # Build the demonstration.
  return Demonstration(
      tensor=tensor,
      num_factors=num_factors,
      factors=factors_with_gadgets,
      factors_complete_toffoli_gadget=factors_complete_toffoli_gadget,
      factors_complete_cs_gadget=factors_complete_cs_gadget,
  )


@jax.vmap
def get_action_and_value(
    demonstration: Demonstration,
    move_index: jt.Integer[jt.Scalar, '']
) -> tuple[jt.Integer[jt.Scalar, ''], jt.Float[jt.Scalar, '']]:
  """Returns the next action and the value of a demonstration at a given index.

  Args:
    demonstration: A synthetic demonstration.
    move_index: The index of the move to consider, an integer in
      {0, ..., demonstration.num_factors - 1}. This method does not check
      whether `move_index` is within valid range.

  Returns:
    A 2-tuple containing:
    - The factor at `move_index` as an action index, i.e., as a scalar in
      {0, ..., num_actions - 1}, where `num_actions = 2 ** tensor_size - 1`.
    - The value at the given move index, i.e., the sum of all future rewards.
  """
  # Obtain the action indexed by `move_index`.
  action_as_factor = jnp.take(demonstration.factors, move_index, axis=0)
  action_as_index = factors_lib.action_factor_to_index(action_as_factor)

  # Auxiliary function for the `fori_loop` below that adjusts the value of the
  # demonstration according to the gadgets it contains.
  def _body_fun(i: int, v: jt.Float[jt.Scalar, '']) -> jt.Float[jt.Scalar, '']:
    return v + jnp.where(
        demonstration.factors_complete_toffoli_gadget[i],
        factors_lib.TOFFOLI_REWARD_SAVING,
        jnp.where(
            demonstration.factors_complete_cs_gadget[i],
            factors_lib.CS_REWARD_SAVING,
            0.0
        )
    )

  # Obtain the value from the given `move_index`, taking into account gadgets.
  value = jax.lax.fori_loop(
      lower=move_index,
      upper=demonstration.num_factors,
      body_fun=_body_fun,
      # The `init_val` is the value at `move_index` ignoring gadgets.
      init_val=(move_index - demonstration.num_factors).astype(jnp.float_)
  )
  return action_as_index, value

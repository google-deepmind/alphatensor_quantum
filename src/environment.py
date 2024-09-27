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

"""Environment for AlphaTensor-Quantum."""

import functools
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jaxtyping as jt

from alphatensor_quantum.src import change_of_basis as change_of_basis_lib
from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations
from alphatensor_quantum.src import factors
from alphatensor_quantum.src import tensors


class EnvState(NamedTuple):
  """State of the environment (or states, if considering batch dimensions).

  Attributes:
    tensor: The residual tensor.
    past_factors: The past played factors. Initially, `past_factors` contains
      all-zero factors, and as the game progresses, the factors are inserted
      from the back (i.e., in the last row), shifting the previous factors
      accordingly.
    num_moves: The current number of moves.
    last_reward: The immediate reward (corresponding to the last played action).
    sum_rewards: The sum of the rewards so far.
    is_terminal: Whether the current environment state is terminal (i.e., the
      game has ended), as a boolean.
    init_tensor_index: The index of the chosen initial tensor from
      `target_circuit_types` (useful for identifying which game is being
      played). If <0, the state does not correspond to any target tensor.
    change_of_basis: The change of basis matrix applied at the beginning of the
      game (or the identity matrix if no change of basis was applied).
    factors_in_gadgets: Which factors are part of a gadget of any type.
  """
  tensor: jt.Integer[jt.Array, '*batch size size size']
  past_factors: jt.Integer[jt.Array, '*batch max_num_moves size']
  num_moves: jt.Integer[jt.Array, '*batch']
  last_reward: jt.Float[jt.Array, '*batch']
  sum_rewards: jt.Float[jt.Array, '*batch']
  is_terminal: jt.Bool[jt.Array, '*batch']
  init_tensor_index: jt.Integer[jt.Array, '*batch']
  change_of_basis: jt.Integer[jt.Array, '*batch size size']
  factors_in_gadgets: jt.Bool[jt.Array, '*batch max_num_moves']


class Observation(NamedTuple):
  """Observation to be passed to the network (possibly with batch dimensions).

  Attributes:
    tensor: The residual tensor.
    past_factors_as_planes: The outer products of the past played factors.
    sqrt_played_fraction: The square root of the ratio of played moves and the
      maximum number of allowed moves.
  """
  tensor: jt.Float[jt.Array, '*batch size size size']
  past_factors_as_planes: jt.Float[jt.Array, '*batch num_factors size size']
  sqrt_played_fraction: jt.Float[jt.Array, '*batch']


class Environment:
  """Environment for AlphaTensor-Quantum."""

  def __init__(self, rng: chex.PRNGKey, config: config_lib.EnvironmentParams):
    """Initializes the environment."""
    self._config = config

    # Obtain the target signature tensors.
    unpadded_target_tensors = [
        tensors.get_signature_tensor(circuit_type)
        for circuit_type in self._config.target_circuit_types
    ]
    self._target_tensors = jnp.stack([
        tensors.zero_pad_tensor(tensor, self._config.max_tensor_size)
        for tensor in unpadded_target_tensors
    ], axis=0)  # Shape (num_target_tensors, size, size, size).

    # Generate a set of change of basis matrices.
    self._change_of_basis = change_of_basis_lib.generate_change_of_basis(
        self._config.max_tensor_size,
        self._config.change_of_basis.prob_zero_entry,
        jax.random.split(
            rng, self._config.change_of_basis.num_change_of_basis_matrices
        )
    )

  @property
  def change_of_basis(self) -> jt.Integer[jt.Array, 'num_matrices size size']:
    return self._change_of_basis

  @functools.partial(jax.vmap, in_axes=(None, 0, 0))
  def step(
      self,
      action: jt.Integer[jt.Scalar, ''],
      env_state: EnvState
  ) -> EnvState:
    """Advances the environment state by applying the given action.

    Args:
      action: The action to apply, as an integer in {0, ..., num_actions - 1}.
      env_state: The current environment state.

    Returns:
      The new environment state.
    """
    factor = factors.action_index_to_factor(
        action, self._config.max_tensor_size
    )
    # Obtain the new environment state and past factors.
    new_tensor = factors.rank_one_update_to_tensor(env_state.tensor, factor)
    new_past_factors = jnp.concatenate(
        [env_state.past_factors[1:], factor[None]], axis=0
    )
    new_num_moves = env_state.num_moves + 1
    # The episode terminates when either we reach the all-zero tensor, or we
    # exceed the maximum number of moves.
    is_terminal = jnp.logical_or(
        jnp.all(new_tensor == 0), new_num_moves >= self._config.max_num_moves
    )
    # Determine whether the last action completed a gadget. Due to the specific
    # ordering of the actions defining the Toffoli and CS gadgets, at most one
    # of the two gadgets can be completed at any given step.
    action_completed_toffoli_gadget = jnp.logical_and(
        self._config.use_gadgets,
        jnp.logical_and(
            jnp.logical_and(
                new_num_moves >= 7,
                jnp.all(jnp.logical_not(env_state.factors_in_gadgets[-6:]))
            ),
            factors.factors_form_toffoli_gadget(new_past_factors[-7:])
        )
    )
    action_completed_cs_gadget = jnp.logical_and(
        self._config.use_gadgets,
        jnp.logical_and(
            jnp.logical_and(
                new_num_moves >= 3,
                jnp.all(jnp.logical_not(env_state.factors_in_gadgets[-2:]))
            ),
            factors.factors_form_cs_gadget(new_past_factors[-3:])
        )
    )
    new_factors_in_gadgets = jnp.concatenate([
        env_state.factors_in_gadgets[1:], jnp.zeros((1,), dtype=jnp.bool_)
    ], axis=0)
    new_factors_in_gadgets = new_factors_in_gadgets.at[-7:].set(
        jnp.where(
            action_completed_toffoli_gadget, True, new_factors_in_gadgets[-7:]
        )
    )
    new_factors_in_gadgets = new_factors_in_gadgets.at[-3:].set(
        jnp.where(action_completed_cs_gadget, True, new_factors_in_gadgets[-3:])
    )
    # In TensorGame, the reward is -1 per move, plus an additional penalty for
    # terminal games that exceeded the maximum number of moves without reaching
    # the all-zero tensor.
    reward = jnp.array(-1.0)
    reward -= jnp.where(is_terminal, jnp.sum(new_tensor), 0.0)
    # Adjust the reward if the last action completed a gadget.
    reward += jnp.where(
        action_completed_toffoli_gadget,
        # The 7 actions in the Toffoli gadget get a net reward of -2.0.
        factors.TOFFOLI_REWARD_SAVING,
        jnp.where(
            action_completed_cs_gadget, factors.CS_REWARD_SAVING, 0.0
        )  # The 3 actions in the CS gadget get a net reward of -2.0.
    )
    return EnvState(
        tensor=new_tensor,
        past_factors=new_past_factors,
        num_moves=new_num_moves,
        is_terminal=is_terminal,
        last_reward=reward,
        sum_rewards=env_state.sum_rewards + reward,
        init_tensor_index=env_state.init_tensor_index,
        change_of_basis=env_state.change_of_basis,
        factors_in_gadgets=new_factors_in_gadgets,
    )

  def _get_init_tensor(
      self, rng: chex.PRNGKey
  ) -> tuple[jt.Integer[jt.Array, 'size size size'],
             jt.Integer[jt.Scalar, '']]:
    """Returns a tensor from the set of target signature tensors.

    Args:
      rng: A Jax random key.

    Returns:
      A 2-tuple:
      - The target tensor, randomly chosen from the set of target signature
        tensors.
      - The index of that tensor in the set of target signature tensors.
    """
    num_target_tensors = len(self._config.target_circuit_types)
    tensor_index = jax.random.choice(
        rng,
        jnp.arange(num_target_tensors),
        p=(
            None if self._config.target_circuit_probabilities is None
            else jnp.array(self._config.target_circuit_probabilities)
        ),
    )
    return self._target_tensors[tensor_index], tensor_index

  def _apply_random_change_of_basis(
      self,
      tensor: jt.Integer[jt.Array, 'size size size'],
      rng: chex.PRNGKey
  ) -> tuple[jt.Integer[jt.Array, 'size size size'],
             jt.Integer[jt.Array, 'size size']]:
    """Applies a randomly chosen change of basis to the given tensor.

    Args:
      tensor: The tensor to apply the change of basis to.
      rng: A Jax random key.

    Returns:
      A 2-tuple:
      - The tensor after applying the change of basis.
      - The applied change of basis matrix.
    """
    rng_canonical, rng_cob = jax.random.split(rng)
    use_canonical_basis = jax.random.bernoulli(
        rng_canonical, self._config.change_of_basis.prob_canonical_basis
    )
    cob_matrix = jax.random.choice(rng_cob, self._change_of_basis)
    matrix = jnp.where(
        use_canonical_basis,
        jnp.eye(self._config.max_tensor_size, dtype=jnp.int32),
        cob_matrix
    )
    return change_of_basis_lib.apply_change_of_basis(tensor, matrix), matrix

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def init_state(self, rng: chex.PRNGKey) -> EnvState:
    """Initializes and returns an environment state.

    Args:
      rng: A Jax random key.

    Returns:
      A new environment state.
    """
    rng_init_tensor, rng_cob = jax.random.split(rng)
    init_tensor, tensor_index = self._get_init_tensor(rng_init_tensor)
    init_tensor_cob, cob_matrix = self._apply_random_change_of_basis(
        init_tensor, rng_cob
    )
    return EnvState(
        tensor=init_tensor_cob,
        past_factors=jnp.zeros(
            (self._config.max_num_moves, self._config.max_tensor_size),
            dtype=jnp.int32
        ),
        num_moves=jnp.zeros((), dtype=jnp.int32),
        is_terminal=jnp.zeros((), dtype=jnp.bool_),
        last_reward=jnp.zeros(()),
        sum_rewards=jnp.zeros(()),
        init_tensor_index=tensor_index,
        change_of_basis=cob_matrix,
        factors_in_gadgets=jnp.zeros(
            (self._config.max_num_moves,), dtype=jnp.bool_
        ),
    )

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def init_state_from_demonstration(
      self, demonstration: demonstrations.Demonstration
  ) -> EnvState:
    """Initializes an environment state from a demonstration.

    Args:
      demonstration: A synthetic demonstration.

    Returns:
      A newly initialized environment state.
    """
    return EnvState(
        tensor=demonstration.tensor,
        past_factors=jnp.zeros(
            (self._config.max_num_moves, self._config.max_tensor_size),
            dtype=jnp.int32
        ),
        num_moves=jnp.zeros((), dtype=jnp.int32),
        is_terminal=jnp.zeros((), dtype=jnp.bool_),
        last_reward=jnp.zeros(()),
        sum_rewards=jnp.zeros(()),
        init_tensor_index=-1,  # Dummy value to indicate that the state does not
                               # correspond to any target from
                               # `target_circuit_types`.
        change_of_basis=jnp.eye(self._config.max_tensor_size, dtype=jnp.int32),
        factors_in_gadgets=jnp.zeros(
            (self._config.max_num_moves,), dtype=jnp.bool_
        ),
    )

  @functools.partial(jax.vmap, in_axes=(None, 0))
  def get_observation(self, env_state: EnvState) -> Observation:
    """Returns the observation that will be passed to the neural network.

    Args:
      env_state: The current environment state.

    Returns:
      The observation that will be passed to the neural network.
    """
    past_active_factors = env_state.past_factors[
        -self._config.num_past_factors_to_observe:
    ].astype(jnp.float_)  # (num_factors, size).
    past_factors_not_in_gadgets = jnp.expand_dims(jnp.logical_not(
        env_state.factors_in_gadgets[-self._config.num_past_factors_to_observe:]
    ), axis=(-1, -2))  # (num_factors, 1, 1).
    return Observation(
        tensor=env_state.tensor.astype(jnp.float_),
        past_factors_as_planes=past_factors_not_in_gadgets * jnp.einsum(
            'fu,fv->fuv', past_active_factors, past_active_factors
        ),
        sqrt_played_fraction=jnp.sqrt(
            env_state.num_moves / self._config.max_num_moves
        ),
    )

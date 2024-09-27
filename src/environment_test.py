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

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import change_of_basis as change_of_basis_lib
from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import factors as factors_utils
from alphatensor_quantum.src import tensors


_SMALL_TCOUNT3_FACTORS = np.array(
    [[1, 1, 1], [0, 1, 1], [1, 0, 1]], dtype=np.int32
)


class EnvironmentTest(parameterized.TestCase):

  def test_init_state(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.
    with self.subTest('tensor'):
      np.testing.assert_array_equal(
          env_state.tensor,
          tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)[None]
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors, np.zeros((1, 10, 3), dtype=np.int32)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.zeros((1,)))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.zeros((1,)))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          env_state.is_terminal, np.zeros((1,), dtype=np.bool_)
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_init_state_change_of_basis(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            prob_zero_entry=0.3,
            num_change_of_basis_matrices=1,
            prob_canonical_basis=0.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.
    with self.subTest('cob_matrix'):
      np.testing.assert_array_equal(
          env_state.change_of_basis[0], env.change_of_basis[0]
      )
    expected_tensor = change_of_basis_lib.apply_change_of_basis(
        tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3),
        env.change_of_basis[0],
    )
    with self.subTest('tensor_is_in_non_canonical_basis'):
      np.testing.assert_array_equal(env_state.tensor[0], expected_tensor)

  def test_init_state_from_demonstration(self):
    env_config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    dem_config = config_lib.DemonstrationsParams(
        max_num_factors=10,
        max_num_gadgets=2,
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, dem_config, jax.random.PRNGKey(0)[None]  # Add batch dim.
    )
    env = environment.Environment(jax.random.PRNGKey(1), env_config)
    env_state = env.init_state_from_demonstration(demonstration)
    with self.subTest('tensor'):
      np.testing.assert_array_equal(env_state.tensor, demonstration.tensor)
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors, np.zeros((1, 10, 3), dtype=np.int32)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.zeros((1,)))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.zeros((1,)))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          env_state.is_terminal, np.zeros((1,), dtype=np.bool_)
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, -np.ones((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_one_step(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    factor1 = jnp.array([1, 1, 1], dtype=jnp.int32)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor1)[None], env_state
    )
    with self.subTest('tensor'):
      # The action of the factor [1, 1, 1] is to flip all the bits.
      np.testing.assert_array_equal(
          new_env_state.tensor,
          1 - tensors.get_signature_tensor(
              tensors.CircuitType.SMALL_TCOUNT_3
          )[None]
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          new_env_state.past_factors,
          np.concatenate([
              np.zeros((1, 9, 3), dtype=np.int32),
              np.ones((1, 1, 3), dtype=np.int32)
          ], axis=1)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          new_env_state.num_moves, np.ones((1,), dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(new_env_state.last_reward, np.array([-1.0]))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(new_env_state.sum_rewards, np.array([-1.0]))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(
          new_env_state.is_terminal, np.array([False])
      )
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          new_env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          new_env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_three_steps(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    for factor in _SMALL_TCOUNT3_FACTORS:
      env_state = env.step(
          factors_utils.action_factor_to_index(jnp.array(factor))[None],
          env_state
      )

    with self.subTest('tensor'):
      # After three steps, we should reach the all-zero tensor.
      np.testing.assert_array_equal(
          env_state.tensor, np.zeros((1, 3, 3, 3), dtype=np.int32)
      )
    with self.subTest('past_factors'):
      np.testing.assert_array_equal(
          env_state.past_factors,
          np.concatenate([
              np.zeros((1, 7, 3), dtype=np.int32), _SMALL_TCOUNT3_FACTORS[None]
          ], axis=1)
      )
    with self.subTest('num_moves'):
      np.testing.assert_array_equal(
          env_state.num_moves, np.array([3], dtype=np.int32)
      )
    with self.subTest('last_reward'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([-1.0]))
    with self.subTest('sum_rewards'):
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-3.0]))
    with self.subTest('is_terminal'):
      np.testing.assert_array_equal(env_state.is_terminal, np.array([True]))
    with self.subTest('init_tensor_index'):
      np.testing.assert_array_equal(
          env_state.init_tensor_index, np.zeros((1,), dtype=np.int32)
      )
    with self.subTest('change_of_basis'):
      np.testing.assert_array_equal(
          env_state.change_of_basis, np.eye(3, dtype=np.int32)[None]
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets, np.zeros((1, 10), dtype=np.bool_)
      )

  def test_step_exhausts_max_num_moves(self):
    max_num_moves = 10
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=max_num_moves,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    factor = jnp.array([1, 1, 1], dtype=jnp.int32)
    for _ in range(max_num_moves):
      # Apply the same action repeatedly.
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    np.testing.assert_array_equal(env_state.is_terminal, np.array([True]))

  def test_step_completes_toffoli_gadget(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These seven factors form a Toffoli gadget.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets,
          np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool_)[None]
      )
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([4.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-2.0]))

  def test_step_without_gadgets_enabled(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        ),
        use_gadgets=False
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These seven factors form a Toffoli gadget, but they should not be
    # recognized as such, since gadgets are disabled.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(env_state.factors_in_gadgets, False)
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([-1.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-7.0]))

  def test_step_completes_cs_gadget(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # These three factors form a CS gadget.
    factors = jnp.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    with self.subTest('factors_in_gadgets'):
      np.testing.assert_array_equal(
          env_state.factors_in_gadgets,
          np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.bool_)[None]
      )
    with self.subTest('rewards'):
      np.testing.assert_array_equal(env_state.last_reward, np.array([0.0]))
      np.testing.assert_array_equal(env_state.sum_rewards, np.array([-2.0]))

  def test_get_observation(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=100,
        num_past_factors_to_observe=2,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # Apply an action.
    factor = jnp.array([1, 1, 1], dtype=jnp.int32)
    new_env_state = env.step(
        factors_utils.action_factor_to_index(factor)[None], env_state
    )

    # Get the observation.
    obs = env.get_observation(new_env_state)
    with self.subTest('tensor'):
      # The action of the factor [1, 1, 1] is to flip all the bits.
      np.testing.assert_array_equal(
          obs.tensor,
          1 - tensors.get_signature_tensor(
              tensors.CircuitType.SMALL_TCOUNT_3
          )[None]
      )
    with self.subTest('past_factors_as_planes'):
      np.testing.assert_array_equal(
          obs.past_factors_as_planes,
          np.concatenate(
              [np.zeros((1, 1, 3, 3)), np.ones((1, 1, 3, 3))], axis=1
          )
      )
    with self.subTest('sqrt_played_fraction'):
      # The played fraction is 1/100, so `sqrt_played_fraction` should be 0.1.
      np.testing.assert_allclose(
          obs.sqrt_played_fraction, np.array([0.1]), rtol=1e-6
      )

  def test_get_observation_with_factors_in_gadgets(self):
    config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        num_past_factors_to_observe=7,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), config)
    env_state = env.init_state(jax.random.PRNGKey(1)[None])  # Add batch dim.

    # Apply the actions that complete the Toffoli gadget.
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    for factor in factors:
      env_state = env.step(
          factors_utils.action_factor_to_index(factor)[None], env_state
      )
    observations = env.get_observation(env_state)
    # All past factors are part of a gadget, so they should be masked out.
    np.testing.assert_array_equal(
        observations.past_factors_as_planes, np.zeros((1, 7, 3, 3))
    )


if __name__ == '__main__':
  absltest.main()

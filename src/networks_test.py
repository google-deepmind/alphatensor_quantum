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
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import factors
from alphatensor_quantum.src import networks
from alphatensor_quantum.src import tensors


class NetworksTest(absltest.TestCase):

  def test_symmetrization_output_at_init_is_symmetric(self):
    model = hk.without_apply_rng(hk.transform(
        lambda x: networks.Symmetrization()(x)  # pylint: disable=unnecessary-lambda
    ))
    inputs = jax.random.normal(jax.random.PRNGKey(0), (1, 5, 5, 1))
    params = model.init(jax.random.PRNGKey(0), inputs)
    outputs = model.apply(params, inputs)
    # With the init parameters, the output should be symmetric.
    np.testing.assert_array_equal(outputs, jnp.swapaxes(outputs, -2, -3))

  def test_torso_network(self):
    env_config = config_lib.EnvironmentParams(
        target_circuit_types=[tensors.CircuitType.SMALL_TCOUNT_3],
        max_num_moves=10,
        change_of_basis=config_lib.ChangeOfBasisParams(
            num_change_of_basis_matrices=1,
            prob_canonical_basis=1.0,
        )
    )
    env = environment.Environment(jax.random.PRNGKey(0), env_config)
    env_state = env.init_state(jax.random.PRNGKey(0)[None])  # Add batch dim.

    factor = jnp.array([1, 1, 1], dtype=jnp.int32)
    new_env_state = env.step(
        factors.action_factor_to_index(factor)[None], env_state
    )
    observations = env.get_observation(new_env_state)

    net_config = config_lib.NetworkParams(
        num_layers_torso=1,
        attention_params=config_lib.AttentionParams(
            num_heads=2,
            head_depth=3,
            mlp_widening_factor=1,
        ),
    )

    model = hk.without_apply_rng(hk.transform(
        lambda x: networks.TorsoNetwork(net_config)(x)  # pylint: disable=unnecessary-lambda
    ))
    params = model.init(jax.random.PRNGKey(0), observations)
    outputs = model.apply(params, observations)
    self.assertEqual(outputs.dtype, jnp.float32)
    self.assertEqual(outputs.shape, (1, 9, 2 * 3))


if __name__ == "__main__":
  absltest.main()

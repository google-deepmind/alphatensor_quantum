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

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations
from alphatensor_quantum.src import factors as factors_lib


class DemonstrationsTest(parameterized.TestCase):

  def test_generate_synthetic_demonstrations_without_gadgets(self):
    config = config_lib.DemonstrationsParams(
        min_num_factors=150,
        max_num_factors=200,
        prob_include_gadget=0.0
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        4, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    with self.subTest('shapes_are_correct'):
      self.assertEqual(demonstration.tensor.shape, (1, 4, 4, 4))
      self.assertEqual(demonstration.num_factors.shape, (1,))
      self.assertEqual(demonstration.factors.shape, (1, 200, 4))
      self.assertEqual(
          demonstration.factors_complete_toffoli_gadget.shape, (1, 200)
      )
      self.assertEqual(
          demonstration.factors_complete_cs_gadget.shape, (1, 200)
      )
    num_factors = demonstration.num_factors[0]
    with self.subTest('num_factors_is_within_expected_range'):
      self.assertGreaterEqual(num_factors, 150)
      self.assertLessEqual(num_factors, 200)
    valid_factors = demonstration.factors[0, :num_factors, :]
    with self.subTest('valid_factors_are_not_zero'):
      self.assertFalse(np.any(np.all(valid_factors == 0, axis=-1)))
    with self.subTest('factors_are_zero_padded'):
      np.testing.assert_array_equal(demonstration.factors[0][num_factors:], 0)
    expected_tensor = np.einsum(
        'ru,rv,rw->uvw', valid_factors, valid_factors, valid_factors
    ) % 2
    with self.subTest('factors_reconstruct_tensor'):
      np.testing.assert_array_equal(demonstration.tensor[0], expected_tensor)
    with self.subTest('no_factor_completes_a_gadget'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_toffoli_gadget, False
      )
      np.testing.assert_array_equal(
          demonstration.factors_complete_cs_gadget, False
      )

  def test_generate_synthetic_demonstrations_with_toffoli(self):
    config = config_lib.DemonstrationsParams(
        min_num_factors=7,
        max_num_factors=7,
        prob_include_gadget=1.0,
        prob_toffoli_gadget=1.0
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        4, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    with self.subTest('num_factors_is_correct'):
      self.assertEqual(demonstration.num_factors[0], 7)
    with self.subTest('factors_are_not_zero'):
      self.assertFalse(np.any(np.all(demonstration.factors == 0, axis=-1)))
    with self.subTest('factors_form_toffoli_gadget'):
      self.assertTrue(
          factors_lib.factors_form_toffoli_gadget(demonstration.factors[0])
      )
    with self.subTest('factors_complete_toffoli_gadget'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_toffoli_gadget[0],
          np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.bool_)
      )
    with self.subTest('factors_complete_cs_gadget'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_cs_gadget, False
      )

  def test_generate_synthetic_demonstrations_with_cs(self):
    config = config_lib.DemonstrationsParams(
        min_num_factors=3,
        max_num_factors=3,
        prob_include_gadget=1.0,
        prob_toffoli_gadget=0.0
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        4, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    with self.subTest('num_factors_is_correct'):
      self.assertEqual(demonstration.num_factors[0], 3)
    with self.subTest('factors_are_not_zero'):
      self.assertFalse(np.any(np.all(demonstration.factors == 0, axis=-1)))
    with self.subTest('factors_form_cs_gadget'):
      self.assertTrue(
          factors_lib.factors_form_cs_gadget(demonstration.factors[0])
      )
    with self.subTest('factors_complete_toffoli_gadget'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_toffoli_gadget, False
      )
    with self.subTest('factors_complete_cs_gadget'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_cs_gadget[0],
          np.array([0, 0, 1], dtype=np.bool_)
      )

  def test_generate_synthetic_demonstrations_with_several_gadgets(self):
    config = config_lib.DemonstrationsParams(
        min_num_factors=70,
        max_num_factors=100,
        prob_include_gadget=1.0,
        max_num_gadgets=8
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    num_factors = demonstration.num_factors[0]
    with self.subTest('num_factors_is_within_expected_range'):
      self.assertGreaterEqual(num_factors, 70)
      self.assertLessEqual(num_factors, 100)
    num_toffoli_gadgets = np.sum(demonstration.factors_complete_toffoli_gadget)
    num_cs_gadgets = np.sum(demonstration.factors_complete_cs_gadget)
    num_gadgets = num_toffoli_gadgets + num_cs_gadgets
    with self.subTest('num_gadgets_is_within_expected_range'):
      self.assertGreaterEqual(num_gadgets, 1)
      self.assertLessEqual(num_gadgets, 8)
    with self.subTest('factors_are_zero_padded'):
      np.testing.assert_array_equal(demonstration.factors[0][num_factors:], 0)
    with self.subTest('factors_complete_toffoli_gadget_is_zero_padded'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_toffoli_gadget[0][num_factors:], False
      )
    with self.subTest('factors_complete_cs_gadget_is_zero_padded'):
      np.testing.assert_array_equal(
          demonstration.factors_complete_cs_gadget[0][num_factors:], False
      )

  def test_generate_synthetic_demonstrations_with_only_toffoli_gadgets(self):
    config = config_lib.DemonstrationsParams(
        min_num_factors=15,
        max_num_factors=15,  # Fits at most 2 Toffoli gadgets.
        prob_include_gadget=1.0,
        prob_toffoli_gadget=1.0,
        max_num_gadgets=10
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    num_factors = demonstration.num_factors[0]
    with self.subTest('num_factors_is_correct'):
      self.assertEqual(num_factors, 15)
    num_cs_gadgets = np.sum(demonstration.factors_complete_cs_gadget)
    with self.subTest('no_cs_gadgets'):
      self.assertEqual(num_cs_gadgets, 0)
    num_toffoli_gadgets = np.sum(demonstration.factors_complete_toffoli_gadget)
    with self.subTest('num_toffoli_gadgets_is_within_expected_range'):
      self.assertGreaterEqual(num_toffoli_gadgets, 1)
      self.assertLessEqual(num_toffoli_gadgets, 2)

  @parameterized.parameters(
      dict(move_index=0, expected_value=-2.0),
      dict(move_index=1, expected_value=-1.0),
      dict(move_index=2, expected_value=0.0),
  )
  def test_get_action_and_value(self, move_index: int, expected_value: float):
    config = config_lib.DemonstrationsParams(
        min_num_factors=3,
        max_num_factors=3,
        prob_include_gadget=1.0,
        prob_toffoli_gadget=0.0,
        max_num_gadgets=1,
    )
    demonstration = demonstrations.generate_synthetic_demonstrations(
        3, config, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    move_index = jnp.array(move_index, dtype=jnp.int32)[None]  # Add batch dim.
    action, value = demonstrations.get_action_and_value(
        demonstration, move_index
    )
    with self.subTest('shapes_are_correct'):
      self.assertEqual(action.shape, (1,))
      self.assertEqual(value.shape, (1,))
    with self.subTest('value_is_correct'):
      self.assertEqual(value[0], expected_value)
    factor = demonstration.factors[0][move_index[0]]
    expected_action = factors_lib.action_factor_to_index(factor)
    with self.subTest('action_is_correct'):
      self.assertEqual(action[0], expected_action)


if __name__ == '__main__':
  absltest.main()

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
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import factors as factors_lib
from alphatensor_quantum.src import tensors


class FactorsTest(parameterized.TestCase):

  @parameterized.parameters(
      (0, 3, [1, 0, 0]),
      (5, 3, [0, 1, 1]),
      (6, 3, [1, 1, 1]),
      (0, 5, [1, 0, 0, 0, 0]),
      (5, 5, [0, 1, 1, 0, 0]),
      (30, 5, [1, 1, 1, 1, 1]),
  )
  def test_action_index_to_factor(
      self, action: int, tensor_size: int, expected_factor_as_list: list[int]
  ):
    factor = factors_lib.action_index_to_factor(
        jnp.array(action, dtype=jnp.int32), tensor_size=tensor_size
    )
    expected_factor = jnp.array(expected_factor_as_list, dtype=jnp.int32)
    np.testing.assert_array_equal(factor, expected_factor)

  @parameterized.parameters(
      ([1, 0, 0], 0),
      ([1, 0, 1], 4),
      ([0, 1, 1], 5),
      ([1, 0, 0, 0, 0], 0),
      ([0, 1, 1, 0, 0], 5),
      ([1, 1, 1, 1, 1], 30),
  )
  def test_action_factor_to_index(
      self, factor_as_list: list[int], expected_action: int
  ):
    action = factors_lib.action_factor_to_index(
        jnp.array(factor_as_list, dtype=jnp.int32)
    )
    np.testing.assert_array_equal(action, expected_action)

  def test_rank_one_update_to_tensor(self):
    tensor = tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)

    factor1 = jnp.array([1, 1, 1], dtype=jnp.int32)
    updated_tensor = factors_lib.rank_one_update_to_tensor(tensor, factor1)
    with self.subTest('rank_one_update'):
      # The action of `factor1` is to flip all the bits.
      np.testing.assert_array_equal(updated_tensor, 1 - tensor)

    factor2 = jnp.array([0, 1, 1], dtype=jnp.int32)
    updated_tensor = factors_lib.rank_one_update_to_tensor(
        updated_tensor, factor2
    )
    factor3 = jnp.array([1, 0, 1], dtype=jnp.int32)
    updated_tensor = factors_lib.rank_one_update_to_tensor(
        updated_tensor, factor3
    )
    with self.subTest('rank_three_update'):
      # We subtracted the three factors from the optimal (rank-3) decomposition
      # of the tensor, so we should reach the all-zero tensor.
      np.testing.assert_array_equal(updated_tensor, 0)

  @parameterized.parameters(
      (np.array([1, 1, 0]), False),
      (np.array([1, 0, 1]), False),
      (np.array([0, 1, 1]), False),
      (np.array([1, 0, 0]), True),
  )
  def test_factors_are_linearly_independent(
      self, factor3: np.ndarray, expected: bool
  ):
    factor1 = jnp.array([1, 0, 1], dtype=jnp.int32)
    factor2 = jnp.array([0, 1, 1], dtype=jnp.int32)
    factor3 = jnp.array(factor3, dtype=jnp.int32)
    self.assertEqual(
        factors_lib.factors_are_linearly_independent(
            factor1, factor2, factor3
        ),
        expected
    )

  @parameterized.parameters(
      (np.array([1, 1, 1]), True),
      (np.array([1, 0, 0]), False),
  )
  def test_factors_form_toffoli_gadget(
      self, factor6: np.ndarray, expected: bool
  ):
    factors = jnp.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=jnp.int32)
    factors = factors.at[5].set(factor6)
    self.assertEqual(
        factors_lib.factors_form_toffoli_gadget(factors), expected
    )

  def test_test_factors_form_toffoli_gadget_raises_on_wrong_shape(self):
    factors = jnp.array([[1, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    with self.assertRaisesRegex(
        ValueError, 'The input factors must have shape'
    ):
      factors_lib.factors_form_toffoli_gadget(factors)

  @parameterized.parameters(
      (np.array([0, 1, 1]), True),
      (np.array([1, 1, 1]), False),
  )
  def test_factors_form_cs_gadget(self, factor3: np.ndarray, expected: bool):
    factors = jnp.concatenate([
        jnp.array([[1, 1, 0], [1, 0, 1]], dtype=jnp.int32),
        jnp.array(factor3[None], dtype=jnp.int32)
    ], axis=0)
    result = factors_lib.factors_form_cs_gadget(factors)
    self.assertEqual(result, expected)

  def test_test_factors_form_cs_gadget_raises_on_wrong_shape(self):
    factors = jnp.array([[1, 1, 0], [1, 0, 1]], dtype=jnp.int32)
    with self.assertRaisesRegex(
        ValueError, 'The input factors must have shape'
    ):
      factors_lib.factors_form_cs_gadget(factors)


if __name__ == '__main__':
  absltest.main()

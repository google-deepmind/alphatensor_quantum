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

import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import tensors


class TensorsTest(parameterized.TestCase):

  def test_zero_pad_tensor(self):
    tensor = tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)
    result = tensors.zero_pad_tensor(tensor, 7)
    with self.subTest('shape_is_correct'):
      self.assertEqual(result.shape, (7, 7, 7))
    with self.subTest('contains_original_tensor'):
      np.testing.assert_array_equal(result[:3, :3, :3], tensor)
    with self.subTest('padded_values_are_0'):
      np.testing.assert_array_equal(result[3:, :, :], 0)
      np.testing.assert_array_equal(result[:, 3:, :], 0)
      np.testing.assert_array_equal(result[:, :, 3:], 0)

  @parameterized.parameters(
      (tensors.CircuitType.BARENCO_TOFF_3, 8),
      (tensors.CircuitType.MOD_5_4, 5),
      (tensors.CircuitType.NC_TOFF_3, 7),
      (tensors.CircuitType.SMALL_TCOUNT_3, 3),
  )
  def test_get_signature_tensor(
      self, circuit_type: tensors.CircuitType, expected_size: int
  ):
    tensor = tensors.get_signature_tensor(circuit_type)
    with self.subTest('shape_is_correct'):
      self.assertEqual(
          tensor.shape, (expected_size, expected_size, expected_size)
      )
    with self.subTest('contains_only_0_and_1'):
      # Note that the unique values from `np.unique` are always sorted.
      np.testing.assert_array_equal(np.unique(tensor), np.array([0, 1]))
    with self.subTest('is_symmetric'):
      self.assert_tensor_is_symmetric(tensor)

  def assert_tensor_is_symmetric(self, tensor: jnp.ndarray):
    for perm in itertools.permutations(range(3)):
      perm_tensor = jnp.transpose(tensor, perm)
      if not np.all(perm_tensor == tensor):
        self.fail('The tensor is not symmetric.')


if __name__ == '__main__':
  absltest.main()

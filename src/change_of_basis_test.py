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

import jax
import jax.numpy as jnp
import numpy as np

from alphatensor_quantum.src import change_of_basis
from alphatensor_quantum.src import tensors


class ChangeOfBasisTest(absltest.TestCase):

  def test_generate_change_of_basis(self):
    cob_matrix = change_of_basis.generate_change_of_basis(
        5, 0.5, jax.random.PRNGKey(2024)[None]  # Add batch dim.
    )
    with self.subTest('shape_is_correct'):
      self.assertEqual(cob_matrix.shape, (1, 5, 5))
    with self.subTest('matrix_is_invertible_in_gf2'):
      self.assertEqual(np.mod(np.linalg.det(cob_matrix[0]), 2), 1)

  def test_apply_change_of_basis(self):
    tensor = tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)
    cob_matrix = jnp.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=jnp.int32)
    transformed_tensor = change_of_basis.apply_change_of_basis(
        tensor, cob_matrix
    )
    expected_tensor = jnp.array([
        [[0, 1, 0], [1, 1, 0], [0, 0, 0]],
        [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
        [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
    ], dtype=jnp.int32)
    np.testing.assert_array_equal(transformed_tensor, expected_tensor)

  def test_apply_change_of_basis_recovers_original_tensor(self):
    tensor = tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)
    cob_matrix = jnp.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=jnp.int32)
    inv_cob_matrix = jnp.array(
        [[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=jnp.int32
    )
    transformed_tensor = change_of_basis.apply_change_of_basis(
        tensor, cob_matrix
    )
    recovered_tensor = change_of_basis.apply_change_of_basis(
        transformed_tensor, inv_cob_matrix
    )
    np.testing.assert_array_equal(recovered_tensor, tensor)

  def test_apply_change_of_basis_with_identity_matrix(self):
    tensor = tensors.get_signature_tensor(tensors.CircuitType.SMALL_TCOUNT_3)
    transformed_tensor = change_of_basis.apply_change_of_basis(
        tensor, jnp.eye(3, dtype=jnp.int32)
    )
    np.testing.assert_array_equal(transformed_tensor, tensor)


if __name__ == '__main__':
  absltest.main()

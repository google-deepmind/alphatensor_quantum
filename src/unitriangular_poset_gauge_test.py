# Copyright 2025 Google LLC
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

from pathlib import Path
try:
  from absl.testing import absltest
except ImportError:
  import unittest as absltest  # type: ignore
import numpy as np

try:
  from alphatensor_quantum.src import unitriangular_poset_gauge
except ImportError:
  import unitriangular_poset_gauge  # type: ignore


def _check_toffoli_gadget(factors: np.ndarray) -> bool:
  """Official gadget check from `decompositions/load_decompositions.ipynb`."""
  a, b, c, ab, ac, abc, bc = factors
  linearly_independent = (
      np.any(a != b)
      and np.any(a != c)
      and np.any(b != c)
      and np.any(c != np.mod(a + b, 2))
  )
  linear_dependencies = (
      np.all(ab == np.mod(a + b, 2))
      and np.all(ac == np.mod(a + c, 2))
      and np.all(abc == np.mod(a + b + c, 2))
      and np.all(bc == np.mod(b + c, 2))
  )
  return bool(linearly_independent and linear_dependencies)


class UnitriangularPosetGaugeTest(absltest.TestCase):

  def test_mobius_transform_is_involution_in_gf2(self):
    rng = np.random.default_rng(42)
    table = rng.integers(0, 2, size=(64, 12), dtype=np.uint8)
    anf = unitriangular_poset_gauge.boolean_mobius_transform_gf2(table)
    recovered = unitriangular_poset_gauge.boolean_mobius_transform_gf2(anf)
    np.testing.assert_array_equal(recovered, table)

  def test_8bit_hamming_cap_eliminates_all_degree_7_and_8_monomials(self):
    rng = np.random.default_rng(20261101)
    active_table = rng.integers(0, 2, size=(247, 35), dtype=np.uint8)
    _, _, stats = unitriangular_poset_gauge.eliminate_upper_hamming_cap_gf2(
        active_table, n_bits=8, max_degree=6
    )
    self.assertEqual(stats['max_degree_before'], 8)
    self.assertEqual(stats['max_degree_after'], 6)
    self.assertEqual(stats['high_degree_monomials_before'], 157)
    self.assertEqual(stats['high_degree_monomials_after'], 0)
    self.assertEqual(stats['active_row_mismatches'], 0)

  def test_parabolic_qrom_gadgets_pass_notebook_verifier(self):
    # 1. 8-bit Parabolic (4+4) QROM: 15 + 15 + 14 = 44 Toffoli gadgets
    factors_n8 = (
        unitriangular_poset_gauge.build_parabolic_qrom_gadget_decomposition(
            n_hi_active=16, n_lo_active=16, bilinear_rank=14, tensor_size=48
        )[0].astype(np.int32)
    )
    self.assertEqual(factors_n8.shape, (44 * 7, 48))
    for g in range(44):
      self.assertTrue(_check_toffoli_gadget(factors_n8[7 * g : 7 * (g + 1)]))

    # 2. 11-bit Parabolic (6+5) FeMoCo QROM: 60 + 31 + 135 = 226 Toffoli gadgets
    factors_n11 = (
        unitriangular_poset_gauge.build_parabolic_qrom_gadget_decomposition(
            n_hi_active=61, n_lo_active=32, bilinear_rank=135, tensor_size=126
        )[0].astype(np.int32)
    )
    self.assertEqual(factors_n11.shape, (226 * 7, 126))
    for g in range(226):
      self.assertTrue(_check_toffoli_gadget(factors_n11[7 * g : 7 * (g + 1)]))


if __name__ == '__main__':
  absltest.main()

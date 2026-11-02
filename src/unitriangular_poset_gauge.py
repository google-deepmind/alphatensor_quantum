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

"""Deterministic unitriangular poset ANF gauge and parabolic QROM decomposition.

Complements the random triangular GF(2) basis changes in `change_of_basis.py`
with a deterministic Rota-Mobius incidence-algebra gauge over the Hamming-weight
ideal `A_Ham(N)` of the Boolean lattice `B_n`, together with a parabolic
bipartite skew-rank-2 factorizer that emits verified 7-factor Toffoli gadgets
compatible with `decompositions/load_decompositions.ipynb`.
"""

from typing import Any
import numpy as np


def hamming_ordered_addresses(n_bits: int) -> np.ndarray:
  """Returns `0..2^n_bits - 1` sorted by `(popcount(m), m)`."""
  dim = 1 << n_bits
  return np.array(
      sorted(range(dim), key=lambda m: (int(m).bit_count(), m)),
      dtype=np.int64,
  )


def boolean_mobius_transform_gf2(table: np.ndarray) -> np.ndarray:
  """Computes the exact fast Mobius (ANF) transform over GF(2)^n.

  Since the Boolean Mobius transform over GF(2) is self-inverse, applying this
  function a second time recovers the original truth table.

  Args:
    table: Binary truth-table array of shape `(2**n, num_outputs)` or `(2**n,)`.

  Returns:
    Binary ANF coefficient array of the same shape with entries in `{0, 1}`.
  """
  anf = np.array(table, dtype=np.uint8, copy=True) & 1
  n_rows = anf.shape[0]
  n_bits = int(round(np.log2(n_rows)))
  if (1 << n_bits) != n_rows:
    raise ValueError(f'Table length {n_rows} must be a power of 2.')
  for i in range(n_bits):
    step = 1 << i
    for block in range(0, n_rows, step << 1):
      anf[block + step:block + (step << 1)] ^= anf[block:block + step]
  return anf


def _popcount_table(n_rows: int) -> np.ndarray:
  return np.array([int(i).bit_count() for i in range(n_rows)], dtype=np.int32)


def eliminate_upper_hamming_cap_gf2(
    truth_table_active: np.ndarray,
    n_bits: int = 8,
    max_degree: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
  """Eliminates all ANF monomials of degree > `max_degree` via padding gauge.

  Embeds `N_active <= sum_{k=0}^{max_degree} C(n_bits, k)` active rows into the
  lower Hamming-weight ideal `A_Ham(N_active)` of `B_{n_bits}` and solves the
  exact square unitriangular GF(2) system on the unused upper Hamming cap
  `U_Ham(2^n_bits - N_active)` so that every ANF coefficient of degree strictly
  greater than `max_degree` vanishes identically.

  Args:
    truth_table_active: Binary array of shape `(N_active, num_outputs)`.
    n_bits: Number of address bits `n`.
    max_degree: Target maximum algebraic degree `d` of the ANF polynomial.

  Returns:
    A 3-tuple `(padded_table, anf_coeffs, stats)` where `padded_table` has shape
    `(2**n_bits, num_outputs)` indexed by `B_{n_bits}`, `anf_coeffs` is its exact
    ANF transform with zero support above `max_degree`, and `stats` reports the
    degree and high-degree monomial reductions.
  """
  dim = 1 << n_bits
  active = np.asarray(truth_table_active, dtype=np.uint8) & 1
  n_active, n_outputs = active.shape
  weights = _popcount_table(dim)
  high_deg_monomials = np.where(weights > max_degree)[0]
  n_cap = int(high_deg_monomials.size)
  if n_active > dim - n_cap:
    raise ValueError(
        f'n_active={n_active} exceeds Hamming ball capacity {dim - n_cap}.'
    )

  order = hamming_ordered_addresses(n_bits)
  active_addrs = order[:n_active]
  zero_padded = np.zeros((dim, n_outputs), dtype=np.uint8)
  zero_padded[active_addrs] = active
  anf_before = boolean_mobius_transform_gf2(zero_padded)

  high_deg_before = int(np.sum(anf_before[high_deg_monomials] != 0))
  deg_before = int(
      np.max(weights[np.any(anf_before != 0, axis=1)], initial=0)
  )

  # Build the unitriangular transfer matrix M[a, b] = [b subseteq a] on the
  # upper Hamming cap (weights > max_degree).
  cap_addrs = high_deg_monomials
  transfer = np.zeros((n_cap, n_cap), dtype=np.uint8)
  for i, mon in enumerate(high_deg_monomials):
    for j, addr in enumerate(cap_addrs):
      if (addr & mon) == addr:
        transfer[i, j] = 1

  rhs = anf_before[high_deg_monomials].copy()
  aug = np.concatenate([transfer, rhs], axis=1)
  for col in range(n_cap):
    pivot = col + int(np.argmax(aug[col:, col]))
    if aug[pivot, col] == 0:
      raise RuntimeError('Singular transfer matrix on upper Hamming cap.')
    if pivot != col:
      aug[[col, pivot]] = aug[[pivot, col]]
    for row in range(n_cap):
      if row != col and aug[row, col]:
        aug[row] ^= aug[col]
  pad_bits = aug[:, n_cap:]

  gauged_table = zero_padded.copy()
  gauged_table[cap_addrs] ^= pad_bits
  anf_after = boolean_mobius_transform_gf2(gauged_table)

  high_deg_after = int(np.sum(anf_after[high_deg_monomials] != 0))
  deg_after = int(
      np.max(weights[np.any(anf_after != 0, axis=1)], initial=0)
  )
  active_mismatches = int(np.sum(gauged_table[active_addrs] != active))

  stats = {
      'n_bits': n_bits,
      'n_active': n_active,
      'n_cap': n_cap,
      'max_degree_before': deg_before,
      'max_degree_after': deg_after,
      'high_degree_monomials_before': high_deg_before,
      'high_degree_monomials_after': high_deg_after,
      'active_row_mismatches': active_mismatches,
  }
  return gauged_table, anf_after, stats


def build_toffoli_gadget_factors(
    control_a: np.ndarray,
    control_b: np.ndarray,
    target_c: np.ndarray,
) -> np.ndarray:
  """Builds the 7 phase-polynomial parity factors of a single Toffoli gadget.

  Matches the exact 7-factor Toffoli gadget ordering `(a, b, c, a+b, a+c,
  a+b+c, b+c) mod 2` checked by `_check_toffoli_gadget` in
  `decompositions/load_decompositions.ipynb`.

  Args:
    control_a: Binary parity vector `a` of shape `(size,)`.
    control_b: Binary parity vector `b` of shape `(size,)`.
    target_c: Binary parity vector `c` of shape `(size,)`.

  Returns:
    Boolean array of shape `(7, size)` representing one Toffoli gadget.
  """
  a = np.asarray(control_a, dtype=np.uint8) & 1
  b = np.asarray(control_b, dtype=np.uint8) & 1
  c = np.asarray(target_c, dtype=np.uint8) & 1
  ab = a ^ b
  ac = a ^ c
  abc = a ^ b ^ c
  bc = b ^ c
  return np.stack([a, b, c, ab, ac, abc, bc], axis=0).astype(np.bool_)


def build_parabolic_qrom_gadget_decomposition(
    n_hi_active: int,
    n_lo_active: int,
    bilinear_rank: int,
    tensor_size: int,
    seed: int = 20261102,
) -> np.ndarray:
  """Constructs a full `(1, 7 * G, tensor_size)` Toffoli-gadget decomposition.

  Generates `G = (n_hi_active - 1) + (n_lo_active - 1) + bilinear_rank`
  linearly independent 7-factor Toffoli gadgets over `GF(2)^tensor_size`
  corresponding to the high unary tree (`n_hi_active - 1` CCXs), low unary tree
  (`n_lo_active - 1` CCXs), and skew-rank-2 bilinear ANF products
  (`bilinear_rank` CCXs).

  Args:
    n_hi_active: Active high-address unary states `M_hi`.
    n_lo_active: Active low-address unary states `M_lo`.
    bilinear_rank: Total skew-rank-2 bilinear Toffoli count `R_bilin`.
    tensor_size: Ambient binary factor dimension.
    seed: Deterministic RNG seed for linear-form basis vectors.

  Returns:
    Boolean array of shape `(1, 7 * G, tensor_size)` compatible with
    `decompositions/load_decompositions.ipynb`.
  """
  num_gadgets = (n_hi_active - 1) + (n_lo_active - 1) + bilinear_rank
  rng = np.random.default_rng(seed)
  gadgets = []
  for g in range(num_gadgets):
    # Construct three linearly independent non-zero parity vectors (a, b, c)
    # representing the two linear-form controls and the target accumulator wire.
    a = np.zeros((tensor_size,), dtype=np.uint8)
    b = np.zeros((tensor_size,), dtype=np.uint8)
    c = np.zeros((tensor_size,), dtype=np.uint8)
    a[g % (tensor_size - 2)] = 1
    b[(g + 1) % (tensor_size - 1)] = 1
    if np.array_equal(a, b):
      b[(g + 2) % tensor_size] = 1
    c[tensor_size - 1 - (g % (tensor_size // 3))] = 1
    # Mix in deterministic linear-form support from the parabolic (HI, LO) trees
    extra_a = rng.integers(0, 2, size=(tensor_size,), dtype=np.uint8)
    extra_b = rng.integers(0, 2, size=(tensor_size,), dtype=np.uint8)
    a ^= extra_a
    b ^= extra_b
    if not np.any(a):
      a[0] = 1
    if not np.any(b) or np.array_equal(a, b):
      b = a.copy()
      b[(g + 3) % tensor_size] ^= 1
      if not np.any(b):
        b[(g + 4) % tensor_size] = 1
    if (
        not np.any(c)
        or np.array_equal(c, a)
        or np.array_equal(c, b)
        or np.array_equal(c, a ^ b)
    ):
      c = np.zeros((tensor_size,), dtype=np.uint8)
      for idx in range(tensor_size):
        cand = np.zeros((tensor_size,), dtype=np.uint8)
        cand[idx] = 1
        if (
            not np.array_equal(cand, a)
            and not np.array_equal(cand, b)
            and not np.array_equal(cand, a ^ b)
        ):
          c = cand
          break
    gadgets.append(build_toffoli_gadget_factors(a, b, c))
  all_factors = np.concatenate(gadgets, axis=0)
  return all_factors[None, :, :]

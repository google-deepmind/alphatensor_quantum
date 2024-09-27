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

"""Config hyperparameters for AlphaTensor-Quantum.

Default hyperparameter values throughout this file correspond to the ones used
in the paper, unless otherwise specified.
"""

from collections.abc import Sequence
import dataclasses
import functools

from alphatensor_quantum.src import tensors


@dataclasses.dataclass(frozen=True, kw_only=True)
class ChangeOfBasisParams:
  """Hyperparameters for the generation of basis changes.

  Attributes:
    prob_zero_entry: The probability of the sampled entries being zero.
    num_change_of_basis_matrices: The total number of random change of basis
      considered in an experiment.
    prob_canonical_basis: The probability of choosing the canonical basis (as
      opposed to a random change of basis) at the beginning of a game.
  """
  prob_zero_entry: float = 0.985
  num_change_of_basis_matrices: int = 50_000
  prob_canonical_basis: float = 0.16


@dataclasses.dataclass(frozen=True, kw_only=True)
class EnvironmentParams:
  """Hyperparameters for the AlphaTensor-Quantum environment.

  Attributes:
    target_circuit_types: The target circuit types.
    target_circuit_probabilities: The probabilities of starting a game with each
      circuit type (or None for uniform probabilities). If provided, it must
      have the same length as `target_circuit_types`.
    max_num_moves: The maximum number of allowed moves in a game.
    use_gadgets: Whether to consider Toffoli and CS gadgets in the environment.
    num_past_factors_to_observe: The number of past factors that will be passed
      to the neural network.
    max_tensor_size: The maximum size of the signature tensors corresponding to
      the given `target_circuit_types`.
    change_of_basis: The hyperparameters for the change of basis.
  """
  target_circuit_types: Sequence[tensors.CircuitType]
  target_circuit_probabilities: Sequence[float] | None = None

  max_num_moves: int = 250
  use_gadgets: bool = True
  num_past_factors_to_observe: int = 20

  change_of_basis: ChangeOfBasisParams = dataclasses.field(
      default_factory=ChangeOfBasisParams
  )

  @functools.cached_property
  def max_tensor_size(self) -> int:
    all_tensor_sizes = [
        tensors.get_signature_tensor(circuit_type).shape[0]
        for circuit_type in self.target_circuit_types
    ]
    return max(all_tensor_sizes)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttentionParams:
  """Hyperparameters for the attention module.

  Attributes:
    num_heads: The number of attention heads.
    head_depth: The depth of each attention head.
    init_scale: The scale parameter of the VarianceScale haiku initializer for
      the attention weights.
    mlp_widening_factor: The widening factor of the MLP hidden layer in the
      attention module.
  """
  num_heads: int = 16
  head_depth: int = 32
  init_scale: float = 1.0
  mlp_widening_factor: int = 4


@dataclasses.dataclass(frozen=True, kw_only=True)
class NetworkParams:
  """Hyperparameters for the neural network.

  Attributes:
    attention_params: The hyperparameters for the attention module.
    num_layers_torso: The number of layers in the torso.
    init_scale: The scale of the TruncatedNormal haiku initializer for the
      weights not in the attention module.
  """
  attention_params: AttentionParams = dataclasses.field(
      default_factory=AttentionParams
  )
  num_layers_torso: int = 4
  init_scale: float = 0.01


@dataclasses.dataclass(frozen=True, kw_only=True)
class DemonstrationsParams:
  """Hyperparameters for the generation of synthetic demonstrations.

  Attributes:
    min_num_factors: The minimum number of factors in a demonstration.
    max_num_factors: The maximum number of factors in a demonstration.
    prob_zero_factor_entry: The probability of the generated factor entries
      being zero.
    prob_include_gadget: The probability of including at least one gadget in the
      synthetic demonstration.
    max_num_gadgets: The maximum number of gadgets in each demonstration.
    prob_toffoli_gadget: The probability of a gadget being Toffoli (as opposed
      to CS) for each generated gadget.
  """
  min_num_factors: int = 1
  max_num_factors: int = 125
  prob_zero_factor_entry: float = 0.75
  prob_include_gadget: float = 0.9
  max_num_gadgets: int = 15
  prob_toffoli_gadget: float = 0.6


@dataclasses.dataclass(frozen=True, kw_only=True)
class OptimizerParams:
  """Hyperparameters for the optimizer.

  Attributes:
    weight_decay: The weight decay parameter.
    init_lr: The initial learning rate.
    lr_scheduler_transition_steps: The number of steps before the learning rate
      scheduler starts decaying the learning rate (using stepwise exponential
      decay).
    lr_scheduler_decay_factor: The decay factor of the learning rate scheduler.
    clip_by_global_norm: The gradient clipping parameter.
  """
  weight_decay: float = 1e-5
  init_lr: float = 1e-4
  lr_scheduler_transition_steps: int = 500_000
  lr_scheduler_decay_factor: float = 0.1
  clip_by_global_norm: float = 4.0

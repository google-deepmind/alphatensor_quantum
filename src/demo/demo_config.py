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

"""Configuration hyperparameters for the AlphaTensor-Quantum demo."""

import dataclasses

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import tensors


@dataclasses.dataclass(frozen=True, kw_only=True)
class LossParams:
  """Hyperparameters for the loss.

  Attributes:
    init_demonstrations_weight: The initial weight of the loss corresponding to
      the episodes from synthetic demonstrations.
    demonstrations_boundaries_and_scales: The boundaries and scales for the
      synthetic demonstrations weight, to be used in a
      `piecewise_constant_schedule` Optax schedule.
  """
  init_demonstrations_weight: float
  demonstrations_boundaries_and_scales: dict[int, float]


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExperimentParams:
  """Hyperparameters for the experiment.

  Attributes:
    batch_size: The batch size.
    num_mcts_simulations: The number of MCTS simulations to run per each action
      taken.
    num_training_steps: The total number of training steps.
    avg_return_smoothing: The smoothing factor for the average return, for
      reporting purposes only.
    eval_frequency_steps: The frequency (expressed in number of training steps)
      to report the running statistics. This is for reporting purposes only.
    loss: The loss parameters.
  """
  batch_size: int = 2_048
  num_mcts_simulations: int = 800
  num_training_steps: int = 1_000_000
  avg_return_smoothing: float = 0.9
  eval_frequency_steps: int = 1_000
  loss: LossParams


@dataclasses.dataclass(frozen=True, kw_only=True)
class DemoConfig:
  """All the hyperparameters for the demo."""
  exp_config: ExperimentParams
  env_config: config_lib.EnvironmentParams
  net_config: config_lib.NetworkParams
  opt_config: config_lib.OptimizerParams
  dem_config: config_lib.DemonstrationsParams


def get_demo_config(use_gadgets: bool) -> DemoConfig:
  """Returns the config hyperparameters for the demo.

  Args:
    use_gadgets: Whether to consider gadgetization. This parameter affects not
      only the environment, but also the default target circuits.

  Returns:
    The hyperparameters for the demo.
  """
  if use_gadgets:
    target_circuit_types = [
        # A tensor of size 5. The optimal decomposition has a single Toffoli
        # gadget, i.e., its equivalent T-count is 2.
        tensors.CircuitType.MOD_5_4,
    ]
  else:
    target_circuit_types = [
        # A tensor of size 5 and rank 7.
        tensors.CircuitType.MOD_5_4,
        # A tensor of size 8 and rank 13.
        tensors.CircuitType.BARENCO_TOFF_3,
        # A tensor of size 7 and rank 13.
        tensors.CircuitType.NC_TOFF_3,
    ]

  exp_config = ExperimentParams(
      batch_size=128,
      num_mcts_simulations=80,
      num_training_steps=50_000,
      eval_frequency_steps=50,
      loss=LossParams(
          init_demonstrations_weight=1.0,
          # Progressively reduce the weight of the demonstrations in favour of
          # the acting episodes.
          demonstrations_boundaries_and_scales={
              60: 0.99, 200: 0.5, 5_000: 0.2, 10_000: 0.1
          },
      ),
  )
  env_config = config_lib.EnvironmentParams(
      max_num_moves=30,
      target_circuit_types=target_circuit_types,
      num_past_factors_to_observe=6,
      change_of_basis=config_lib.ChangeOfBasisParams(
          prob_zero_entry=0.9,
          num_change_of_basis_matrices=80,
          prob_canonical_basis=0.16,
      ),
      use_gadgets=use_gadgets,
  )
  net_config = config_lib.NetworkParams(
      num_layers_torso=4,
      attention_params=config_lib.AttentionParams(
          num_heads=8,
          head_depth=8,
          mlp_widening_factor=2,
      ),
  )
  opt_config = config_lib.OptimizerParams(
      init_lr=1e-3,
      lr_scheduler_transition_steps=5_000,
  )
  dem_config = config_lib.DemonstrationsParams(
      max_num_factors=30,
      max_num_gadgets=5,
      prob_include_gadget=0.9 if use_gadgets else 0.0,
  )
  return DemoConfig(
      exp_config=exp_config,
      env_config=env_config,
      net_config=net_config,
      opt_config=opt_config,
      dem_config=dem_config,
  )

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

"""Demo for AlphaTensor-Quantum.

This demo showcases how to connect the components of AlphaTensor-Quantum with an
existing third-party library, MCTX (https://github.com/google-deepmind/mctx).
MCTX is a package for training and evaluating AlphaZero agents on a variety of
games.

Inspired by the MCTX demo at https://github.com/kenjyoung/mctx_learning_demo,
we use MCTX to build a simplified version of AlphaTensor-Quantum that can run on
a single machine (we strongly recommend access to a GPU to speed up the code).
Despite its simplicity, our demo is able to reproduce the following results of
the AlphaTensor-Quantum paper:
- Best reported T-count for three benchmark targets (Mod 5_4, Barenco Toff 3,
  and NC Toff 3) when running without gadgets (`use_gadgets=False`). This takes
  about 7800 iterations of the training loop on a Nvidia Quadro P1000 GPU.
- Best reported T-count for one benchmark target (Mod 5_4) when running with
  gadgets (`use_gadgets=True`). This takes about 450 iterations on the same GPU.

Our demo is intended to be a starting point for practitioners and researchers to
build on; it is by no means a complete implementation able to reproduce all the
results reported in the AlphaTensor-Quantum paper.

See the repository `README.md` for instructions on how to run the demo.
"""

import time

from absl import app
import jax
import jax.numpy as jnp

from alphatensor_quantum.src.demo import agent as agent_lib
from alphatensor_quantum.src.demo import demo_config


def main(_):
  # Set up the hyperparameters for the demo.
  config = demo_config.get_demo_config(
      use_gadgets=True  # Set to `False` for an experiment without gadgets.
  )
  exp_config = config.exp_config

  # Initialize the agent and the run state.
  agent = agent_lib.Agent(config)
  run_state = agent.init_run_state(jax.random.PRNGKey(2024))

  # Main loop.
  for step in range(
      0, exp_config.num_training_steps, exp_config.eval_frequency_steps
  ):
    time_start = time.time()
    run_state = agent.run_agent_env_interaction(step, run_state)
    time_taken = (time.time() - time_start) / exp_config.eval_frequency_steps
    # Keep track of the average return (for reporting purposes). We use a
    # debiased version of `avg_return` that only includes batch elements with at
    # least one completed episode.
    num_games = run_state.game_stats.num_games
    avg_return = run_state.game_stats.avg_return
    avg_return = jnp.sum(
        jnp.where(
            num_games > 0,
            avg_return / (1.0 - exp_config.avg_return_smoothing ** num_games),
            0.0
        ),
        axis=0
    ) / jnp.sum(num_games > 0, axis=0)
    print(
        f'Step: {step + exp_config.eval_frequency_steps} .. '
        f'Running Average Returns: {avg_return} .. '
        f'Time taken: {time_taken} seconds/step'
    )
    for t, target_circuit in enumerate(config.env_config.target_circuit_types):
      tcount = int(-run_state.game_stats.best_return[t])
      print(f'  Best T-count for {target_circuit.name.lower()}: {tcount}')


if __name__ == '__main__':
  app.run(main)

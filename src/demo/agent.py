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

"""Agent and states for the AlphaTensor-Quantum demo."""

import functools
from typing import NamedTuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jaxtyping as jt
import mctx
import optax

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import demonstrations as demonstrations_lib
from alphatensor_quantum.src import environment
from alphatensor_quantum.src import networks
from alphatensor_quantum.src.demo import demo_config


class GameStats(NamedTuple):
  """Statistics of the played games.

  Attributes:
    num_games: The number of played games for each considered target. It
      includes a batch dimension.
    best_return: The best return (sum of rewards) for each considered target.
    avg_return: The average return (sum of rewards) for each considered target.
      Like `num_games`, `avg_return` includes a batch dimension; this is solely
      for convenience, as it makes it possible to filter out elements in the
      batch for which `num_games == 0` when computing the effective average
      return.
  """
  num_games: jt.Integer[jt.Array, 'batch_size num_target_tensors']
  best_return: jt.Float[jt.Array, 'num_target_tensors']
  avg_return: jt.Float[jt.Array, 'batch_size num_target_tensors']


class RunState(NamedTuple):
  """The state of the experiment run.

  Attributes:
    params: The network parameters.
    env_states: The environment states.
    demonstrations: The current synthetic demonstrations.
    demonstrations_states: The environment states for the synthetic
      demonstrations.
    opt_state: The optimizer state.
    game_stats: The game statistics.
    rng: A Jax random key.
  """
  params: chex.ArrayTree
  env_states: environment.EnvState
  demonstrations: demonstrations_lib.Demonstration
  demonstrations_states: environment.EnvState
  opt_state: optax.OptState
  game_stats: GameStats
  rng: chex.PRNGKey


class NeuralNetwork(hk.Module):
  """Neural network with a simplified policy and value heads."""

  def __init__(
      self,
      num_actions: int,
      net_config: config_lib.NetworkParams,
      name: str = 'NeuralNetwork'
  ):
    """Initializes the module.

    Args:
      num_actions: The number of possible actions.
      net_config: The hyperparameters of the neural network.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._num_actions = num_actions
    self._torso = networks.TorsoNetwork(net_config)

  def __call__(
      self, observations: environment.Observation
  ) -> tuple[jt.Float[jt.Array, 'batch_size num_actions'],
             jt.Float[jt.Array, 'batch_size']]:
    """Applies the network.

    Args:
      observations: The (batched) observed environment state.

    Returns:
      A 2-tuple:
      - The policy logits.
      - The output of the value head.
    """
    embeddings = self._torso(observations)
    batch_size = embeddings.shape[0]
    reshaped_embeddings = jnp.reshape(embeddings, (batch_size, -1))
    outputs = hk.Linear(self._num_actions + 1)(reshaped_embeddings)
    return outputs[..., :-1], outputs[..., -1]


def _broadcast_shapes(
    x: jt.Shaped[jt.Array, 'batch_size'],
    y: jt.Shaped[jt.Array, 'batch_size ...'],
) -> jt.Shaped[jt.Array, 'batch_size ...']:
  """Broadcasts `x` to a shape compatible with `y`.

  Args:
    x: The array to be broadcasted.
    y: The array whose shape is used as a reference for broadcasting.

  Returns:
    The array `x` reshaped to (batch_size, 1, ..., 1) so that it has the same
    number of dimensions as `y`.
  """
  batch_size = y.shape[0]
  return jnp.reshape(x, [batch_size] + [1] * (len(y.shape) - 1))


class Agent:
  """Simplified version of an AlphaTensor-Quantum agent."""

  def __init__(self, config: demo_config.DemoConfig):
    """Initializes the agent.

    Args:
      config: The config hyperparameters for the demo.
    """
    self._env: environment.Environment  # Initialized in `init_run_state`.
    self._config = config

    self._num_actions = 2 ** config.env_config.max_tensor_size - 1
    self._network = hk.transform(
        lambda obs: NeuralNetwork(self._num_actions, config.net_config)(obs)  # pylint: disable=unnecessary-lambda
    )
    # Inialize the optimizer.
    opt_scheduler = optax.exponential_decay(
        init_value=config.opt_config.init_lr,
        transition_steps=config.opt_config.lr_scheduler_transition_steps,
        decay_rate=config.opt_config.lr_scheduler_decay_factor,
        staircase=True,
    )
    self._opt = optax.chain(
        optax.adamw(
            learning_rate=opt_scheduler,
            weight_decay=config.opt_config.weight_decay
        ),
        optax.clip_by_global_norm(config.opt_config.clip_by_global_norm),
    )

  def init_run_state(self, rng: chex.PRNGKey) -> RunState:
    """Initializes the run state.

    Args:
      rng: A Jax random key.

    Returns:
      A run state.
    """
    (
        rng_env,
        rng_env_states,
        rng_demonstrations,
        rng_params,
        rng_run_state
    ) = jax.random.split(rng, num=5)

    # Initialize the environment, the environment states, the synthetic
    # demonstrations, and the network parameters.
    self._env = environment.Environment(rng_env, self._config.env_config)
    env_states = self._env.init_state(
        jax.random.split(rng_env_states, self._config.exp_config.batch_size)
    )
    demonstrations = demonstrations_lib.generate_synthetic_demonstrations(
        self._config.env_config.max_tensor_size,
        self._config.dem_config,
        jax.random.split(
            rng_demonstrations, num=self._config.exp_config.batch_size
        )
    )
    params = self._network.init(
        rng_params, self._env.get_observation(env_states)
    )
    # Initialize the game statistics.
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    game_stats = GameStats(
        num_games=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors,),
            dtype=jnp.int32
        ),
        best_return=jnp.array([-jnp.inf] * num_target_tensors),
        avg_return=jnp.zeros(
            (self._config.exp_config.batch_size, num_target_tensors)
        ),
    )
    return RunState(
        params=params,
        env_states=env_states,
        demonstrations=demonstrations,
        demonstrations_states=self._env.init_state_from_demonstration(
            demonstrations
        ),
        opt_state=self._opt.init(params),
        game_stats=game_stats,
        rng=rng_run_state,
    )

  def _recurrent_fn(
      self,
      params: chex.ArrayTree,
      rng: chex.PRNGKey,
      actions: jt.Integer[jt.Array, 'batch_size'],
      env_states: environment.EnvState
  ) -> tuple[mctx.RecurrentFnOutput, environment.EnvState]:
    """Implements the recurrent policy.

    In AlphaTensor-Quantum, the environment is deterministic, so there is no
    need for a recurrent function that captures the environment dynamics.
    Instead of a neural network that predicts some embeddings representing the
    environment state, we return the environment state itself.

    Args:
      params: The network parameters.
      rng: A Jax random key.
      actions: The batched action indices.
      env_states: The batched environment states.

    Returns:
      A 2-tuple:
      - The output of the recurrent function.
      - The new environment states.
    """
    env_states = self._env.step(actions, env_states)
    observations = self._env.get_observation(env_states)
    policy_logits, values = self._network.apply(params, rng, observations)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        prior_logits=policy_logits,
        value=values,
        reward=env_states.last_reward,
        discount=1.0 - env_states.is_terminal
    )
    return recurrent_fn_output, env_states

  def _loss_fn(
      self,
      params: chex.ArrayTree,
      global_step: int,
      acting_observations: environment.Observation,
      acting_policy_targets: jt.Float[jt.Array, 'batch_size num_actions'],
      acting_value_targets: jt.Float[jt.Array, 'batch_size'],
      demonstrations_observations: environment.Observation,
      demonstrations_policy_targets: jt.Float[jt.Array,
                                              'batch_size num_actions'],
      demonstrations_value_targets: jt.Float[jt.Array, 'batch_size'],
      rng: chex.PRNGKey,
  ) -> jt.Float[jt.Scalar, '']:
    """Obtains the loss.

    Args:
      params: The network parameters.
      global_step: The training step.
      acting_observations: The (batched) observed environment state.
      acting_policy_targets: The (batched) policy targets from the actors.
      acting_value_targets: The (batched) value targets from the actors.
      demonstrations_observations: The (batched) observed environment state from
        the synthetic demonstrations.
      demonstrations_policy_targets: The (batched) policy targets for the
        synthetic demonstrations.
      demonstrations_value_targets: The (batched) value targets for the
        synthetic demonstrations.
      rng: A Jax random key.

    Returns:
      The sum of the policy and value losses.
    """
    rng_acting, rng_demonstrations = jax.random.split(rng, num=2)

    # Loss corresponding to the episodes from acting.
    acting_policy_logits, acting_values = self._network.apply(
        params, rng_acting, acting_observations
    )
    acting_policy_logprobs = jax.nn.log_softmax(acting_policy_logits)
    acting_policy_loss = jnp.sum(acting_policy_targets * (
        jnp.log(acting_policy_targets) - acting_policy_logprobs
    ), axis=-1)
    acting_value_loss = jnp.square(acting_values - acting_value_targets)
    acting_loss = jnp.mean(acting_policy_loss + acting_value_loss)

    # Loss corresponding to the episodes from synthetic demonstrations.
    demonstrations_policy_logits, demonstrations_values = self._network.apply(
        params, rng_demonstrations, demonstrations_observations
    )
    demonstrations_policy_logprobs = jax.nn.log_softmax(
        demonstrations_policy_logits
    )
    demonstrations_policy_loss = -jnp.sum(
        demonstrations_policy_targets * demonstrations_policy_logprobs,
        axis=-1
    )
    demonstrations_value_loss = jnp.square(
        demonstrations_values - demonstrations_value_targets
    )
    demonstrations_loss = jnp.mean(
        demonstrations_policy_loss + demonstrations_value_loss
    )

    # Obtain the weight for the two terms in the loss.
    demonstrations_weight = optax.piecewise_constant_schedule(
        init_value=self._config.exp_config.loss.init_demonstrations_weight,
        boundaries_and_scales=(
            self._config.exp_config.loss.demonstrations_boundaries_and_scales
        )
    )(global_step)
    return (
        (1.0 - demonstrations_weight) * acting_loss
        + demonstrations_weight * demonstrations_loss
    )

  def _update_game_stats(
      self, run_state: RunState, new_env_states: environment.EnvState
  ) -> GameStats:
    """Returns the new game statistics."""
    is_terminal = new_env_states.is_terminal
    new_num_games_if_terminal = jax.vmap(
        lambda x, idx: x.at[idx].set(x[idx] + 1)
    )(run_state.game_stats.num_games, new_env_states.init_tensor_index)
    new_num_games = jnp.where(
        _broadcast_shapes(is_terminal, run_state.game_stats.num_games),
        new_num_games_if_terminal,
        run_state.game_stats.num_games
    )
    smoothing = self._config.exp_config.avg_return_smoothing
    new_avg_return_if_terminal = jax.vmap(
        lambda x, v, i: x.at[i].set(smoothing * x[i] + (1 - smoothing) * v)
    )(
        run_state.game_stats.avg_return,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    new_avg_return = jnp.where(
        _broadcast_shapes(is_terminal, run_state.game_stats.avg_return),
        new_avg_return_if_terminal,
        run_state.game_stats.avg_return
    )
    num_target_tensors = len(self._config.env_config.target_circuit_types)
    negative_inf = -jnp.inf * jnp.ones(
        (self._config.exp_config.batch_size, num_target_tensors)
    )
    new_best_return_if_terminal = jax.vmap(lambda x, v, i: x.at[i].set(v))(
        negative_inf,
        new_env_states.sum_rewards,
        new_env_states.init_tensor_index
    )
    new_best_return = jnp.maximum(
        run_state.game_stats.best_return,
        jnp.max(jnp.where(
            _broadcast_shapes(is_terminal, new_best_return_if_terminal),
            new_best_return_if_terminal,
            negative_inf
        ), axis=0)
    )
    return GameStats(
        num_games=new_num_games,
        avg_return=new_avg_return,
        best_return=new_best_return,
    )

  def _update_demonstrations_and_states(
      self,
      demonstrations_actions: jt.Integer[jt.Array, 'batch_size'],
      run_state: RunState,
      rng: chex.PRNGKey
  ) -> tuple[demonstrations_lib.Demonstration, environment.EnvState]:
    """Updates the synthetic demonstrations and their states."""

    # Take a step for the environment states.
    new_demonstrations_states = self._env.step(
        demonstrations_actions, run_state.demonstrations_states
    )

    # Update the demonstrations if their corresponding episodes have terminated.
    new_demonstrations_if_terminal = (
        demonstrations_lib.generate_synthetic_demonstrations(
            self._config.env_config.max_tensor_size,
            self._config.dem_config,
            jax.random.split(rng, num=self._config.exp_config.batch_size),
        )
    )
    new_demonstrations = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_if_terminal,
        run_state.demonstrations
    )

    # Update the demonstrations states for terminated episodes.
    new_demonstrations_states_if_terminal = (
        self._env.init_state_from_demonstration(new_demonstrations_if_terminal)
    )
    new_demonstrations_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(
            _broadcast_shapes(new_demonstrations_states.is_terminal, x), x, y
        ),
        new_demonstrations_states_if_terminal,
        new_demonstrations_states
    )
    return new_demonstrations, new_demonstrations_states

  def _run_iteration_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs one iteration of the agent-environment interaction loop.

    Args:
      global_step: The training step.
      run_state: The run state.

    Returns:
      The new run state.
    """
    rngs = jax.random.split(run_state.rng, num=7)

    acting_observations = self._env.get_observation(run_state.env_states)
    policy_logits, values = self._network.apply(
        run_state.params, rngs[0], acting_observations
    )
    root = mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=values,
        embedding=run_state.env_states,
    )
    policy_output = mctx.muzero_policy(
        params=run_state.params,
        rng_key=rngs[1],
        root=root,
        recurrent_fn=self._recurrent_fn,
        num_simulations=self._config.exp_config.num_mcts_simulations,
        qtransform=mctx.qtransform_by_parent_and_siblings,
    )
    search_value = policy_output.search_tree.node_values[
        :, policy_output.search_tree.ROOT_INDEX
    ]

    # Obtain the observations and the policy and value targets for the synthetic
    # demonstrations.
    demonstrations_observations = self._env.get_observation(
        run_state.demonstrations_states
    )
    (
        demonstrations_actions,
        demonstrations_value_targets
    ) = demonstrations_lib.get_action_and_value(
        run_state.demonstrations,
        run_state.demonstrations_states.num_moves,
    )

    # Compute the gradient of the loss and take a grad step.
    grads = jax.grad(self._loss_fn)(
        run_state.params,
        global_step,
        acting_observations,
        policy_output.action_weights,
        search_value,
        demonstrations_observations,
        jax.nn.one_hot(demonstrations_actions, self._num_actions),
        demonstrations_value_targets,
        rngs[2]
    )
    updates, new_opt_state = self._opt.update(
        grads, run_state.opt_state, run_state.params
    )
    new_params = optax.apply_updates(run_state.params, updates)

    # Select next action probabilistically based on visit counts.
    actions = jax.vmap(
        lambda r, p: jax.random.choice(r, a=self._num_actions, p=p)
    )(
        jax.random.split(rngs[3], self._config.exp_config.batch_size),
        policy_output.action_weights
    )
    new_env_states = self._env.step(actions, run_state.env_states)
    is_terminal = new_env_states.is_terminal

    # Update game statistics.
    new_game_stats = self._update_game_stats(run_state, new_env_states)

    # Reset the environment state if the episode has terminated.
    new_env_states = jax.tree_util.tree_map(
        lambda x, y: jnp.where(_broadcast_shapes(is_terminal, x), x, y),
        self._env.init_state(
            jax.random.split(rngs[4], num=self._config.exp_config.batch_size)
        ),
        new_env_states
    )

    # Reset the demonstrations and their states if the corresponding episodes
    # have terminated.
    (
        new_demonstrations, new_demonstrations_states
    ) = self._update_demonstrations_and_states(
        demonstrations_actions, run_state, rngs[5]
    )

    return RunState(
        params=new_params,
        env_states=new_env_states,
        demonstrations=new_demonstrations,
        demonstrations_states=new_demonstrations_states,
        opt_state=new_opt_state,
        game_stats=new_game_stats,
        rng=rngs[6],
    )

  @functools.partial(jax.jit, static_argnums=(0,))
  def run_agent_env_interaction(
      self, global_step: int, run_state: RunState
  ) -> RunState:
    """Runs a few iterations of the agent-environment interaction loop.

    Args:
      global_step: The training step.
      run_state: The run state.

    Returns:
      The new run state, after running `eval_frequency_steps` tranining steps.
    """
    return jax.lax.fori_loop(
        lower=global_step,
        upper=self._config.exp_config.eval_frequency_steps + global_step,
        body_fun=self._run_iteration_agent_env_interaction,
        init_val=run_state,
    )

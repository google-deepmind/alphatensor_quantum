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

"""Neural networks used in AlphaTensor-Quantum."""

import dataclasses
import math

import einshape
import haiku as hk
import jax
import jax.numpy as jnp
import jaxtyping as jt

from alphatensor_quantum.src import config as config_lib
from alphatensor_quantum.src import environment


@dataclasses.dataclass
class _SelfAttention(hk.Module):
  """Implementation of a self-attention module.

  Attributes:
    config: The attention hyperparameters.
    name: The name of the module.
  """

  config: config_lib.AttentionParams
  name: str = 'SelfAttention'

  def _project(
      self,
      inputs: jt.Float[jt.Array, 'batch_size num_tokens num_heads*head_depth'],
      name: str
  ) -> jt.Float[jt.Array, 'batch_size num_tokens num_heads head_depth']:
    """Applies a linear projection to the inputs and reshapes the outputs."""
    outputs_flattened = hk.Linear(
        self.config.num_heads * self.config.head_depth,
        with_bias=True,
        name=name,
        w_init=hk.initializers.VarianceScaling(self.config.init_scale),
    )(inputs)
    return einshape.jax_einshape(
        'bt(hd)->bthd',
        outputs_flattened,
        h=self.config.num_heads,
        d=self.config.head_depth
    )

  def __call__(
      self,
      inputs: jt.Float[jt.Array, 'batch_size num_tokens dimension']
  ) -> jt.Float[jt.Array, 'batch_size num_tokens dimension']:
    """Applies the attention module.

    Args:
      inputs: The input embeddings.

    Returns:
      The output of the self-attention module.
    """
    query = self._project(inputs, 'LinearProjectQuery')
    key = self._project(inputs, 'LinearProjectKey')
    value = self._project(inputs, 'LinearProjectValue')
    logits = jnp.einsum('bthd,bThd->btTh', query, key)
    logits /= math.sqrt(self.config.head_depth)
    weights = jax.nn.softmax(logits, axis=-2)
    outputs_unflattened = jnp.einsum('btTh,bThd->bthd', weights, value)
    return einshape.jax_einshape('bthd->bt(hd)', outputs_unflattened)


class _TransformerDecoderBlock(hk.Module):
  """Implementation of a transformer decoder block.

  The decoder block consists of a self-attention module, a feed-forward module,
  layer normalization, and skip connections.
  """

  def __init__(
      self,
      config: config_lib.AttentionParams,
      name: str = 'TransformerDecoderBlock',
  ):
    """Initializes the module.

    Args:
      config: The attention hyperparameters.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config

    self._self_attention = _SelfAttention(config)

    embedding_dim = self._config.num_heads * self._config.head_depth
    self._feed_forward = hk.Sequential([
        hk.Linear(
            embedding_dim * self._config.mlp_widening_factor,
            with_bias=True,
            w_init=hk.initializers.VarianceScaling(self._config.init_scale),
        ),
        jax.nn.gelu,
        hk.Linear(
            embedding_dim,
            with_bias=True,
            w_init=hk.initializers.VarianceScaling(self._config.init_scale),
        ),
    ])

  def __call__(
      self,
      inputs: jt.Float[jt.Array, 'batch_size num_tokens dimension'],
  ) -> jt.Float[jt.Array, 'batch_size num_tokens dimension']:
    """Applies the transformer decoder block.

    Args:
      inputs: The input embeddings.

    Returns:
      The output of the self-attention module.
    """
    normed_inputs = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name='LayerNorm1',
    )(inputs)
    attended = self._self_attention(normed_inputs)
    embeddings = inputs + attended
    normed_embeddings = hk.LayerNorm(
        axis=-1, create_scale=True, create_offset=True, name='LayerNorm2',
    )(embeddings)
    return embeddings + self._feed_forward(normed_embeddings)


@dataclasses.dataclass
class Symmetrization(hk.Module):
  """Implements a symmetrization layer.

  For an input `X` of shape (..., S, S, c), the Symmetrization layer applies
  the operation `A X + (1 - A) X.T`, where the multiplication is performed
  element-wise, `X.T` denotes the (..., S, S, c) tensor obtained after
  transposing the two S-sized axes of `X`, and `A` is a learnable S x S matrix.

  Attributes:
    name: The name of the module.
  """

  name: str = 'Symmetrization'

  def __call__(
      self,
      inputs: jt.Float[jt.Array, '*batch_size size size dimension'],
  ) -> jt.Float[jt.Array, '*batch_size size size dimension']:
    """Applies the symmetrization operation.

    Args:
      inputs: The input `X`.

    Returns:
      The output of the Symmetrization block, i.e., `A X + (1 - A) X.T`.
    """
    side, side2, _ = inputs.shape[-3:]
    if side != side2:
      raise ValueError(f'Sizes do not match: {side} != {side2}.')

    logits = hk.get_parameter(
        name='logits',
        shape=(side, side, 1),
        init=jnp.zeros,  # So that output is symmetric with init parameters.
    )
    weights = jax.nn.sigmoid(logits)
    return weights * inputs + (1 - weights) * jnp.swapaxes(inputs, -2, -3)


class _SymmetrizedAxialAttention(hk.Module):
  """Implementation of the symmetrized axial attention network.

  Symmetrized axial attention is a network that alternates axial attention
  operations with symmetrization operations.
  """

  def __init__(
      self,
      config: config_lib.NetworkParams,
      name: str = 'SymmetrizedAxialAttention'
  ):
    """Initializes the module.

    Args:
      config: The network hyperparameters.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config

    def make_transformer_block(i: int, j: int) -> _TransformerDecoderBlock:
      return _TransformerDecoderBlock(
          config=self._config.attention_params,
          name=f'TransformerDecoderBlock_{i}_{j}'
      )

    def make_symmetrization_block(i: int) -> Symmetrization:
      return Symmetrization(name=f'Symmetrization_{i}')

    self._axial_attention: list[
        tuple[_TransformerDecoderBlock, _TransformerDecoderBlock]
    ] = []
    self._symmetrization: list[Symmetrization] = []
    for layer in range(self._config.num_layers_torso):
      self._axial_attention.append((
          make_transformer_block(layer, 0), make_transformer_block(layer, 1)
      ))
      self._symmetrization.append(make_symmetrization_block(layer))

  def __call__(
      self,
      inputs: jt.Float[jt.Array, 'batch_size size size dimension'],
  ) -> jt.Float[jt.Array, 'batch_size size size dimension']:
    """Applies the symmetrized axial attention network.

    Args:
      inputs: The input tensor.

    Returns:
      The output of the symmetrized axial attention.
    """
    batch_size, size, _, _ = inputs.shape

    outputs = inputs
    for (axial_attention_1, axial_attention_2), symmetrization_block in zip(
        self._axial_attention, self._symmetrization
    ):
      outputs = einshape.jax_einshape('bnme->(bn)me', outputs)
      outputs = axial_attention_1(outputs)
      outputs = einshape.jax_einshape(
          '(bn)me->(bm)ne', outputs, b=batch_size, n=size
      )
      outputs = axial_attention_2(outputs)
      outputs = einshape.jax_einshape(
          '(bm)ne->bnme', outputs, b=batch_size, n=size
      )
      outputs = symmetrization_block(outputs)
    return outputs


class TorsoNetwork(hk.Module):
  """Implementation of the torso network."""

  def __init__(
      self,
      config: config_lib.NetworkParams,
      name: str = 'TorsoNetwork'
  ):
    """Initializes the module.

    Args:
      config: The network hyperparameters.
      name: The name of the module.
    """
    super().__init__(name=name)
    self._config = config
    self._symmetrized_axial_attention = _SymmetrizedAxialAttention(config)

  def __call__(
      self, observations: environment.Observation
  ) -> jt.Float[jt.Array, 'batch_size sq_size dimension']:
    """Applies the torso network.

    Args:
      observations: A (batched) observation of the environment.

    Returns:
      The output of the torso network, to be sent to the policy and value
      networks.
    """
    tensor_size = observations.tensor.shape[-1]
    scalars = hk.Linear(tensor_size * tensor_size, name='LinearProjectScalars')(
        jnp.expand_dims(observations.sqrt_played_fraction, axis=-1)
    )
    scalars = einshape.jax_einshape(
        'b(nm)->bnm1', scalars, n=tensor_size, m=tensor_size
    )
    past_factors = einshape.jax_einshape(
        'btnm->bnmt', observations.past_factors_as_planes
    )
    embedding_dim = (
        self._config.attention_params.num_heads *
        self._config.attention_params.head_depth
    )
    all_inputs = jnp.concatenate([
        observations.tensor, past_factors, scalars
    ], axis=-1)
    all_inputs_projected = hk.Linear(
        embedding_dim,
        w_init=hk.initializers.TruncatedNormal(self._config.init_scale),
        name='LinearProjectInputs'
    )(all_inputs)
    outputs = self._symmetrized_axial_attention(all_inputs_projected)
    return einshape.jax_einshape('bnmc->b(nm)c', outputs)

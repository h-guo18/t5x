# Copyright 2023 The T5X Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""T5.1.1 Transformer model."""

from typing import Any, Sequence

from flax import linen as nn
from flax import struct
import jax.numpy as jnp
from t5x.examples.t5 import layers

from jax.config import config
# config.update('jax_disable_jit', True)

@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""
  vocab_size: int
  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 512
  num_heads: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  head_dim: int = 64
  mlp_dim: int = 2048
  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0.1
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  # Whether to accumulate attention logits in float32 regardless of dtype.
  float32_attention_logits: bool = False
  linformer : bool = False
  linformer_dim:int = 128 #projection dim of linformer
  kernel_method:str = None # whether or not use kernel method
  absolute_positional_embedding:bool = False # if False, use relative postional encoding

class EncoderLayer(nn.Module):
  """Transformer encoder layer."""
  config: T5Config
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self, inputs, encoder_mask=None, deterministic=False, layer_idx = -1):
    cfg = self.config

    # Relative position embedding as attention biases.
    # kv_length = inputs.shape[-2]
    kv_length = inputs.shape[-2] if not self.config.linformer else self.config.linformer_dim
    encoder_bias = self.relative_embedding(inputs.shape[-2], kv_length,
                                           True)
    if cfg.absolute_positional_embedding:
        #when using absolute positional embedding, set relative bias to zero
        encoder_bias = jnp.zeros_like(encoder_bias)

    # Attention block.
    assert inputs.ndim == 3
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_attention_layer_norm')(
            inputs)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='attention',
        linformer=cfg.linformer,
        linformer_dim = cfg.linformer_dim,
        kernel_method = cfg.kernel_method,
        attn_type= "self-attn"
        )(
            x, x, encoder_mask, encoder_bias, E_key=layer_idx+10,F_key=layer_idx+100,deterministic=deterministic)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    # [batch, length, emb_dim] -> [batch, length, emb_dim]
    y = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(y, deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x

    return y


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config
  relative_embedding: nn.Module

  @nn.compact
  def __call__(self,
               inputs,
               encoded,
               decoder_mask=None,
               encoder_decoder_mask=None,
               kvmask=None,
               qmask=None,
               layer_idx:int = -1, # for generating linfomrer projection RNG
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    # Relative position embedding as attention biases.   
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = self.relative_embedding(l, l, False)
    if cfg.absolute_positional_embedding:
        decoder_bias = jnp.zeros_like(decoder_bias)

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)

    # Self-attention block
    x = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='self_attention',
        attn_type = "self-attn-causal",
        linformer=False,
        kernel_method = cfg.kernel_method
        )(
            x,
            x,
            decoder_mask,
            decoder_bias,
            kvmask= kvmask,
            qmask= qmask,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs

    # Encoder-Decoder block.
    y = layers.LayerNorm(
        dtype=cfg.dtype, name='pre_cross_attention_layer_norm')(
            x)
    y = layers.MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        float32_logits=cfg.float32_attention_logits,
        name='encoder_decoder_attention',
        attn_type="cross-attn",
        linformer=False,
        linformer_dim = cfg.linformer_dim,
        kernel_method = cfg.kernel_method
        )(
            y, encoded, encoder_decoder_mask, kvmask=kvmask,qmask=qmask,E_key=layer_idx+10,F_key=layer_idx+100,deterministic=deterministic)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y + x 

    # MLP block.
    z = layers.LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(y)
    z = layers.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + y

    return z


class Encoder(nn.Module):
  """A stack of encoder layers."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoder_input_tokens,
               encoder_mask=None,
               deterministic=False):
    cfg = self.config
    assert encoder_input_tokens.ndim == 2  # [batch, length]
    rel_emb = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    x = self.shared_embedding(encoder_input_tokens.astype('int32'))
    
    if cfg.absolute_positional_embedding:
        #apply absolute postional embedding at the begin of encoder
        x = layers.absolute_positional_embedding(x,start_position=0,d_model=cfg.emb_dim,dtype=cfg.dtype)
    
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x.astype(cfg.dtype)

    for lyr in range(cfg.num_encoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      x = EncoderLayer(
          config=cfg, relative_embedding=rel_emb,
          name=f'layers_{lyr}')(x, encoder_mask, deterministic,layer_idx=lyr)

    x = layers.LayerNorm(dtype=cfg.dtype, name='encoder_norm')(x)
    return nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=deterministic)


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               encoded,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               encoder_decoder_mask=None,
               kvmask=None,
               qmask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None,
               decode_positional_encoding_index:int =-1):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]
    rel_emb = layers.RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    
    if cfg.absolute_positional_embedding:
        #apply absolute positional encoding at the input of decoder
        # if decode:
            # breakpoint()
        #during decoding, we input on token at a time, so add positional encoding starting from the current index
        ape_start = decode_positional_encoding_index if decode else 0 
        y = layers.absolute_positional_embedding(y,start_position=ape_start,d_model=cfg.emb_dim,dtype=cfg.dtype)
        
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    for lyr in range(cfg.num_decoder_layers):
      # [batch, length, emb_dim] -> [batch, length, emb_dim]
      y = DecoderLayer(
          config=cfg, relative_embedding=rel_emb, name=f'layers_{lyr}')(
              y,
              encoded,
              decoder_mask=decoder_mask,
              encoder_decoder_mask=encoder_decoder_mask,
              kvmask=kvmask,
              qmask=qmask,
              layer_idx=lyr,
              deterministic=deterministic,
              decode=decode,
              max_decode_length=max_decode_length)

    y = layers.LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = layers.DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              y)
    return logits


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config

  def setup(self):
    cfg = self.config
    self.shared_embedding = layers.Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.encoder = Encoder(config=cfg, shared_embedding=self.shared_embedding)
    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

  def encode(self,
             encoder_input_tokens,
             encoder_segment_ids=None,
             enable_dropout=True):
    """Applies Transformer encoder-branch on the inputs."""
    cfg = self.config
    assert encoder_input_tokens.ndim == 2, (
        f'Expected `encoder_input_tokens` to be of shape (batch, len). '
        f'Got {encoder_input_tokens.shape}')

    # Make padding attention mask.
    encoder_mask = layers.make_attention_mask(
        encoder_input_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
    # Add segmentation block-diagonal attention mask if using segmented data.
    if self.config.linformer:
        assert encoder_segment_ids is None, "in linformer setting, must not use packing."
    if encoder_segment_ids is not None:
      encoder_mask = layers.combine_masks(
          encoder_mask,
          layers.make_attention_mask(
              encoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))
    #truncate encoder_mask shape
    return self.encoder(
        encoder_input_tokens, encoder_mask, deterministic=not enable_dropout)

  def decode(
      self,
      encoded,
      encoder_input_tokens,  # only needed for masks
      decoder_input_tokens,
      decoder_target_tokens,
      encoder_segment_ids=None,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None,
      decode_positional_encoding_index:int = -1
      ):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
      encoder_decoder_mask = layers.make_attention_mask(
          jnp.ones_like(decoder_target_tokens),
          encoder_input_tokens > 0,
          dtype=cfg.dtype)
      qmask=None
    else:
      decoder_mask = layers.make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)
      encoder_decoder_mask = layers.make_attention_mask(
          decoder_target_tokens > 0, encoder_input_tokens > 0, dtype=cfg.dtype)
      qmask= layers.make_attention_mask(
          decoder_target_tokens > 0,
          decoder_target_tokens > 0,
          dtype=cfg.dtype)
    #no matter decoder or not, kv mask (i.e. input mask) is the same
    kvmask=layers.make_attention_mask(
          encoder_input_tokens > 0,
          encoder_input_tokens > 0,
          dtype=cfg.dtype)

    # Add segmentation block-diagonal attention masks if using segmented data.
    assert encoder_segment_ids is None, "Should not have input packing here."
    if encoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

      encoder_decoder_mask = layers.combine_masks(
          encoder_decoder_mask,
          layers.make_attention_mask(
              decoder_segment_ids,
              encoder_segment_ids,
              jnp.equal,
              dtype=cfg.dtype))

    logits = self.decoder(
        encoded,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        encoder_decoder_mask=encoder_decoder_mask,
        kvmask=kvmask,
        qmask=qmask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length,
        decode_positional_encoding_index = decode_positional_encoding_index)
    return logits

  def __call__(self,
               encoder_input_tokens,
               decoder_input_tokens,
               decoder_target_tokens,
               encoder_segment_ids=None,
               decoder_segment_ids=None,
               encoder_positions=None,
               decoder_positions=None,
               *,
               enable_dropout: bool = True,
               decode: bool = False,
               decode_positional_encoding_index:int=-1):
    """Applies Transformer model on the inputs.

    This method requires both decoder_target_tokens and decoder_input_tokens,
    which is a shifted version of the former. For a packed dataset, it usually
    has additional processing applied. For example, the first element of each
    sequence has id 0 instead of the shifted EOS id from the previous sequence.

    Args:
      encoder_input_tokens: input data to the encoder.
      decoder_input_tokens: input token to the decoder.
      decoder_target_tokens: target token to the decoder.
      encoder_segment_ids: encoder segmentation info for packed examples.
      decoder_segment_ids: decoder segmentation info for packed examples.
      encoder_positions: encoder subsequence positions for packed examples.
      decoder_positions: decoder subsequence positions for packed examples.
      enable_dropout: Ensables dropout if set to True.
      decode: Whether to prepare and use an autoregressive cache.

    Returns:
      logits array from full transformer.
    """
    encoded = self.encode(
        encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        enable_dropout=enable_dropout)

    return self.decode(
        encoded,
        encoder_input_tokens,  # only used for masks
        decoder_input_tokens,
        decoder_target_tokens,
        encoder_segment_ids=encoder_segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode,
        decode_positional_encoding_index=decode_positional_encoding_index)

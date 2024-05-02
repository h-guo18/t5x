import math
import warnings

# import numpy as np
from .attn_utils import pad_to_multiple, prm_projection
from einops import rearrange
from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import lax
def masked_fill(mask, a, fill):
#    return jax.lax.select(mask, a, jax.lax.broadcast(fill, a.shape))
    return jnp.where(mask, fill, a)
# class EVA(LocalAttention):
#     def __init__(self,
#                  adaptive_proj='default',
#                  num_landmarks=128,
#                  use_t5_rpe=False,
#                  linformer = False,
#                  *args,
#                  **kwargs):
        # super(EVA, self).__init__(*args, **kwargs)
        # adaptive_proj = adaptive_proj
        # if adaptive_proj in ['default']:
        #     adaptive_mu_q = nn.Sequential(
        #         nn.Linear(head_dim, head_dim),
        #         nn.LayerNorm(head_dim),
        #     )
        #     adaptive_mu_k = nn.Sequential(
        #         nn.Linear(head_dim, head_dim),
        #         nn.LayerNorm(head_dim),
        #     )
        # elif adaptive_proj in ['no-ln']:
        #     adaptive_mu_q = nn.Sequential(
        #         nn.Linear(head_dim, head_dim),
        #     )
        #     adaptive_mu_k = nn.Sequential(
        #         nn.Linear(head_dim, head_dim),
        #     )
        # elif adaptive_proj in ['none']:
        #     adaptive_mu_k = nn.Sequential(
        #         nn.Linear(head_dim, head_dim),
        #         nn.LayerNorm(head_dim),
        #     )
        # use_t5_rpe = use_t5_rpe
        # num_landmarks = num_landmarks
        # if use_rpe and not use_t5_rpe:
        #     warnings.warn(
        #         "By setting --use-rpe, the default relative positional embedding for local window is used."
        #         "We also implement a T5-style positional encoding, which we observe performs slightly better;"
        #         "This feature can be enabled by only setting --use-t5-rpe."
        #     )
        # elif use_rpe and use_t5_rpe:
        #     raise NotImplementedError("Default RPE and T5-style RPE cannot be true simultaneously.")
        # if use_t5_rpe:
        #     rel_pos_bias = T5RelativePositionBias(
        #         scale, 
        #         num_heads = num_heads,
        #         causal = False, 
        #         num_buckets=max(min(int((window_size + ext_size) / 2), 64), 16), 
        #         max_distance=window_size + ext_size
        #     )
        # linformer = linformer
        # apply(_init_weights)
def window_partition( x, shape, ext_window_size, pad_val=0, window_size=2):
    return rearrange(x, '... (g w) d -> ... g w d', w=window_size)

def _process_input( x, key_padding_mask,window_size):
    # this function re-implements the parent method.
    B, seq_shape,h, C = x.shape
    if key_padding_mask is None:
        x, key_padding_mask = pad_to_multiple(x, window_size, dim=1, create_mask=True)
    else:
        x = pad_to_multiple(x, window_size, dim=1)
        key_padding_mask = pad_to_multiple(key_padding_mask, window_size, dim=1, value=True)
    return x, key_padding_mask, seq_shape

def eva_attn(q,k,v,adaptive_mu_q,adaptive_mu_k,training,linformer=False,num_landmarks=128,
            ext_size=0,adaptive_proj='default',window_size = 2,head_dim= 64,linformer_dim=128,
            **kwargs):
    mask_val = -5e4
    ######################## Generate Proposal Parameters ###############################
    B, q_orig_n,h, C =q.shape
    B, kv_origin_n,h, C =k.shape
    scale = head_dim ** -0.5
    
    q, q_key_padding_mask, q_len = _process_input(q, key_padding_mask=None,window_size=window_size)
    k, kv_key_padding_mask, kv_len = _process_input(k, key_padding_mask=None,window_size=window_size)
    v, _, _ = _process_input(v, key_padding_mask=None,window_size=window_size)
    #(b,l,h,d)
    
    # N = seq_shape
    # q, k, v = proj_and_split_heads(x)
    #qkv: (b,h,l,d)
    q = jnp.einsum('blhd->bhld',q)
    k = jnp.einsum('blhd->bhld',k)
    v = jnp.einsum('blhd->bhld',v)
    

    # key_padding_mask = jnp.zeros(B, N, dtype=k.dtype, device=k.device)
    q_key_padding_mask = jnp.expand_dims(jnp.expand_dims(q_key_padding_mask,1),-1).astype(jnp.bool_) # [b, 1, n, 1]
    kv_key_padding_mask = jnp.expand_dims(jnp.expand_dims(kv_key_padding_mask,1),-1).astype(jnp.bool_) # [b, 1, n, 1]
    
    w_q_winsize = window_size*(q_len//linformer_dim) if linformer else window_size
    w_q = window_partition(q, q_len, window_size=w_q_winsize,ext_window_size=0)
    w_k = window_partition(k, kv_len, ext_window_size=ext_size)
    w_v = window_partition(v, kv_len, ext_window_size=ext_size) # [b, h, w, j, d]


    q_rf_win_size = int(q_len // num_landmarks)
    kv_rf_win_size = int(kv_len // num_landmarks)
    # [b, h, c, j, d]
    rf_w_q = window_partition(q, q_len, window_size=q_rf_win_size, ext_window_size=ext_size)
    # [b, h, c, j, d]
    rf_w_k = window_partition(k, kv_len, window_size=kv_rf_win_size, ext_window_size=ext_size)
    # [b, h, c, j, d]
    rf_w_v = window_partition(v, kv_len, window_size=kv_rf_win_size, ext_window_size=ext_size)
    # compute local attention
    if not linformer:
        # [b, 1, c, j, 1]
        q_rf_w_mask = window_partition(
            q_key_padding_mask, 
            q_len, 
            window_size=q_rf_win_size,
            ext_window_size=ext_size,
            pad_val=1
            ).astype(jnp.bool_)
        kv_rf_w_mask = window_partition(
            kv_key_padding_mask, 
            kv_len, 
            window_size=kv_rf_win_size,
            ext_window_size=ext_size,
            pad_val=1
            ).astype(jnp.bool_)
        rf_w_q = masked_fill(q_rf_w_mask, rf_w_q,0.)
        rf_w_k = masked_fill(kv_rf_w_mask, rf_w_k,0.)
        rf_w_v = masked_fill(kv_rf_w_mask, rf_w_v,0.)

    if adaptive_proj in ['default', 'no-ln']:
        rf_q_bar = adaptive_mu_q(rf_w_q.mean(axis=-2))
        rf_k_bar = adaptive_mu_k(rf_w_k.mean(axis=-2))
        # [b, h, c, d]
        mu = 0.5 * (rf_q_bar + rf_k_bar)
    else:
        raise NotImplementedError
    ######################## Sampling from proposal ###############################
    if training:
        seed = lax.convert_element_type(jnp.ceil(jnp.sum(q) * 10000000.0), jnp.int32)
        weights = mu + jax.random.normal(key=jax.random.PRNGKey(seed),shape=mu.shape,dtype=k.dtype)
    else:
        weights = mu    
    # [b, h, c, j, d], [b, h, c, 1, d] -> [b, h, c, j]
    log_proj_w_k = prm_projection(rf_w_k, jnp.expand_dims(weights,-2), normalize=False).squeeze(-2)
    
    if not linformer:
        log_proj_w_k = masked_fill(kv_rf_w_mask.squeeze(-1), log_proj_w_k,mask_val)

    # [b, h, c, j] [b, h, c, j, d] -> [b, h, c, d]
    beta = jnp.einsum('...cj,...cjd->...cd', jax.nn.softmax(log_proj_w_k, axis=-1), rf_w_v)
    
    # compute approx. expectation of CVs.
    # [b, h, c, d]
    rfa_chunk = jnp.einsum('...wid,...cd->...wic', w_q, scale * rf_k_bar)
    num_rfa_chunks = rfa_chunk.shape[-1]

    if not linformer:
        # compute local attention
        #(b,1,w,i)
        q_local_dots_mask = window_partition( 
            q_key_padding_mask, 
            q_len, 
            ext_window_size=ext_size,
            pad_val=1
            ).astype(jnp.bool_).squeeze(-1)
        #(b,1,w,j)
        kv_local_dots_mask = window_partition(
            kv_key_padding_mask, 
            kv_len, 
            ext_window_size=ext_size,
            pad_val=1
            ).astype(jnp.bool_).squeeze(-1)
        local_dots_mask = jnp.einsum('bhwi,bhwj->bhwij',q_local_dots_mask,kv_local_dots_mask)

    log_qk_local_dot = jnp.einsum('bhwie,bhwje->bhwij', w_q, w_k) * scale # [b, h, w, i, j]
    
    if not linformer:
        log_qk_local_dot = masked_fill(local_dots_mask, log_qk_local_dot,mask_val)
        
    local_len = log_qk_local_dot.shape[-1]
    
    # compute attention weights along with normalizing constant.
    attn = jax.nn.softmax(jnp.concatenate([log_qk_local_dot, rfa_chunk], axis=-1), axis=-1)
    local_attn, ra_attn = jnp.split(attn, [local_len,], axis=-1)
    output_local = jnp.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
    output_snis = jnp.einsum('bhwic,bhcd->bhwid', ra_attn, beta) 
    ######################## Combine them together ############################
    output = rearrange(output_snis + output_local, '... g w d ->... (g w) d') # [b, h, n, d]
    x = jnp.transpose(output,(0, 2, 1, 3)) #(b,n,h,d)
    if q_orig_n is not None:
        x = x[..., :q_orig_n, :,:]
    return x

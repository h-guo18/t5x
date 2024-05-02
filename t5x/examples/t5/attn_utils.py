# import torch
# from torch import nn
# import torch.nn.functional as F
from flax import linen as nn
import jax.numpy as jnp
import jax
from einops import rearrange, repeat
import math

# adapted from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
# def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False):
#     assert dim < 0 # only accept ``dim'' index in a reverse manner
#     seqlen = int(tensor.shape[dim])
#     m = seqlen / multiple
#     if m.is_integer():
#         if create_mask:
#             return tensor, jnp.zeros(size=(tensor.shape[0], tensor.shape[-2]), dtype=jnp.bool_, device=tensor.device)
#         else:
#             return tensor
#     remainder = math.ceil(m) * multiple - seqlen
#     pad_offset = (0,) * (-1 - dim) * 2
#     padded_res = jnp.pad(tensor, (*pad_offset, 0, remainder), value=value)
#     if create_mask:
#         # assume dim 0 is the batch size
#         padding_mask = jnp.zeros(size=(padded_res.shape[0], padded_res.shape[-2]), dtype=jnp.bool_, device=padded_res.device)
#         padding_mask[:, -remainder:] = True
#         return padded_res, padding_mask
#     else:
#         return padded_res
def pad_to_multiple(tensor, multiple, dim=1, value=0, create_mask=False):
    # assert dim < 0 # only accept ``dim'' index in a reverse manner
    # tensor:(b,l,h,d)
    seqlen = int(tensor.shape[dim])
    m = seqlen / multiple
    if m.is_integer():
        if create_mask:
            #mask:(b,l)
            return tensor, jnp.zeros(shape=(tensor.shape[0], tensor.shape[1]), dtype=jnp.bool_)
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    padded_res = jnp.pad(tensor, (0,0, 0,remainder, 0,0 ,0,0), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = jnp.zeros(shape=(padded_res.shape[0], padded_res.shape[1]), dtype=jnp.bool_)
        padding_mask[:, -remainder:] = True
        return padded_res, padding_mask
    else:
        return padded_res


def prm_projection(
    data: jnp.ndarray,
    projection_matrix: jnp.ndarray,
    normalize: bool=True,
    diagonal: bool=False,
    return_exp: bool=False,
    is_query: bool=False,
    eps: float=1e-8):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    is_query: predicate indicating whether input data corresponds to queries or
        keys
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: scaler with 0.5 could considerably stablizes training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    # normalized_data = (data.shape[-1] ** -0.5) * data
    # data_dash = torch.einsum('...nd,...md->...nm', 
    #                         projection_matrix,
    #                         normalized_data,
    #                         ) # [n, b, h, c, lq]
    # norm = torch.sum(normalized_data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    data_normalizer = (data.shape[-1] ** -0.5)
    if diagonal:
        data_dash = jnp.einsum('...nd,...nd->...n', 
                            projection_matrix,
                            (data_normalizer * data),
                            ) # [n, b, h, lq, lk]
        norm = data_normalizer * jnp.sum(data ** 2, axis=-1) / 2.0# [n, b, h, 1, lk]
    else:
        data_dash = jnp.einsum('...nd,...md->...nm', 
                                projection_matrix,
                                (data_normalizer * data),
                                ) # [n, b, h, lq, lk]
        norm = data_normalizer * jnp.expand_dims(jnp.sum(data ** 2, axis=-1),-2) / 2.0# [n, b, h, 1, lk]
    if normalize:
        proj_data = jax.nn.softmax(data_dash - norm, axis=-1)  # [n, b, h, l_c, l_k]
    elif return_exp:
        if is_query:
            proj_data = jnp.exp(
                data_dash - norm - jnp.amax(data_dash, axis=-2, keepdim=True).detach()) + eps       
        else:
            proj_data = jnp.exp(
                data_dash - norm - jnp.amax(data_dash, axis=(-1, -2, -3), keepdim=True).detach()) + eps           
    else:
        proj_data = data_dash - norm
    return proj_data

import math
import warnings

# import numpy as np
from efficient_attention import add_nested_argument
from efficient_attention.attn_utils import pad_to_multiple, prm_projection
from efficient_attention.local_attention import LocalAttention
from einops import rearrange
from flax import linen as nn
import jax.numpy as jnp
import jax
from jax import lax


# adapted from 
# https://github.com/lucidrains/FLASH-pytorch/blob/main/flash_pytorch/flash_pytorch.py#L54
class T5RelativePositionBias(nn.Module):
    def __init__(
        self,
        scale,
        num_heads,
        causal = False,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        causal = True,
        num_buckets = 32,
        max_distance = 128
    ):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.max(n, jnp.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            jnp.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = jnp.min(val_if_large, jnp.full_like(val_if_large, num_buckets - 1))

        ret += jnp.where(is_small, n, val_if_large)
        return ret

    def forward(self, x):
        i, j, device = *x.shape[-2:], x.device
        q_pos = jnp.arange(i, dtype = jnp.long, device = device)
        k_pos = jnp.arange(j, dtype = jnp.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        bias = self.relative_attention_bias(rp_bucket).permute([2, 0, 1]).unsqueeze(0).unsqueeze(2)
        return bias * self.scale



class EVA(LocalAttention):
    def __init__(self,
                 adaptive_proj='default',
                 num_landmarks=49,
                 use_t5_rpe=False,
                 *args,
                 **kwargs):
        super(EVA, self).__init__(*args, **kwargs)
        self.adaptive_proj = adaptive_proj
        if self.adaptive_proj in ['default']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        elif self.adaptive_proj in ['no-ln']:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
        elif self.adaptive_proj in ['none']:
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        self.use_t5_rpe = use_t5_rpe
        self.num_landmarks = num_landmarks
        if self.use_rpe and not self.use_t5_rpe:
            warnings.warn(
                "By setting --use-rpe, the default relative positional embedding for local window is used."
                "We also implement a T5-style positional encoding, which we observe performs slightly better;"
                "This feature can be enabled by only setting --use-t5-rpe."
            )
        elif self.use_rpe and self.use_t5_rpe:
            raise NotImplementedError("Default RPE and T5-style RPE cannot be true simultaneously.")
        if self.use_t5_rpe:
            self.rel_pos_bias = T5RelativePositionBias(
                self.scale, 
                num_heads = self.num_heads,
                causal = False, 
                num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16), 
                max_distance=self.window_size + self.ext_size
            )
        self.apply(self._init_weights)

    def _process_input(self, x, key_padding_mask):
        # this function re-implements the parent method.
        B, seq_shape,h, C = x.shape
        if self.window_size > 0:
            if key_padding_mask is None:
                x, key_padding_mask = pad_to_multiple(x, self.window_size, dim=1, create_mask=True)
            else:
                x = pad_to_multiple(x, self.window_size, dim=1)
                key_padding_mask = pad_to_multiple(key_padding_mask, self.window_size, dim=1, value=True)
        return x, key_padding_mask, seq_shape

    def forward(self, q,k,v,**kwargs):
        mask_val = -5e4
        ######################## Generate Proposal Parameters ###############################
        B, q_orig_n,h, C =q.shape
        B, kv_origin_n,h, C =k.shape
        
        q, q_key_padding_mask, q_len = self._process_input(q, key_padding_mask=None)
        k, kv_key_padding_mask, kv_len = self._process_input(k, key_padding_mask=None)
        v, _, _ = self._process_input(v, key_padding_mask=None)
        #(b,l,h,d)
        
        # N = seq_shape
        # q, k, v = self.proj_and_split_heads(x)
        #qkv: (b,h,l,d)
        q = jnp.einsum('blhd->bhld',q)
        k = jnp.einsum('blhd->bhld',k)
        v = jnp.einsum('blhd->bhld',v)
        

        # key_padding_mask = jnp.zeros(B, N, dtype=k.dtype, device=k.device)
        q_key_padding_mask = q_key_padding_mask.expand_dims(1).expand_dims(-1).astype(jnp.bool) # [b, 1, n, 1]
        kv_key_padding_mask = kv_key_padding_mask.expand_dims(1).expand_dims(-1).astype(jnp.bool) # [b, 1, n, 1]
        
       
        w_q = self.window_partition(q, q_len, ext_window_size=0)
        w_k = self.window_partition(k, kv_len, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, kv_len, ext_window_size=self.ext_size) # [b, h, w, j, d]


        q_rf_win_size = int(q_len // self.num_landmarks)
        kv_rf_win_size = int(kv_len // self.num_landmarks)
        # [b, h, c, j, d]
        rf_w_q = self.window_partition(q, q_len, window_size=q_rf_win_size, ext_window_size=self.ext_size)
        # [b, h, c, j, d]
        rf_w_k = self.window_partition(k, kv_len, window_size=kv_rf_win_size, ext_window_size=self.ext_size)
        # [b, h, c, j, d]
        rf_w_v = self.window_partition(v, kv_len, window_size=kv_rf_win_size, ext_window_size=self.ext_size)
        # compute local attention
        # [b, 1, c, j, 1]
        q_rf_w_mask = self.window_partition(
            q_key_padding_mask, 
            q_len, 
            window_size=q_rf_win_size,
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(jnp.bool)
        kv_rf_w_mask = self.window_partition(
            kv_key_padding_mask, 
            kv_len, 
            window_size=kv_rf_win_size,
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(jnp.bool)
        rf_w_q = rf_w_q.masked_fill(q_rf_w_mask, 0.)
        rf_w_k = rf_w_k.masked_fill(kv_rf_w_mask, 0.)
        rf_w_v = rf_w_v.masked_fill(kv_rf_w_mask, 0.)

        if self.adaptive_proj in ['default', 'no-ln']:
            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(dim=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            # [b, h, c, d]
            mu = 0.5 * (rf_q_bar + rf_k_bar)
        elif self.adaptive_proj == 'none':
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            mu = jnp.zeros_like(rf_k_bar)
        ######################## Sampling from proposal ###############################
        if self.training:
            seed = lax.convert_element_type(jnp.ceil(jnp.sum(q) * 10000000.0), jnp.int32)
            weights = mu + jax.random.normal(key=jax.random.PRNGKey(seed),shape=mu.shape,dtype=k.dtype)
        else:
            weights = mu    
        # [b, h, c, j, d], [b, h, c, 1, d] -> [b, h, c, j]
        log_proj_w_k = prm_projection(rf_w_k, jax.expand_dims(weights,-2), normalize=False).squeeze(-2)
        log_proj_w_k = log_proj_w_k.masked_fill(kv_rf_w_mask.squeeze(-1), mask_val)

        # [b, h, c, j] [b, h, c, j, d] -> [b, h, c, d]
        beta = jnp.einsum('...cj,...cjd->...cd', jax.nn.softmax(log_proj_w_k, dim=-1), rf_w_v)
        
        # compute approx. expectation of CVs.
        # [b, h, c, d]
        rfa_chunk = jnp.einsum('...wid,...cd->...wic', w_q, self.scale * rf_k_bar)
        num_rfa_chunks = rfa_chunk.shape[-1]

        # compute local attention
        #(b,1,w,i)
        q_local_dots_mask = self.window_partition( 
            q_key_padding_mask, 
            q_len, 
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(jnp.bool).squeeze(-1)
        #(b,1,w,j)
        kv_local_dots_mask = self.window_partition(
            kv_key_padding_mask, 
            kv_len, 
            ext_window_size=self.ext_size,
            pad_val=1
            ).to(jnp.bool).squeeze(-1)
        local_dots_mask = jnp.einsum('bhwi,bhwj->bhwij',q_local_dots_mask,kv_local_dots_mask)

        log_qk_local_dot = jnp.einsum('bhwie,bhwje->bhwij', w_q, w_k) * self.scale # [b, h, w, i, j]
        if self.use_t5_rpe:
            # here the t5-rpe-bias has already been scaled by \sqrt{d}
            log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)
        if self.use_rpe:
            log_qk_local_dot = self.add_rel_pos_bias(log_qk_local_dot)
        
        log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)
        local_len = log_qk_local_dot.shape[-1]
        
        # compute attention weights along with normalizing constant.
        attn = jax.nn.softmax(jnp.concatenate   ([log_qk_local_dot, rfa_chunk], dim=-1), dim=-1)
        local_attn, ra_attn = jnp.split(attn, [local_len, num_rfa_chunks], dim=-1)
        output_local = jnp.einsum('bhwij,bhwjd->bhwid', local_attn, w_v)
        output_snis = jnp.einsum('bhwic,bhcd->bhwid', ra_attn, beta) 
        ######################## Combine them together ############################
        output = self.window_merge(output_snis + output_local, q_len) # [b, h, n, d]
        x = output.permute(0, 2, 1, 3) #(b,n,h,d)
        x = self.proj(x)
        if q_orig_n is not None:
            x = x[..., :q_orig_n, :]
        x = self.proj_drop(x)
        return x

    @staticmethod
    def add_attn_specific_args(parent_parser, struct_name="attn_args", prefix=""):
        if hasattr(LocalAttention, "add_attn_specific_args"):
            parent_parser = LocalAttention.add_attn_specific_args(parent_parser, struct_name=struct_name, prefix=prefix)
        parser = parent_parser.add_argument_group("attention")
        _name_prefix = prefix + "-" if len(prefix) > 1 else ""
        add_nested_argument(parser, '--{}adaptive-proj'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default='default', type=str)
        add_nested_argument(parser, '--{}num-landmarks'.format(_name_prefix), struct_name=struct_name, prefix=prefix, default=49, type=int)
        add_nested_argument(parser, '--{}use-t5-rpe'.format(_name_prefix), action='store_true', struct_name=struct_name, prefix=prefix, default=False)
        return parent_parser
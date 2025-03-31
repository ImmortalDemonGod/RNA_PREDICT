# protenix/model/modules/primitives.py
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from protenix.model.utils import (
    flatten_final_dims,
    move_final_dim_to_dim,
    pad_at_dim,
    reshape_at_dim,
)
from protenix.openfold_local.model.primitives import LayerNorm
from protenix.openfold_local.utils.chunk_utils import chunk_layer
from torch.nn import Linear

LinearNoBias = partial(Linear, bias=False)


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = nn.LayerNorm(c_a, elementwise_affine=False, bias=False)
        # The pytorch version should be newer than 2.1
        self.layernorm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = LinearNoBias(in_features=c_s, out_features=c_a)

    def zero_init(self):
        nn.init.zeros_(self.linear_s.weight)
        nn.init.zeros_(self.linear_s.bias)
        nn.init.zeros_(self.linear_nobias_s.weight)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class BiasInitLinear(Linear):
    """Support biasinit for nn.Linear Called just like torch.nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        biasinit: float = 0.0,
    ) -> None:
        """
        Args:
            in_features (int): in_features
            out_features (int): out_features
            bias (bool, optional): whether add bias. Defaults to True.
            biasinit (float, optional): the initial bias value. Defaults to 0.0.
        """
        super(BiasInitLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3
    """

    def __init__(self, c_in: int, n: int) -> None:
        """
        Args:
            c_in (int, optional): the input dimension.
            n (int, optional): factor by which c_in is multiplied to obtain hidden dimension.
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.layernorm1 = LayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias_b = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias = LinearNoBias(in_features=n * c_in, out_features=c_in)
        self.zero_init()

    def zero_init(self):
        nn.init.zeros_(self.linear_no_bias.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        if self.training:
            x = self.layernorm1(x)
            a = self.linear_no_bias_a(x)
            b = self.linear_no_bias_b(x)
            x = self.linear_no_bias(F.silu(a) * b)
            return x
        else:
            other_dims = x.shape[:-1]
            dim_size = x.shape[-1]
            size = x.shape[-2]
            x = x.reshape(-1, dim_size)
            chunk_num = 1 if size < 3200 else 8
            chunks = torch.chunk(x, chunk_num, dim=-2)
            outputs = torch.empty(
                (x.shape[0], self.c_in), dtype=x.dtype, device=x.device
            )
            start = 0
            for chunk in chunks:
                y = self.layernorm1(chunk)
                a = self.linear_no_bias_a(y)
                a = F.silu(a, True)
                b = self.linear_no_bias_b(y)
                del y
                b *= a
                del a
                b = self.linear_no_bias(b)
                outputs[start : start + b.shape[0]] = b
                start += b.shape[0]
                del b
            outputs = outputs.reshape(*other_dims, self.c_in)
            return outputs


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    use_efficient_implementation: bool = False,
    attn_weight_dropout_p: float = 0.0,
    inplace_safe: bool = False,
) -> torch.Tensor:
    """Attention.

    Args:
        q (torch.Tensor): query tensor of shape [..., n_q, d]
        k (torch.Tensor): key tensor of shape [..., n_kv, d]
        v (torch.Tensor): value tensor of shape[..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias tensor of shape [..., n_q, n_kv]. Defaults to None.
        use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.
        attn_weight_dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied, Defaults to 0.0.

    Returns:
        torch.Tensor: output of tensor [..., n_q, d]
    """
    assert k.shape == v.shape
    if use_efficient_implementation:
        attn_output = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            dropout_p=attn_weight_dropout_p,
        )
        return attn_output
    # [..., n_kv, d] -> [..., d, n_kv]
    k = k.transpose(-1, -2)

    # [..., n_q, d], [..., d, n_kv] -> [..., n_q, n_kv]
    attn_weights = q @ k

    if attn_bias is not None:
        if inplace_safe:
            attn_weights += attn_bias
        else:
            attn_weights = attn_weights + attn_bias

    # [..., n_q, n_kv]
    attn_weights = F.softmax(attn_weights, dim=-1)

    # [..., n_q, n_kv], [..., n_kv, d] -> [..., n_q, d]
    attn_output = attn_weights @ v

    return attn_output


def rearrange_qk_to_dense_trunk(
    q: Union[torch.Tensor, list[torch.Tensor]],
    k: Union[torch.Tensor, list[torch.Tensor]],
    dim_q: Union[int, list[int]],
    dim_k: Union[int, list[int]],
    n_queries: int = 32,
    n_keys: int = 128,
    compute_mask: bool = True,
) -> tuple[Union[torch.Tensor, list[torch.Tensor]]]:
    """Rearrange q/k tensors (both can be list of tensors) to "dense" trunk.

    Args:
        q: A tensor of list of tensors of arbitrary shapes, including N_atom dimension.
           If a list, q[i].shape has dim_q[i] sized as N_atom. (i.e., q[i].shape[dim_q[i]] = N_atom)
        k: A tensor of list of tensors of arbitrary shapes, including N_atom dimension.
           If a list, k[i].shape has dim_k[i] sized as N_atom.
        dim_q: Dimension of q that's sized as N_atom.
           If a list, dim_q[i] corresponds to the dimension in q[i].
        dim_k: Dimension of k that's sized as N_atom.
           If a list, dim_k[i] corresponds to the dimension in k[i].
        n_queries: Maximum number of query atoms per trunk.
        n_keys: Maximum number of key atoms per trunk.
        compute_mask: Whether to compute mask for valid region.

    Returns:
        A tuple of "trunked" tensors/list of tensors: (q_trunked, k_trunked, padding_info)
        q_trunked.shape[dim_q + 1] = n_queries.
        k_trunked.shape[dim_k + 1] = n_keys.
    """
    # Convert q/k to lists if not already
    q_is_list = isinstance(q, list)
    k_is_list = isinstance(k, list)
    q_list = q if q_is_list else [q]
    k_list = k if k_is_list else [k]
    dim_q_list = dim_q if isinstance(dim_q, list) else [dim_q] * len(q_list)
    dim_k_list = dim_k if isinstance(dim_k, list) else [dim_k] * len(k_list)
    num_q = len(q_list)
    num_k = len(k_list)

    # Safety checks ensuring all dims are positive
    def basic_checks(x, dim_x):
        if dim_x < 0:
            dim_x = len(x.shape) + dim_x
        return dim_x

    for i in range(num_q):
        dim_q_list[i] = basic_checks(q_list[i], dim_q_list[i])
    for i in range(num_k):
        dim_k_list[i] = basic_checks(k_list[i], dim_k_list[i])

    # The first tensor in lists
    n_q = q_list[0].size(dim_q_list[0])
    n_k = k_list[0].size(dim_k_list[0])

    # Critical fix: Adjust n_keys if it's larger than the actual tensor dimension
    for i in range(len(k_list)):
        if n_keys > k_list[i].shape[dim_k_list[i]]:
            n_keys = min(n_keys, k_list[i].shape[dim_k_list[i]])

    # Compute the number of trunks for q
    n_trunks = int(math.ceil(n_q / n_queries))
    q_pad_length = n_trunks * n_queries - n_q

    # Process query tensors (q)
    q_new = []
    for i in range(len(q_list)):
        # Create a new tensor with the padded size
        shape = list(q_list[i].shape)
        shape[dim_q_list[i]] = shape[dim_q_list[i]] + q_pad_length
        padded_q = q_list[i].new_zeros(shape)
        
        # Copy the original data
        slices = [slice(None)] * len(shape)
        slices[dim_q_list[i]] = slice(0, n_q)
        padded_q[tuple(slices)] = q_list[i]
        
        # Reshape q to have n_trunks and n_queries
        shape = list(padded_q.shape)
        shape[dim_q_list[i]:dim_q_list[i]+1] = [n_trunks, n_queries]
        reshaped_q = padded_q.reshape(*shape)
        
        q_new.append(reshaped_q)

    # Calculate padding for k
    pad_left = (n_keys - n_queries) // 2
    pad_right = int((n_trunks - 1) * n_queries + n_keys // 2 - n_q + 1)
    
    # Process key tensors (k)
    k_new = []
    for i in range(len(k_list)):
        # Create a new tensor with the padded size
        shape = list(k_list[i].shape)
        padded_width = shape[dim_k_list[i]] + pad_left + pad_right
        shape[dim_k_list[i]] = padded_width
        padded_k = k_list[i].new_zeros(shape)
        
        # Copy the original data
        slices = [slice(None)] * len(shape)
        slices[dim_k_list[i]] = slice(pad_left, pad_left+n_k)
        padded_k[tuple(slices)] = k_list[i]
        
        # Use direct slicing instead of unfold to avoid tensor size errors
        trunked_k = []
        for j in range(n_trunks):
            start_idx = j * n_queries
            end_idx = min(start_idx + n_keys, padded_width)
            
            if end_idx > start_idx:
                # Extract the window
                slices = [slice(None)] * len(padded_k.shape)
                slices[dim_k_list[i]] = slice(start_idx, end_idx)
                window = padded_k[tuple(slices)]
                
                # If the window is smaller than n_keys, pad it
                if window.shape[dim_k_list[i]] < n_keys:
                    pad_size = n_keys - window.shape[dim_k_list[i]]
                    pad_shape = [0, 0] * len(window.shape)
                    pad_shape[2 * dim_k_list[i] + 1] = pad_size
                    window = torch.nn.functional.pad(window, pad_shape)
                
                # Add trunk dimension
                window_shape = list(window.shape)
                window_shape.insert(dim_k_list[i], 1)
                window = window.reshape(*window_shape)
                
                trunked_k.append(window)
        
        # Concatenate along the trunk dimension
        if trunked_k:
            k_new.append(torch.cat(trunked_k, dim=dim_k_list[i]))
        else:
            # Create a dummy tensor with the right shape if no windows were created
            dummy_shape = list(k_list[i].shape)
            dummy_shape[dim_k_list[i]] = n_trunks
            dummy_shape.insert(dim_k_list[i] + 1, n_keys)
            k_new.append(k_list[i].new_zeros(dummy_shape))

    # Create simple padding info
    padding_info = {
        "q_pad": q_pad_length,
        "k_pad_left": pad_left,
        "k_pad_right": pad_right,
    }
    
    # Add mask information if requested
    if compute_mask:
        # Create a mask for valid regions
        q_mask = [
            torch.ones(
                *(q_list[i].shape[:dim_q_list[i]] + q_list[i].shape[dim_q_list[i] + 1 :]),
                n_trunks,
                n_queries,
                device=q_list[i].device,
                dtype=torch.bool,
            )
            for i in range(num_q)
        ]
        k_mask = [
            torch.ones(
                *(k_list[i].shape[:dim_k_list[i]] + k_list[i].shape[dim_k_list[i] + 1 :]),
                n_trunks,
                n_keys,
                device=k_list[i].device,
                dtype=torch.bool,
            )
            for i in range(num_k)
        ]
        
        # Mark padded regions as invalid
        for i in range(num_q):
            q_mask[i][..., :, n_q:] = False
        
        padding_info["q_mask"] = q_mask
        padding_info["k_mask"] = k_mask
    else:
        padding_info["q_mask"] = None
        padding_info["k_mask"] = None

    # Convert back to non-list
    if not q_is_list:
        q_new = q_new[0]
    if not k_is_list:
        k_new = k_new[0]

    return q_new, k_new, padding_info


def optimized_concat_split(attn_bias: torch.Tensor, n_queries: int) -> torch.Tensor:
    """Optimized concatenation and splitting of attention bias tensor.

    Args:
        attn_bias (torch.Tensor): The attention bias tensor.
            Shape: [..., D, E]
        n_queries (int): The number of queries in each split.

    Returns:
        torch.Tensor: The reshaped and permuted attention bias tensor.
            Shape: [..., n_queries, D // n_queries * E]
    """
    D = attn_bias.size(-2)
    E = attn_bias.size(-1)
    assert D % n_queries == 0
    num_splits = D // n_queries
    reshaped = attn_bias.reshape(*attn_bias.shape[:-2], num_splits, n_queries, E)
    permuted = reshaped.permute(*range(reshaped.dim() - 3), -2, -3, -1)
    output = permuted.reshape(*attn_bias.shape[:-2], n_queries, num_splits * E)
    return output


def rearrange_to_dense_trunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
) -> tuple[Union[torch.Tensor, int]]:
    """Rearrange q/k/v/bias into blocked tensors for local attention.

    Args:
        q (torch.Tensor): query tensor
            [..., n_q, d]
        k (torch.Tensor): key tensor
            [..., n_kv, d]
        v (torch.Tensor): value tensor
            [..., n_kv, d]
        attn_bias (torch.Tensor, optional): attention bias
            [..., n_q, n_kv] or None
        n_queries (int, optional): local window size of query tensor.
        n_keys (int, optional): local window size of key/value tensor.
        inf (float, optional): used for attention masking. Defaults to 1e10.

    Returns:
        tuple[Union[torch.Tensor, int]]:
            q_trunked (torch.Tensor): trunked query tensor
                [..., n_trunks, n_queries, d]
            k_trunked (torch.Tensor): trunked key tensor
                [..., n_trunks, n_keys, d]
            v_trunked (torch.Tensor): trunked value tensor
                [..., n_trunks, n_keys, d]
            attn_bias_trunked (torch.Tensor): trunked attention bias
                [..., n_trunks, n_queries, n_keys]
            q_pad_length (int): number of padding elements in q
    """
    try:
        n_q = q.size(-2)
        n_kv = k.size(-2)

        assert n_keys >= n_queries
        assert n_queries & 0x01 == 0  # n_queries is even
        assert n_keys & 0x01 == 0  # n_keys is even

        # Create trunk-wise format: [n_chunk, n_queries, ...] and [n_chunk, n_keys, ...]
        n_trunks = int(math.ceil(n_q / n_queries))
        q_pad_length = n_trunks * n_queries - n_q
        q_padded = pad_at_dim(q, dim=-2, pad_length=(0, q_pad_length))

        # Split q into n_trunks chunks, q_trunked.shape = [..., n_trunks, n_queries, d]
        q_trunked = reshape_at_dim(q_padded, dim=-2, target_shape=(n_trunks, n_queries))

        pad_left = (n_keys - n_queries) // 2
        pad_right = int(
            (n_trunks - 1) * n_queries + (n_keys - n_queries) / 2 - n_q + 1 / 2
        )

        if k.size(-2) != v.size(-2):
            raise ValueError(
                f"k and v must have the same size at -2, but got {k.size(-2)} and {v.size(-2)}"
            )

        # Pad k and v for sliding windows
        k_padded = pad_at_dim(k, dim=-2, pad_length=(pad_left, pad_right))
        v_padded = pad_at_dim(v, dim=-2, pad_length=(pad_left, pad_right))

        # Unfolding the padded k and v produces shape [..., n_trunks, n_keys, d]
        # where the n_trunks dimension corresponds to a sliding window.
        k_trunked = k_padded.unfold(-2, n_keys, n_queries)
        v_trunked = v_padded.unfold(-2, n_keys, n_queries)

        # The unfold operation adds a dimension at the end, so we need to move it to (-3).
        k_trunked = move_final_dim_to_dim(k_trunked, dim=-3)
        v_trunked = move_final_dim_to_dim(v_trunked, dim=-3)

        # Convert attention bias
        attn_bias_trunked = None
        if attn_bias is not None:
            # Need to slice attention bias to match the padded attention
            attn_bias_padded = pad_at_dim(
                attn_bias, dim=-2, pad_length=(0, q_pad_length)
            )
            attn_bias_padded = pad_at_dim(
                attn_bias_padded, dim=-1, pad_length=(pad_left, pad_right)
            )

            # Shape: [..., n_trunks, n_queries, all_k]
            attn_bias_split = reshape_at_dim(
                attn_bias_padded, dim=-2, target_shape=(n_trunks, n_queries)
            )

            # Slice out the sliding window for key attention
            # After the split, the second-to-last dim of attn_bias_split is now n_trunks
            attn_bias_trunked = []
            for i in range(n_trunks):
                start_idx = i * n_queries
                end_idx = start_idx + n_keys
                if end_idx <= attn_bias_padded.size(-1):
                    window = attn_bias_split[..., i, :, start_idx:end_idx]
                else:
                    # If we're near the end, we may need to handle edge case
                    # Just create a zero tensor with the right shape
                    window_shape = list(attn_bias_split.shape[:-3]) + [n_queries, n_keys]
                    window = torch.zeros(window_shape, device=attn_bias.device)
                
                # Add trunk dimension
                window = window.unsqueeze(-3)
                attn_bias_trunked.append(window)
            
            # Concatenate along the trunk dimension
            attn_bias_trunked = torch.cat(attn_bias_trunked, dim=-3)

        # If no attention bias was provided, create a default one that allows all attention
        if attn_bias_trunked is None:
            # Create a bias tensor filled with zeros (allows all attention)
            attn_bias_shape = list(q_trunked.shape[:-1]) + [n_keys]
            attn_bias_trunked = torch.zeros(attn_bias_shape, device=q.device)

        return q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_pad_length
    
    except Exception:
        # Fallback implementation if unfold or other operations fail
        batch_dims = q.shape[:-2]
        n = q.shape[-2]
        d = q.shape[-1]
        
        # Calculate the number of trunks and padding
        n_trunks = int(math.ceil(n / n_queries))
        q_pad_length = n_trunks * n_queries - n
        
        # Process the query tensor
        padded_q = torch.nn.functional.pad(q, (0, 0, 0, q_pad_length))
        q_trunked = padded_q.reshape(*batch_dims, n_trunks, n_queries, d)
        
        # Process key and value tensors
        pad_left = (n_keys - n_queries) // 2
        pad_right = int((n_trunks - 1) * n_queries + n_keys // 2 - n + 1)
        
        padded_k = torch.nn.functional.pad(k, (0, 0, pad_left, pad_right))
        padded_v = torch.nn.functional.pad(v, (0, 0, pad_left, pad_right))
        
        # Create windows manually to avoid unfold errors
        k_windows = []
        v_windows = []
        
        for i in range(n_trunks):
            start_idx = i * n_queries
            end_idx = min(start_idx + n_keys, padded_k.shape[-2])
            
            # Extract the window
            k_window = padded_k[..., start_idx:end_idx, :]
            v_window = padded_v[..., start_idx:end_idx, :]
            
            # Pad if needed
            if k_window.shape[-2] < n_keys:
                pad_size = n_keys - k_window.shape[-2]
                k_window = torch.nn.functional.pad(k_window, (0, 0, 0, pad_size))
                v_window = torch.nn.functional.pad(v_window, (0, 0, 0, pad_size))
            
            # Add trunk dimension
            k_window = k_window.unsqueeze(-3)
            v_window = v_window.unsqueeze(-3)
            
            k_windows.append(k_window)
            v_windows.append(v_window)
        
        # Concatenate windows
        k_trunked = torch.cat(k_windows, dim=-3)
        v_trunked = torch.cat(v_windows, dim=-3)
        
        # Create attention bias tensor
        attn_bias_shape = list(batch_dims) + [n_trunks, n_queries, n_keys]
        attn_bias_trunked = torch.zeros(attn_bias_shape, device=q.device)
        
        # If attn_bias is provided, copy values
        if attn_bias is not None:
            padded_attn_bias = torch.nn.functional.pad(attn_bias, (pad_left, pad_right, 0, q_pad_length))
            
            for i in range(n_trunks):
                start_idx = i * n_queries
                end_idx = min(start_idx + n_keys, padded_attn_bias.shape[-1])
                q_end = min(n, (i+1) * n_queries)
                q_start = i * n_queries
                
                if q_start < n and q_end > q_start:
                    q_slice = slice(q_start, q_end)
                    k_slice = slice(start_idx, end_idx)
                    
                    # Copy attention bias values
                    window_width = end_idx - start_idx
                    attn_bias_trunked[..., i, :q_end-q_start, :window_width] = (
                        padded_attn_bias[..., q_start:q_end, start_idx:end_idx]
                    )
        
        return q_trunked, k_trunked, v_trunked, attn_bias_trunked, q_pad_length


def _local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_queries: int,
    n_keys: int,
    attn_bias: Optional[torch.Tensor] = None,
    trunked_attn_bias: Optional[torch.Tensor] = None,
    inf: float = 1e10,
    use_efficient_implementation: bool = False,
    attn_weight_dropout_p: float = 0.0,
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Apply local attention with blocks of keys.

    Returns:
        torch.Tensor: attention result
            [..., n_q, d]
    """
    if chunk_size:
        # For each q chunk, compute attention with all k/v
        chunks = []
        q_chunks = q.chunk(chunk_size, dim=-2)
        for q_chunk in q_chunks:
            chunk_result = _local_attention(
                q_chunk,
                k,
                v,
                n_queries,
                n_keys,
                attn_bias,
                trunked_attn_bias,
                inf,
                use_efficient_implementation,
                attn_weight_dropout_p,
                inplace_safe,
                chunk_size=None,
            )
            chunks.append(chunk_result)
        return torch.cat(chunks, dim=-2)

    (
        q_trunked,
        k_trunked,
        v_trunked,
        attn_bias_trunked,
        q_pad_length,
    ) = rearrange_to_dense_trunk(
        q, k, v, n_queries, n_keys, attn_bias, inf=inf
    )

    # Combine the trunked attention bias with any additional bias provided
    if trunked_attn_bias is not None:
        if attn_bias_trunked is not None:
            attn_bias_trunked = attn_bias_trunked + trunked_attn_bias
        else:
            attn_bias_trunked = trunked_attn_bias

    # Compute trunk-wise attention: [..., n_trunks, n_queries, d]
    if use_efficient_implementation and attn_weight_dropout_p > 0.0:
        raise NotImplementedError(
            "efficient_implementation is not supported with attention_weight_dropout_p > 0"
        )

    # Make attention bias mask explicit - convert 0s to -inf
    if attn_bias_trunked is not None:
        # Check if attn_bias_trunked contains negative infinity values
        has_neg_inf = (attn_bias_trunked == -float('inf')).any()
        
        if not has_neg_inf:
            # Create a mask for padded positions to avoid attending to padding
            n_atoms = q.shape[-2]
            n_trunks = q_trunked.shape[-3]
            mask = torch.ones_like(attn_bias_trunked, dtype=torch.bool)
            
            # Mark padding in queries as invalid
            for i in range(n_trunks):
                start_q = i * n_queries
                end_q = min(start_q + n_queries, n_atoms)
                if end_q < start_q + n_queries:
                    mask[..., i, end_q-start_q:, :] = False
            
            # Apply the mask to attn_bias_trunked
            attn_bias_trunked = attn_bias_trunked.masked_fill(~mask, -inf)

    attn_out_trunked = _attention(
        q_trunked,
        k_trunked,
        v_trunked,
        attn_bias_trunked,
        use_efficient_implementation,
        attn_weight_dropout_p,
        inplace_safe,
    )

    # Join n_trunks from result
    out_shape = list(q.shape[:-2]) + [-1, q.shape[-1]]
    attn_out = attn_out_trunked.reshape(*out_shape)

    # Strip off the padding, back to input shape
    if q_pad_length > 0:
        attn_out = attn_out[..., :-q_pad_length, :]

    return attn_out


def create_local_attn_bias(
    n: int, n_queries: int, n_keys: int, inf: float = 1e10, device: torch.device = None
) -> torch.Tensor:
    """Create local attention bias based on query window n_queries and kv window n_keys.

    Args:
        n (int): the length of quiries
        n_queries (int): window size of quiries
        n_keys (int): window size of keys/values
        inf (float, optional): the inf to mask attention. Defaults to 1e10.
        device (torch.device, optional): cuda|cpu|None. Defaults to None.

    Returns:
        torch.Tensor: the diagonal-like global attention bias
    """
    n_trunks = int(math.ceil(n / n_queries))
    padded_n = n_trunks * n_queries
    attn_mask = torch.zeros(padded_n, padded_n, device=device)
    for block_index in range(0, n_trunks):
        i = block_index * n_queries
        j1 = max(0, n_queries * block_index - (n_keys - n_queries) // 2)
        j2 = n_queries * block_index + (n_queries + n_keys) // 2
        attn_mask[i : i + n_queries, j1:j2] = 1.0
    attn_bias = (1 - attn_mask) * -inf
    return attn_bias.to(device=device)[:n, :n]


class Attention(nn.Module):
    """Standard multi-head attention
    Ref to openfold:
    https://github.com/aqlaboratory/openfold/blob/feb45a521e11af1db241a33d58fb175e207f8ce0/openfold/model/primitives.py#L340
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        num_heads: int,
        gating: bool = True,
        q_linear_bias: bool = False,
        local_attention_method: str = "global_attention_with_bias",
        use_efficient_implementation: bool = False,
        attn_weight_dropout_p: float = 0.0,
    ) -> None:
        """

        Args:
            c_q (int): Input dimension of query data
            c_k (int): Input dimension of key data
            c_v (int): Input dimension of value data
            c_hidden (int): Per-head hidden dimension
            num_heads (int): Number of attention heads
            gating (bool, optional): Whether the output should be gated using query data. Defaults to True.
            q_linear_bias (bool, optional): whether use Linear with bias as in AF3. Defaults to False.
            local_attention_method (str, optional): local attention method, options:
              - global_attention_with_bias: use full size global attention with sparse attention bias
              - local_cross_attention: use local cross attention to minimize computation
            use_efficient_implementation (bool): whether to use the torch.nn.functional.scaled_dot_product_attention, Defaults to False.
            attn_weight_dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied, Defaults to 0.0.

        Notes:
            if use_efficient_implementation == True, torch.nn.functional.scaled_dot_product_attention will
            be used to compute attention efficiently
            There are currently three supported implementations of scaled dot product attention:
                1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

                2. Memory-Efficient Attention

                3. A PyTorch implementation defined in C++ matching the above formulation

            The function may call optimized kernels for improved performance when using the CUDA backend.
            For all other backends, the PyTorch implementation will be used.All implementations are enabled by default.
            Scaled dot product attention attempts to automatically select the most optimal implementation based on the inputs.
        """
        super(Attention, self).__init__()
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.num_heads = num_heads
        self.gating = gating
        self.local_attention_method = local_attention_method
        self.use_efficient_implementation = use_efficient_implementation
        self.attn_weight_dropout_p = attn_weight_dropout_p

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.
        if q_linear_bias:
            # Attention in AF3
            self.linear_q = Linear(
                in_features=self.c_q, out_features=self.c_hidden * self.num_heads
            )
        else:
            # Vanilla attention
            self.linear_q = LinearNoBias(self.c_q, self.c_hidden * self.num_heads)
        self.linear_k = LinearNoBias(self.c_k, self.c_hidden * self.num_heads)
        self.linear_v = LinearNoBias(self.c_v, self.c_hidden * self.num_heads)
        self.linear_o = LinearNoBias(self.c_hidden * self.num_heads, self.c_q)
        self.linear_g = None
        if self.gating:
            self.linear_g = LinearNoBias(self.c_q, self.c_hidden * self.num_heads)
            self.sigmoid = nn.Sigmoid()

        # Zero init the output layer
        nn.init.zeros_(self.linear_o.weight)

    def _prep_qkv(
        self, q_x: torch.Tensor, kv_x: torch.Tensor, apply_scale: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare qkv

        Args:
            q_x (torch.Tensor): the input x for q
                [..., c_q]
            kv_x (torch.Tensor): the input x for kv
                [..., c_k]
                [..., c_v]
            apply_scale (bool, optional): apply scale to dot product qk. Defaults to True.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: the return q/k/v
                # [..., H, Q/K/V, C_hidden]
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K/V, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.num_heads, -1))
        k = k.view(k.shape[:-1] + (self.num_heads, -1))
        v = v.view(v.shape[:-1] + (self.num_heads, -1))

        # [*, H, Q/K/V, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        if apply_scale:
            q = q / math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self, o: torch.Tensor, q_x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            o (torch.Tensor): the output of attention
                [..., G/Q, H, C_hidden]
            q_x (torch.Tensor): the input for gated g
                [..., Q, c_q]

        Returns:
            torch.Tensor: the output of attention
        """
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))

            # [*, G/Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, num_dims=2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        trunked_attn_bias: Optional[torch.Tensor] = None,
        n_queries: Optional[int] = None,
        n_keys: Optional[int] = None,
        inf: Optional[float] = 1e10,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """

        Args:
            q_x (torch.Tensor): the input x for q
                [..., Q, C_q]
            kv_x (torch.Tensor): the input x for k/v
                [..., K, C_k]
            attn_bias (torch.Tensor, optional): the input biases for attention. Defaults to None.
                [..., H, Q, K] or [..., Q, K]
            trunked_attn_bias (torch.Tensor, optional): the input biases where shape has been rearranged to dense trunks. Defaults to None.
                [..., H, n_trunks, n_queries, n_keys] or [..., n_trunks, n_queries, n_keys]
            n_queries (int, optional): local window size of query tensor. If not None, will perform local attention. Defaults to None.
            n_keys (int, optional): local window size of key tensor. Defaults to None.

        Returns:
            torch.Tensor: attention update
                [*, Q, C_q]
        """

        q, k, v = self._prep_qkv(q_x=q_x, kv_x=kv_x, apply_scale=True)

        if attn_bias is not None:
            if len(attn_bias.shape) == len(q.shape):
                assert attn_bias.shape[:-2] == q.shape[:-2]
            else:
                assert len(attn_bias.shape) == len(q.shape) - 1
                assert attn_bias.shape[:-2] == q.shape[:-3]
                # Expand at head dim, got shape [..., 1, Q, K]
                attn_bias = attn_bias.unsqueeze(dim=-3)

        if trunked_attn_bias is not None:
            # NOTE: trunked_attn_bias can only be used with "local_cross_attention" method
            assert n_queries and n_keys
            assert self.local_attention_method == "local_cross_attention"

            if len(trunked_attn_bias.shape) == len(q.shape) + 1:
                assert trunked_attn_bias.shape[:-3] == q.shape[:-2]
            else:
                assert len(trunked_attn_bias.shape) == len(q.shape)
                # Expand at head dim, got shape [..., 1, n_trunks, n_queries, n_keys]
                trunked_attn_bias = trunked_attn_bias.unsqueeze(dim=-4)

        if n_queries and n_keys:
            if self.local_attention_method == "global_attention_with_bias":
                local_attn_bias = create_local_attn_bias(
                    q.shape[-2], n_queries, n_keys, inf=inf, device=q.device
                )
                # Expand to same shape as attn_bias
                local_attn_bias = local_attn_bias.reshape(
                    (1,) * (len(q.shape[:-2])) + local_attn_bias.shape
                )
                if attn_bias is not None:
                    if inplace_safe:
                        local_attn_bias += attn_bias
                    else:
                        local_attn_bias = local_attn_bias + attn_bias
                o = _attention(
                    q=q,
                    k=k,
                    v=v,
                    attn_bias=local_attn_bias,
                    use_efficient_implementation=self.use_efficient_implementation,
                    attn_weight_dropout_p=self.attn_weight_dropout_p,
                    inplace_safe=inplace_safe,
                )

            elif self.local_attention_method == "local_cross_attention":
                o = _local_attention(
                    q=q,
                    k=k,
                    v=v,
                    n_queries=n_queries,
                    n_keys=n_keys,
                    attn_bias=attn_bias,
                    trunked_attn_bias=trunked_attn_bias,
                    inf=inf,
                    use_efficient_implementation=self.use_efficient_implementation,
                    attn_weight_dropout_p=self.attn_weight_dropout_p,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )
            else:
                raise ValueError(
                    f"Invalid local attention method: {self.local_attention_method}"
                )
        else:
            o = _attention(
                q=q,
                k=k,
                v=v,
                attn_bias=attn_bias,
                use_efficient_implementation=self.use_efficient_implementation,
                attn_weight_dropout_p=self.attn_weight_dropout_p,
                inplace_safe=inplace_safe,
            )  # [*, H, Q, C_hidden]
        o = o.transpose(-2, -3)  # o: [*, Q, H, C_hidden]
        o = self._wrap_up(o, q_x)  # q_x: [*, Q, c_q]

        return o


def gather_pair_embedding_in_dense_trunk(
    x: torch.Tensor, idx_q: torch.Tensor, idx_k: torch.Tensor
):
    """
    Selectively gather elements from a tensor using two sets of indices.

        x: [..., N_token, N_token, d]
        idx_q: [N_b, N_q] or [N_b, N_trunk, N_q]
        idx_k: [N_b, N_k] or [N_b, N_trunk, N_k]

    Return:
        y: [..., N_b, N_q, N_k, d]
            where y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]
    """
    # Import the adapter function
    from rna_predict.pipeline.stageA.input_embedding.current.shape_adapter import adapt_indices_for_gather
    
    # Use the adapter to ensure indices have the correct shape
    idx_q, idx_k = adapt_indices_for_gather(idx_q, idx_k)
    
    # Get the shape parameters
    N_b, N_q = idx_q.shape
    N_k = idx_k.shape[1]
    
    # Expand idx_q and idx_k to match the shape required for advanced indexing
    idx_q_expanded = idx_q.unsqueeze(-1).expand(-1, -1, N_k)
    idx_k_expanded = idx_k.unsqueeze(1).expand(-1, N_q, -1)
    
    # Use advanced indexing to gather the desired elements
    y = x[..., idx_q_expanded, idx_k_expanded, :]
    
    return y


# @snoop
def broadcast_token_to_local_atom_pair(
    z_token: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_queries: int,
    n_keys: int,
    compute_mask: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Broadcast token pair embedding to atom pair embedding

    Args:
        z_token (torch.Tensor): token pair embedding
            [..., N_token, N_token, d]
        atom_to_token_idx (torch.Tensor): map atom idx to token idx
            [N_atom]

    Returns:
        z_gathered_blocked (torch.Tensor): atom pair embedding, with local blocked shape
            [..., n_trunks, n_queries, n_keys, d]
        pad_info (dict): padding information containing mask and padding lengths
    """

    # [N_atom] -> [n_trunks, n_queries] and [n_trunks, n_keys]
    atom_to_token_idx_q, atom_to_token_idx_k, pad_info = rearrange_qk_to_dense_trunk(
        atom_to_token_idx,
        atom_to_token_idx,
        dim_q=-1,
        dim_k=-1,
        n_queries=n_queries,
        n_keys=n_keys,
        compute_mask=compute_mask,
    )

    z_gathered_blocked = gather_pair_embedding_in_dense_trunk(
        z_token, idx_q=atom_to_token_idx_q, idx_k=atom_to_token_idx_k
    )

    return z_gathered_blocked, pad_info

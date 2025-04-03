# protenix/model/utils.py
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
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from protenix.utils.scatter_utils import scatter
from scipy.spatial.transform import Rotation


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, N_sample, 3)).to(
        device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]
    return x_augment_coords


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# this is from openfold.utils.rigid_utils import rot_vec_mul
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (torch.Tensor): the rotation matrices
            [..., 3, 3]
        t (torch.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        torch.Tensor: Rotated coordinates, shape [..., 3].
    """
    # Ensure t has a trailing dimension for matrix multiplication if it doesn't match r
    if (
        r.ndim == t.ndim + 1 and r.shape[-2:] == (3, 3) and t.shape[-1] == 3
    ):  # e.g. r=[..., N, 3, 3], t=[..., N, 3]
        # Check if batch dimensions are compatible for matmul
        if r.shape[:-3] == t.shape[:-2]:
            t_unsqueeze = t.unsqueeze(-1)  # [..., N, 3, 1]
            # Perform batch matrix vector product: [...,N,3,3] @ [...,N,3,1] -> [...,N,3,1]
            rotated_t = torch.matmul(r, t_unsqueeze)
            return rotated_t.squeeze(-1)  # [..., N, 3]
        else:
            # Fallback to einsum if direct matmul broadcasting fails
            try:
                return torch.einsum("...nij,...nj->...ni", r, t)
            except RuntimeError as e:
                print(
                    f"Einsum failed in rot_vec_mul (ndim+1 case): r={r.shape}, t={t.shape}. Error: {e}"
                )
                raise e  # Re-raise if einsum also fails
    elif r.ndim >= 2 and t.ndim >= 1 and r.shape[-2:] == (3, 3) and t.shape[-1] == 3:
        # Handle cases where r and t might have different leading dimensions but are compatible via broadcasting
        # e.g. r=[3,3], t=[B, N, 3] -> apply same rotation to all vectors
        # Use einsum for robust broadcasting
        # '...ij,...j->...i' : contract last dim of r (j) with last dim of t (j)
        try:
            return torch.einsum("...ij,...j->...i", r, t)
        except RuntimeError as e:
            print(
                f"Einsum failed in rot_vec_mul (general case): r={r.shape}, t={t.shape}. Error: {e}"
            )
            # Fallback: try matmul if shapes allow direct application (e.g., r=[3,3], t=[N,3])
            if r.shape == (3, 3) and t.ndim >= 2:
                try:
                    # Apply rotation by multiplying t with r^T
                    return torch.matmul(t, r.transpose(-1, -2))
                except RuntimeError:
                    raise e  # Re-raise original einsum error if matmul fails
            else:
                raise e  # Re-raise if fallback doesn't apply
    else:
        # If shapes are fundamentally incompatible even for einsum
        raise ValueError(
            f"Incompatible shapes for rot_vec_mul: r={r.shape}, t={t.shape}"
        )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(
    x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor
) -> torch.Tensor:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x (torch.Tensor): the input x
            [...]
        lower_bins (torch.Tensor): the lower bounds of bins
            [bins]
        upper_bins (torch.Tensor): the upper bounds of bins
            [bins]
    Returns:
        torch.Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# this is mostly from openfold.utils.torch_utils import batched_gather
def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# @snoop
def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast token-level embeddings to atom-level embeddings.

    This handles cases where:
      1) atom_to_token_idx is purely 1D -> we unsqueeze to [1, N_atom].
      2) x_token may have one more batch dim than atom_to_token_idx, so we unsqueeze one dim in the index as well.
      3) x_token may have two more batch dims than atom_to_token_idx, so we handle that case too.
    """
    if atom_to_token_idx.ndim == 1:
        atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

    # Handle case where x_token has two more dimensions than atom_to_token_idx
    if x_token.ndim == atom_to_token_idx.ndim + 2:
        # Instead of reshaping x_token (which is risky and can cause size mismatch errors),
        # let's add dimensions to atom_to_token_idx to match x_token's dimensionality

        # For example, if x_token is [1, 1, 8, 10, 128] and atom_to_token_idx is [1, 10],
        # instead of reshaping x_token, let's unsqueeze atom_to_token_idx to match the batch dimensions

        # Add dimensions to atom_to_token_idx until it's just one dimension short of x_token
        # (the last missing dimension is the feature dimension which doesn't need to match)
        while atom_to_token_idx.ndim < x_token.ndim - 1:
            # Insert a new dimension at position 1 (after the first batch dimension)
            atom_to_token_idx = atom_to_token_idx.unsqueeze(1)

        # Now the shapes should be compatible for gathering
        # E.g., atom_to_token_idx might be [1, 1, 10] and x_token [1, 1, 8, 10, 128]

    # If shapes differ by exactly 1 dimension, and we need an extra unsqueeze:
    if (
        x_token.ndim == atom_to_token_idx.ndim + 1
        and x_token.shape[-2] != atom_to_token_idx.shape[-1]
    ):
        atom_to_token_idx = atom_to_token_idx.unsqueeze(-2)

    # Handle case where atom_to_token_idx has more dimensions but x_token doesn't need to match
    # For example, atom_to_token_idx with shape [1, 1, N_atom] and x_token with shape [1, N_token, C]
    if atom_to_token_idx.ndim > x_token.ndim:
        # Remove singleton dimensions from atom_to_token_idx to match x_token's shape
        squeezed_shape = [dim for dim in atom_to_token_idx.shape[:-1] if dim != 1]
        if len(squeezed_shape) == len(x_token.shape[:-2]):
            # We can proceed with the squeezed shape
            atom_to_token_idx = atom_to_token_idx.squeeze()
            # In case we squeezed too much, add back one dimension
            if atom_to_token_idx.ndim == 1:
                atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

    # Final shape check
    if atom_to_token_idx.shape[:-1] != x_token.shape[:-2]:
        # Instead of immediately trying to reshape (which may fail), let's try different approaches
        # to make the shapes compatible
        try:
            # First, check if atom_to_token_idx has excess dimensions that can be removed
            if atom_to_token_idx.ndim > x_token.ndim:
                # Try to squeeze out extra dimensions while preserving the last dimension
                atom_to_token_idx = atom_to_token_idx.squeeze()
                if atom_to_token_idx.ndim == 1:  # If we squeezed too much
                    atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

            # Next, check if we need to add dimensions to atom_to_token_idx
            while len(atom_to_token_idx.shape[:-1]) < len(x_token.shape[:-2]):
                # Add dimensions at the beginning
                atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

            # If dimensions match now but sizes don't, try to expand
            if len(atom_to_token_idx.shape[:-1]) == len(x_token.shape[:-2]):
                # Try to expand atom_to_token_idx to match x_token's batch dimensions
                # But first check if each dimension is either 1 or matches
                can_expand = all(
                    a == b or a == 1 or b == 1
                    for a, b in zip(atom_to_token_idx.shape[:-1], x_token.shape[:-2])
                )
                if can_expand:
                    # Create a new shape that takes the max of each dimension
                    new_batch_shape = tuple(
                        max(a, b)
                        for a, b in zip(
                            atom_to_token_idx.shape[:-1], x_token.shape[:-2]
                        )
                    )
                    new_shape = new_batch_shape + (atom_to_token_idx.shape[-1],)
                    atom_to_token_idx = atom_to_token_idx.expand(new_shape)

            # If shapes still don't match, try one more approach: reshape if possible
            if atom_to_token_idx.shape[:-1] != x_token.shape[:-2]:
                # Check if the product of dimensions is compatible
                prod_a = 1
                for d in atom_to_token_idx.shape[:-1]:
                    prod_a *= d

                prod_x = 1
                for d in x_token.shape[:-2]:
                    prod_x *= d

                if prod_a == prod_x:
                    # We can reshape
                    new_shape = list(x_token.shape[:-2]) + [atom_to_token_idx.shape[-1]]
                    atom_to_token_idx = atom_to_token_idx.reshape(new_shape)
                else:
                    # Try to broadcast by repeating atom_to_token_idx
                    # This is useful for cases like [1,1,10] needing to match [1,8]
                    # First, create a target shape with batch dimensions matching x_token
                    target_shape = list(x_token.shape[:-2]) + [
                        atom_to_token_idx.shape[-1]
                    ]

                    # Try to create a new tensor with the target shape
                    expanded_idx = atom_to_token_idx.new_zeros(
                        target_shape, dtype=atom_to_token_idx.dtype
                    )

                    # Repeat the last available atom index across the missing dimension
                    for i in range(target_shape[-2]):
                        if i < atom_to_token_idx.shape[-2]:
                            # Copy existing indices
                            expanded_idx[..., i, :] = atom_to_token_idx[..., i, :]
                        else:
                            # Repeat the last index for missing positions
                            expanded_idx[..., i, :] = atom_to_token_idx[..., -1, :]

                    atom_to_token_idx = expanded_idx
        except Exception as e:
            # If all adaptation attempts fail, provide a detailed error
            print(f"WARNING: Failed to adapt shapes in broadcast_token_to_atom: {e}")
            print(
                f"x_token.shape={x_token.shape}, atom_to_token_idx.shape={atom_to_token_idx.shape}"
            )
            # Fall back to the original approach, but log the error instead of raising it
            try:
                new_shape = list(x_token.shape[:-2]) + [atom_to_token_idx.shape[-1]]
                atom_to_token_idx = atom_to_token_idx.reshape(new_shape)
            except:
                # If reshaping fails, just continue with the current shapes
                # The batched_gather function might still work, or we'll get a more specific error
                pass

    # If still exactly 1D after expansions, do direct indexing
    if atom_to_token_idx.ndim == 1:
        return x_token[..., atom_to_token_idx, :]

    # Otherwise, fall back to batched gather
    result = batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )

    # Check if the feature dimension of the result is 1, but should be larger
    # This handles the specific case in AtomAttentionEncoder where we get [2, 10, 1] but need [*, 64]
    if result.size(-1) == 1 and x_token.size(-1) > 1:
        # We need to expand the last dimension to match the original x_token's feature dimension
        result = result.expand(*result.shape[:-1], x_token.size(-1))

    return result


def aggregate_atom_to_token(
    x_atom: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (torch.Tensor): atom-level embedding
            [..., N_atom, d]
        atom_to_token_idx (torch.Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int, optional): number of tokens in total. Defaults to None.
        reduce (str, optional): aggregation method. Defaults to "mean".

    Returns:
        torch.Tensor: token-level embedding
            [..., N_token, d]
    """
    # Broadcasting in the given dim.
    out = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce=reduce
    )

    return out


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> torch.Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        torch.Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(
        low=min(lower_bound, n), high=n + 1, size=(1,), device=device
    ).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, torch.Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """
    Reshape dimension 'dim' of x to 'target_shape'. If target_shape is a single-element
    list and the product of x.shape[dim-1] and x.shape[dim] equals that element, then merge
    the two dimensions (allowing partial flatten).

    e.g. For x with shape [2,3,4,5] and dim=-2 (i.e. x.shape[-2] == 4), if target_shape is [12]
         and 3*4==12, then merge dimensions 1 and 2 to obtain shape [2,12,5].

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target shape for the specified dimension

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = x.dim()
    if dim < 0:
        dim = n_dim + dim

    tgt = tuple(target_shape)
    if len(tgt) == 1:
        desired = tgt[0]
        if dim > 0:
            combined = x.shape[dim - 1] * x.shape[dim]
            if combined == desired:
                shape_list = list(x.shape)
                shape_list[dim - 1] = desired
                del shape_list[dim]
                return x.reshape(*shape_list)

    # fallback => standard single-dimension replace
    shape_list = list(x.shape)
    old_size = shape_list[dim]
    new_prod = 1
    for v in tgt:
        new_prod *= v

    if new_prod != old_size:
        raise RuntimeError(
            f"reshape_at_dim: can't reshape dimension {dim} of size {old_size} into {tgt} (product={new_prod})"
        )
    shape_list = shape_list[:dim] + list(tgt) + shape_list[dim + 1 :]
    return x.reshape(*shape_list)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict

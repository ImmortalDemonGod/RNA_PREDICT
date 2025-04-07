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
import warnings
from typing import Dict, List, Optional, Union

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
    mask: Optional[torch.Tensor] = None,
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
        # Ensure mask has compatible dimensions for broadcasting
        while mask.ndim < x_input_coords.ndim - 1:
            mask = mask.unsqueeze(0)
        if mask.shape[:-1] != x_input_coords.shape[:-2]:
            # Attempt to expand mask batch dims to match x_input_coords
            try:
                mask = mask.expand(*x_input_coords.shape[:-2], -1)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Cannot expand mask shape {mask.shape} to match coords {x_input_coords.shape} for centering. Error: {e}"
                ) from e

        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1, keepdim=True) + eps
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
    x_augment_coords = rot_vec_mul(
        r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
    ) + trans_random.unsqueeze(-2)  # [..., N_sample, N_atom, 3]
    return x_augment_coords


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


def permute_final_dims(tensor: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    """
    Permutes the final dimensions of a tensor.
    """
    return tensor.permute(
        *range(len(tensor.shape) - len(permutation)),
        *[i + len(tensor.shape) - len(permutation) for i in permutation],
    )


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
        r = torch.arange(s, device=data.device)  # Ensure range is on correct device
        # Adjust view based on the number of dimensions in inds, not hardcoded 1
        view_shape = [1] * len(inds.shape)
        view_shape[i] = -1  # Place the range size at the correct batch dimension
        # Ensure view_shape has enough dimensions before assigning
        while len(view_shape) < len(inds.shape):
            view_shape.append(1)
        r = r.view(*view_shape)
        ranges.append(r)

    remaining_dims = [slice(None)] * (len(data.shape) - no_batch_dims)
    # Calculate the index relative to the remaining dimensions
    # dim is the index in the original tensor after batch dims
    relative_dim = dim - no_batch_dims
    # Handle negative dim relative to the original tensor's non-batch part
    if dim < 0:
        original_non_batch_rank = len(data.shape) - no_batch_dims
        relative_dim = original_non_batch_rank + dim - no_batch_dims

    if 0 <= relative_dim < len(remaining_dims):
        # Ensure inds is broadcastable to the slice it's replacing
        # This typically means inds needs the same number of leading dimensions
        # as the number of dimensions covered by 'ranges'.
        # We'll rely on PyTorch's advanced indexing broadcasting here.
        # If inds has fewer dims than ranges, it should broadcast correctly.
        # If inds has more dims, it might indicate an issue.
        if inds.ndim > len(ranges) + 1:  # +1 for the dimension being indexed
            warnings.warn(
                f"batched_gather: inds.ndim ({inds.ndim}) > expected ({len(ranges) + 1}). Broadcasting might be unexpected."
            )
        remaining_dims[relative_dim] = inds
    else:
        raise IndexError(
            f"Dimension {dim} (relative index {relative_dim}) out of range for remaining dimensions of length {len(remaining_dims)}"
        )

    # Construct the final index tuple
    final_inds = tuple(ranges + remaining_dims)

    return data[final_inds]


def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast token-level embeddings to atom-level embeddings using gather.
    Handles arbitrary leading batch/sample dimensions.

    Args:
        x_token (torch.Tensor): Token features [..., N_token, C]
        atom_to_token_idx (torch.Tensor): Index map [..., N_atom]

    Returns:
        torch.Tensor: Atom features [..., N_atom, C]
    """
    # --- Start Refactor ---
    # Store original leading dimensions and N_atom, C
    original_leading_dims = x_token.shape[:-2]
    n_atom = atom_to_token_idx.shape[-1]
    n_features = x_token.shape[-1]
    n_token = x_token.shape[-2]  # Get N_token
    device = x_token.device

    # Flatten leading dimensions
    x_token_flat = x_token.reshape(-1, n_token, n_features)
    b_flat = x_token_flat.shape[0]

    # Ensure atom_to_token_idx has compatible leading dims before flattening
    # Add missing leading dims to atom_to_token_idx if necessary
    idx_leading_dims = atom_to_token_idx.shape[:-1]
    if idx_leading_dims != original_leading_dims:
        # Check if expansion is possible
        if len(idx_leading_dims) <= len(original_leading_dims):
            can_expand = all(
                i_s == o_s or i_s == 1
                for i_s, o_s in zip(idx_leading_dims, original_leading_dims)
            )
            if can_expand:
                try:
                    atom_to_token_idx = atom_to_token_idx.expand(
                        *original_leading_dims, n_atom
                    )
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Cannot expand atom_to_token_idx shape {atom_to_token_idx.shape} to match x_token leading dims {original_leading_dims}. Error: {e}"
                    ) from e
            else:
                raise ValueError(
                    f"Cannot expand atom_to_token_idx shape {atom_to_token_idx.shape} to match x_token leading dims {original_leading_dims}."
                )
        else:  # Index has more leading dims than token
            raise ValueError(
                f"atom_to_token_idx shape {atom_to_token_idx.shape} has more leading dims than x_token {x_token.shape}."
            )

    atom_to_token_idx_flat = atom_to_token_idx.reshape(b_flat, n_atom)

    # Clamp indices to be within the valid range [0, N_token - 1]
    if atom_to_token_idx_flat.numel() > 0:  # Avoid error on empty tensor
        max_idx = atom_to_token_idx_flat.max()
        if max_idx >= n_token:
            warnings.warn(
                f"Clipping atom_to_token_idx: max index {max_idx} >= N_token {n_token}. "
                f"Original index shape: {atom_to_token_idx.shape}, Token shape: {x_token.shape}"
            )
            atom_to_token_idx_flat = torch.clamp(atom_to_token_idx_flat, 0, n_token - 1)
        # Also clamp lower bound just in case
        min_idx = atom_to_token_idx_flat.min()
        if min_idx < 0:
            warnings.warn(f"Clipping atom_to_token_idx: min index {min_idx} < 0.")
            atom_to_token_idx_flat = torch.clamp(atom_to_token_idx_flat, min=0)

    # Perform gather on flattened tensors
    # Need to expand atom_to_token_idx_flat to match feature dim for gather
    # Shape required by gather: [B_flat, N_atom, C]
    idx_expanded = atom_to_token_idx_flat.unsqueeze(-1).expand(
        b_flat, n_atom, n_features
    )

    # Gather using the expanded index
    try:
        # Gather along the N_token dimension (dim=1)
        x_atom_flat = torch.gather(x_token_flat, 1, idx_expanded)
    except RuntimeError as e:
        raise RuntimeError(
            f"torch.gather failed in broadcast_token_to_atom. "
            f"x_token_flat shape: {x_token_flat.shape}, "
            f"idx_expanded shape: {idx_expanded.shape}. Error: {e}"
        ) from e

    # Reshape back to original leading dimensions
    x_atom = x_atom_flat.reshape(*original_leading_dims, n_atom, n_features)

    return x_atom
    # --- End Refactor ---


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
    # Squeeze last dim of index if it's 1 and index has more than 1 dimension
    if atom_to_token_idx.ndim > 1 and atom_to_token_idx.shape[-1] == 1:
        atom_to_token_idx = atom_to_token_idx.squeeze(-1)

    # Ensure index has compatible leading dimensions with x_atom's non-feature dimensions
    # Expected shapes: x_atom [..., N_atom, d], atom_to_token_idx [..., N_atom]
    idx_shape = atom_to_token_idx.shape  # Shape of index up to N_atom dim
    atom_prefix_shape = x_atom.shape[:-1]  # Shape of x_atom up to N_atom dim

    if idx_shape != atom_prefix_shape:
        # Check if expansion is possible (index dims must be broadcastable to atom prefix dims)
        try:
            # This will raise an error if shapes are not broadcast compatible
            target_idx_shape = torch.broadcast_shapes(idx_shape, atom_prefix_shape)
            # If compatible, expand index to match atom prefix dims for scatter operation
            atom_to_token_idx = atom_to_token_idx.expand(target_idx_shape)
        except RuntimeError as e:
            raise ValueError(
                f"Cannot broadcast atom_to_token_idx shape {idx_shape} to match x_atom prefix shape {atom_prefix_shape} for scatter. Error: {e}"
            ) from e
        # Note: Removed the check `if len(idx_shape) <= len(atom_prefix_shape):` as torch.broadcast_shapes handles it.
        # Also removed the `else` block raising error for index having more leading dims, as broadcast_shapes covers this.

    # Determine the scatter dimension (the N_atom dimension)
    # This should be the dimension *before* the feature dimension in x_atom
    scatter_dim = x_atom.ndim - 2

    # Perform scatter operation
    out = scatter(
        src=x_atom,
        index=atom_to_token_idx,
        dim=scatter_dim,
        dim_size=n_token,
        reduce=reduce,
    )

    return out


def sample_indices(
    n: int,
    sample_size: int,
    strategy: str = "random",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Sample indices using specified strategy.

    Args:
        n: Total number of indices to sample from
        sample_size: Number of indices to sample
        strategy: Sampling strategy ('random' or 'topk')
        device: Device to place the output tensor on

    Returns:
        Tensor of sampled indices
    """
    assert strategy in ["random", "topk"], f"Invalid sampling strategy: {strategy}"
    assert sample_size <= n, f"Cannot sample {sample_size} items from {n} items"

    if strategy == "random":
        # Ensure n is positive for randperm
        if n <= 0:
            return torch.tensor([], dtype=torch.long, device=device)
        indices = torch.randperm(n=n, device=device)[:sample_size]
    elif strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    else:
        raise ValueError(f"Invalid sampling strategy: {strategy}")
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: Dict[str, torch.Tensor],
    sample_size: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sample MSA features randomly without replacement.
    """
    n_seq = next(iter(feat_dict.values())).shape[0]
    indices = sample_indices(n_seq, sample_size, strategy="random", device=device)

    return {k: v[indices] for k, v in feat_dict.items()}


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
    # Recalculate dim relative to the new tensor shape after unsqueeze
    actual_dim = dim if dim >= 0 else x.dim() + dim

    before_shape = x.shape[:actual_dim]
    after_shape = x.shape[actual_dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int, int], list[int]],
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
    if not (-n_dim <= dim < n_dim):  # Added closing parenthesis
        raise IndexError(
            f"Dimension out of range (expected to be in range of [-{n_dim}, {n_dim - 1}], but got {dim})"
        )

    actual_dim = dim if dim >= 0 else n_dim + dim

    # PyTorch pad expects pairs of (left, right) padding for each dimension, starting from the last
    # We only want to pad along the specified dimension 'actual_dim'
    pad_tuple = [0] * (2 * n_dim)
    # Calculate the position for the padding values in the tuple
    # Example: n_dim=4, actual_dim=1 => target indices 4, 5
    # Example: n_dim=4, actual_dim=-2 (i.e., 2) => target indices 2, 3
    pad_idx_start = 2 * (n_dim - 1 - actual_dim)
    pad_tuple[pad_idx_start] = pad_length[0]  # Left padding
    pad_tuple[pad_idx_start + 1] = pad_length[1]  # Right padding

    if tuple(pad_tuple) == (0,) * (2 * n_dim):  # Check if padding is all zeros
        return x

    return nn.functional.pad(x, pad=tuple(pad_tuple), value=value)


def reshape_at_dim(
    x: torch.Tensor,
    dim: int,
    target_shape: Union[tuple[int, ...], list[int]],  # Use ellipsis for tuple
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
        target_shape (Union[Tuple[int, ...], List[int]]): target shape for the specified dimension

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = x.dim()
    actual_dim = dim if dim >= 0 else n_dim + dim
    if not (0 <= actual_dim < n_dim):
        raise IndexError(f"Dimension out of range: {dim}")

    tgt = tuple(target_shape)
    # --- Start Fix: Handle merging previous dimension ---
    # Check if target is single element and previous dim exists and product matches
    if len(tgt) == 1 and actual_dim > 0:
        desired = tgt[0]
        combined = x.shape[actual_dim - 1] * x.shape[actual_dim]
        if combined == desired:
            shape_list = list(x.shape)
            shape_list[actual_dim - 1] = desired  # Replace previous dim size
            del shape_list[actual_dim]  # Delete current dim
            return x.reshape(*shape_list)
    # --- End Fix ---

    # fallback => standard single-dimension replace
    shape_list = list(x.shape)
    old_size = shape_list[actual_dim]
    new_prod = 1
    for v in tgt:
        new_prod *= v

    if new_prod != old_size:
        raise RuntimeError(
            f"reshape_at_dim: can't reshape dimension {actual_dim} of size {old_size} into {tgt} (product={new_prod})"
        )
    # Replace the single dimension size with the elements of target_shape
    shape_list = shape_list[:actual_dim] + list(tgt) + shape_list[actual_dim + 1 :]
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
    actual_dim = dim if dim >= 0 else n_dim + dim
    if not (0 <= actual_dim < n_dim):
        raise IndexError(
            f"Target dimension {dim} out of range for tensor with {n_dim} dimensions."
        )

    if actual_dim == n_dim - 1:  # Already the last dimension
        return x

    # Create the permutation order
    dims = list(range(n_dim))
    final_dim_index = dims.pop(-1)  # Remove the last dimension index
    dims.insert(actual_dim, final_dim_index)  # Insert it at the target position

    return x.permute(dims)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict: dict = {}  # Add type hint

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
        # Ensure all arrays in the list have compatible shapes before concatenating
        if not v:
            continue  # Skip empty lists
        first_shape = v[0].shape
        if not all(item.shape == first_shape for item in v):
            # Attempt to reshape if possible (e.g., scalar vs 1-element array)
            reshaped_v = []
            for item in v:
                if item.shape == () and first_shape == (1,):
                    reshaped_v.append(item.reshape(1))
                elif item.shape == (1,) and first_shape == ():
                    reshaped_v.append(item.reshape(()))
                elif item.shape == first_shape:
                    reshaped_v.append(item)
                else:
                    raise ValueError(
                        f"Incompatible shapes for key '{k}': {first_shape} vs {item.shape}"
                    )
            v = reshaped_v

        try:
            merged_dict[k] = np.concatenate(v)
        except ValueError as e:
            print(f"Error concatenating key '{k}': {e}")
            # Optionally handle error differently, e.g., keep as list
            # merged_dict[k] = v # Keep as list if concat fails
            raise e  # Re-raise for now

    return merged_dict

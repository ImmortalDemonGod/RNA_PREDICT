"""
Core forward pass logic for the AtomAttentionEncoder.
"""
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)
from .config import ProcessInputsParams
from .feature_processing import extract_atom_features, adapt_tensor_dimensions
from .pair_embedding import create_pair_embedding


def _process_simple_embedding(
    encoder: Any, input_feature_dict: InputFeatureDict
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
    """
    Process input features without coordinates.

    Args:
        encoder: The encoder module instance
        input_feature_dict: Dictionary of input features

    Returns:
        Tuple containing:
            - Token-level embedding
            - Atom-level embedding (q_l)
            - Atom-level embedding (c_l)
            - None (no pair embedding)
    """
    # Extract features
    c_l = extract_atom_features(encoder, input_feature_dict)

    # Process through a simple projection
    q_l = c_l

    # Project to token dimension
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))

    # Get token count from restype
    restype = safe_tensor_access(input_feature_dict, "restype")
    # --- Start Fix ---
    # Correctly determine token dimension based on restype ndim
    if restype is not None and hasattr(restype, 'dim'): # Check if restype is a tensor
        if restype.dim() >= 3: # e.g., [B, N_token, C]
            num_tokens = restype.shape[-2]
        elif restype.dim() == 2: # e.g., [B, N_token] or [N_token, C]
            # Heuristic: Assume the larger dimension is the token dimension if ambiguous
            # Or assume [B, N_token] format is most likely
            if restype.shape[0] > restype.shape[1] and restype.shape[1] != a_atom.shape[-1]: # Likely [N_token, C]
                 num_tokens = restype.shape[0]
            else: # Assume [B, N_token]
                 num_tokens = restype.shape[-1]
        elif restype.dim() == 1: # e.g., [N_token]
            num_tokens = restype.shape[0]
        else: # Fallback if restype is scalar or invalid
            warnings.warn(f"Could not determine num_tokens from restype shape {restype.shape}. Falling back to a_atom.")
            num_tokens = a_atom.shape[-2] # Fallback to atom dimension of a_atom
    else: # Fallback if restype is None or not a tensor
         warnings.warn(f"restype is None or not a Tensor. Falling back to a_atom shape for num_tokens.")
         num_tokens = a_atom.shape[-2]
    # --- End Fix ---


    # Get atom to token mapping
    atom_to_token_idx = safe_tensor_access(input_feature_dict, "atom_to_token_idx")

    # Ensure atom_to_token_idx doesn't exceed num_tokens and is not None
    if atom_to_token_idx is not None:
        if atom_to_token_idx.numel() > 0 and atom_to_token_idx.max() >= num_tokens:
            warnings.warn(
                f"[AtomAttentionEncoder] atom_to_token_idx max value {atom_to_token_idx.max()} >= num_tokens {num_tokens}. "
                f"Clipping indices to prevent out-of-bounds error."
            )
            atom_to_token_idx = torch.clamp(atom_to_token_idx, max=num_tokens - 1)
    else:
        # Handle case where atom_to_token_idx might be missing
        warnings.warn("atom_to_token_idx is None. Cannot perform aggregation.")
        # Depending on desired behavior, either raise error or return default
        # For now, let's allow _aggregate_to_token_level to handle it (might error there)
        pass


    # Aggregate atom features to token level
    # Ensure atom_to_token_idx is not None before passing
    if atom_to_token_idx is None:
         # Create a default index if it's missing, mapping all atoms to token 0
         warnings.warn("Creating default atom_to_token_idx mapping all atoms to token 0.")
         atom_to_token_idx = torch.zeros(a_atom.shape[:-1], dtype=torch.long, device=a_atom.device)
         if num_tokens == 0: num_tokens = 1 # Avoid n_token=0 if using default index

    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)

    return a, q_l, c_l, None


def _process_coordinate_encoding(
    encoder: Any, q_l: torch.Tensor, r_l: Optional[torch.Tensor], ref_pos: torch.Tensor
) -> torch.Tensor:
    """
    Process and add coordinate-based positional encoding.

    Args:
        encoder: The encoder module instance
        q_l: Input atom features
        r_l: Atom coordinates, shape [..., N_atom, 3]
        ref_pos: Reference atom positions

    Returns:
        Updated atom features with positional encoding
    """
    if r_l is None:
        return q_l

    # Check coordinates shape matches expected
    if r_l.ndim >= 2 and r_l.size(-1) == 3 and r_l.size(-2) == ref_pos.size(-2):
        return q_l + encoder.linear_no_bias_r(r_l)
    else:
        # Log shape mismatch and skip linear transformation
        warnings.warn(
            f"Warning: r_l shape mismatch. Expected [..., {ref_pos.size(-2)}, 3], "
            f"got {r_l.shape}. Skipping linear_no_bias_r."
        )
        return q_l


def _process_style_embedding(
    encoder: Any,
    c_l: torch.Tensor,
    s: Optional[torch.Tensor],
    atom_to_token_idx: Optional[torch.Tensor], # Allow None
) -> torch.Tensor:
    """
    Process style embedding from token to atom level.

    Args:
        encoder: The encoder module instance
        c_l: Input atom features
        s: Token-level style embedding
        atom_to_token_idx: Mapping from atoms to tokens (can be None)

    Returns:
        Updated atom features with style information
    """
    if s is None or atom_to_token_idx is None: # Check if index is None
        return c_l

    # Broadcast token-level s to atom-level
    broadcasted_s = broadcast_token_to_atom(s, atom_to_token_idx)

    # Ensure compatible shape for layernorm
    if broadcasted_s.size(-1) != encoder.c_s:
        broadcasted_s = adapt_tensor_dimensions(broadcasted_s, encoder.c_s)

    # Apply layer norm and add to atom embedding
    return c_l + encoder.linear_no_bias_s(encoder.layernorm_s(broadcasted_s))


def _aggregate_to_token_level(
    encoder: Any, a_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """
    Aggregate atom-level features to token-level.

    Args:
        encoder: The encoder module instance (not directly used here, but kept for consistency)
        a_atom: Atom-level features [..., N_atom, C]
        atom_to_token_idx: Mapping from atoms to tokens [..., N_atom]
        num_tokens: Number of tokens

    Returns:
        Token-level aggregated features [..., N_token, C]
    """
    # Ensure atom_to_token_idx has compatible batch dimensions with a_atom
    # Target shape for index: [*a_atom.shape[:-2], N_atom]

    original_idx_shape = atom_to_token_idx.shape
    target_batch_shape = a_atom.shape[:-2]  # e.g., [B] or [B, N_sample]
    n_batch_dims_target = len(target_batch_shape)
    n_atom_dim_a = a_atom.shape[-2]  # N_atom dimension from a_atom

    # Ensure index has at least batch_dims + 1 dimensions (batch + atom)
    temp_idx = atom_to_token_idx
    while temp_idx.dim() < n_batch_dims_target + 1:
        temp_idx = temp_idx.unsqueeze(0)

    # Match the number of batch dimensions by squeezing/unsqueezing index
    while temp_idx.dim() > n_batch_dims_target + 1:
        squeezed = False
        # Try squeezing dimensions between the first batch dim and the last (atom) dim
        for i in range(1, temp_idx.dim() - 1):
            if temp_idx.shape[i] == 1:
                temp_idx = temp_idx.squeeze(i)
                squeezed = True
                break
        if not squeezed:
            # If no squeezable dim found, check if the first dim can be squeezed (if it's not the only batch dim)
            if temp_idx.shape[0] == 1 and n_batch_dims_target > 0 and temp_idx.dim() > 2:
                 temp_idx = temp_idx.squeeze(0)
                 squeezed = True

        if not squeezed:
             warnings.warn(
                 f"Could not reduce index dimensions from {temp_idx.dim()} to target {n_batch_dims_target + 1}. "
                 f"Index shape: {temp_idx.shape}, Original: {original_idx_shape}, Target Batch: {target_batch_shape}"
             )
             break # Cannot reduce further

    # Add missing batch dimensions (typically at the start or after first batch dim)
    while temp_idx.dim() < n_batch_dims_target + 1:
        # Add dimensions after the potential first batch dim
        temp_idx = temp_idx.unsqueeze(1 if n_batch_dims_target > 0 else 0)

    # --- Start Fix: Check atom dimension compatibility BEFORE batch expansion ---
    n_atom_dim_idx = temp_idx.shape[-1]
    if n_atom_dim_idx != n_atom_dim_a:
        # Attempt to expand atom dimension only if it's 1
        if n_atom_dim_idx == 1:
            warnings.warn(
                f"Atom dimension mismatch: a_atom has {n_atom_dim_a} atoms, index has 1. "
                f"Expanding index atom dimension. Original idx shape: {original_idx_shape}."
            )
            temp_idx = temp_idx.expand(*temp_idx.shape[:-1], n_atom_dim_a)
            n_atom_dim_idx = temp_idx.shape[-1] # Update after expansion
        else:
            # If atom dimensions mismatch and index atom dim is not 1, it's an irreconcilable error.
            raise ValueError(
                f"Irreconcilable atom dimension mismatch in _aggregate_to_token_level. "
                f"a_atom shape: {a_atom.shape} (N_atom={n_atom_dim_a}), "
                f"processed atom_to_token_idx shape: {temp_idx.shape} (N_atom={n_atom_dim_idx}). "
                f"Original index shape was {original_idx_shape}. Cannot aggregate."
            )
    # --- End Fix ---


    # Match batch dimension sizes via expand (only if atom dimensions now match)
    current_batch_shape = temp_idx.shape[:-1]
    if target_batch_shape != current_batch_shape:
        # Special case: if target_batch_shape is empty, we need to squeeze all batch dimensions
        if len(target_batch_shape) == 0:
            # Keep squeezing until we only have the atom dimension
            while temp_idx.dim() > 1:
                temp_idx = temp_idx.squeeze(0)
        else:
            # Check if expansion is possible (target dim == current dim OR current dim == 1)
            can_expand = all(
                t == c or c == 1 for t, c in zip(target_batch_shape, current_batch_shape)
            )
            # Also need to ensure the number of dimensions is compatible for expansion
            # (temp_idx might have fewer batch dims than target after squeezing)
            if len(current_batch_shape) <= len(target_batch_shape) and can_expand:
                 target_idx_shape = target_batch_shape + (n_atom_dim_a,) # Use the matched atom dim size
                 try:
                     # Expand to the target shape (handles adding dimensions implicitly if needed)
                     temp_idx = temp_idx.expand(target_idx_shape)
                 except RuntimeError as e:
                     # This expansion should ideally not fail if the checks above passed, but catch just in case
                     raise RuntimeError(
                         f"Failed to expand atom_to_token_idx batch dimensions from {current_batch_shape} "
                         f"to {target_batch_shape} (target index shape: {target_idx_shape}). "
                         f"Original idx shape: {original_idx_shape}. Error: {e}"
                     ) from e
            else: # Expansion is not possible
                 # Refined error message for clarity
                 raise ValueError(
                    f"Cannot expand index batch dimensions {current_batch_shape} to target {target_batch_shape}. "
                    f"Expansion possible: {can_expand}, Dim lengths compatible: {len(current_batch_shape) <= len(target_batch_shape)}. "
                    f"Original idx shape: {original_idx_shape}. Aggregation impossible."
                )
    # If shapes already match, do nothing and proceed. The removed 'else' block caused the error.

    # Final check before aggregation (should always pass if logic above is correct)
    if temp_idx.shape[:-1] != a_atom.shape[:-2] or temp_idx.shape[-1] != n_atom_dim_a:
         # This path indicates a bug in the reshaping/expansion logic itself.
         raise AssertionError(
            f"Internal logic error: Shape mismatch persists before aggregation despite checks. "
            f"a_atom shape {a_atom.shape}, atom_to_token_idx shape {temp_idx.shape}. "
            f"Original idx shape was {original_idx_shape}."
        )

    # Ensure atom_to_token_idx doesn't exceed num_tokens to prevent out-of-bounds
    if num_tokens <= 0:
         raise ValueError(f"num_tokens must be positive, but got {num_tokens}.")
    if temp_idx.numel() > 0:
        max_idx = temp_idx.max()
        if max_idx >= num_tokens:
            warnings.warn(
                f"[AtomAttentionEncoder] atom_to_token_idx contains indices ({max_idx}) >= num_tokens ({num_tokens}). "
                f"Clipping indices to prevent out-of-bounds error."
            )
            temp_idx = torch.clamp(temp_idx, max=num_tokens - 1)

    # Aggregate atom features to token level
    return aggregate_atom_to_token(
        x_atom=a_atom,
        atom_to_token_idx=temp_idx,
        n_token=num_tokens,
        reduce="mean",
    )


def process_inputs_with_coords(
    encoder: Any,
    params: ProcessInputsParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process inputs when coordinates are available.

    Args:
        encoder: The encoder module instance
        params: Parameters for processing inputs with coordinates

    Returns:
        Tuple containing:
            - Token-level embedding
            - Atom-level embedding (q_l)
            - Atom-level embedding (c_l)
            - Pair embedding (p_lm)
    """
    # Create pair embedding for atom transformer
    p_lm = create_pair_embedding(encoder, params.input_feature_dict)

    # Get required tensors
    atom_to_token_idx = safe_tensor_access(
        params.input_feature_dict, "atom_to_token_idx"
    )
    ref_pos = safe_tensor_access(params.input_feature_dict, "ref_pos")
    restype = safe_tensor_access(params.input_feature_dict, "restype")

    # Process coordinates and style embedding
    q_l = _process_coordinate_encoding(encoder, params.c_l, params.r_l, ref_pos)
    c_l = _process_style_embedding(encoder, params.c_l, params.s, atom_to_token_idx)

    # Process through atom transformer
    # Unsqueeze p_lm to add a block dimension if it's 4D (output from create_pair_embedding)
    # AtomTransformer expects 3D (global) or 5D (local, with block dim) pair embedding
    if p_lm is not None and p_lm.dim() == 4:
        # Add block dimension: [B, N_queries, N_keys, C] -> [B, 1, N_queries, N_keys, C]
        # This assumes create_pair_embedding returns [B, n_queries, n_keys, C_pair]
        p_for_transformer = p_lm.unsqueeze(1)
    elif p_lm is not None and p_lm.dim() in [3, 5]:
        # Pass 3D or 5D tensors directly
        p_for_transformer = p_lm
    elif p_lm is None:
        warnings.warn("p_lm is None in process_inputs_with_coords. AtomTransformer might fail.")
        p_for_transformer = None # Explicitly set to None
    else:
        # Raise error for unexpected dimensions like 1D, 2D, or > 5D
        raise ValueError(f"Unexpected p_lm dimensions ({p_lm.dim()}) received in process_inputs_with_coords. Shape: {p_lm.shape}")

    # Pass the original token-level 's' for conditioning, not the atom-level 'c_l'
    # Also ensure s is not None before passing
    s_for_transformer = params.s
    if s_for_transformer is None:
        # Create a default s if it's None, matching batch dims of q_l and token dim of q_l
        # This is a fallback, ideally s should always be provided when has_coords=True
        warnings.warn(
            "Token-level style embedding 's' is None in process_inputs_with_coords. Creating a zero tensor."
        )
        batch_dims = q_l.shape[:-2]
        # Get num_tokens from restype if available
        if restype is not None:
            num_tokens = restype.shape[1]  # [B, N_tokens, ...]
        else:
            # Fallback to atom_to_token_idx if restype not available
            num_tokens = int(atom_to_token_idx.max().item()) + 1 if atom_to_token_idx is not None else q_l.shape[-2]

        s_for_transformer = torch.zeros(
            *batch_dims, num_tokens, encoder.c_s, device=q_l.device, dtype=q_l.dtype
        )

    q_l = encoder.atom_transformer(
        q=q_l, s=s_for_transformer, p=p_for_transformer, chunk_size=params.chunk_size # Use aligned p
    )

    # Project to token dimension with ReLU
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))

    # Get number of tokens from restype
    if restype is not None:
        num_tokens = restype.shape[1]  # [B, N_tokens, ...]
    else:
        # Fallback to atom_to_token_idx if restype not available
        num_tokens = int(atom_to_token_idx.max().item()) + 1 if atom_to_token_idx is not None else q_l.shape[-2]

    # Aggregate to token level
    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)

    return a, q_l, c_l, p_lm
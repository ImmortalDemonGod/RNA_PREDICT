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
        a_atom: Atom-level features
        atom_to_token_idx: Mapping from atoms to tokens
        num_tokens: Number of tokens

    Returns:
        Token-level aggregated features
    """
    # Ensure atom_to_token_idx has compatible batch dimensions with a_atom
    # Target shape for index: [*a_atom.shape[:-2], N_atom]

    original_idx_shape = atom_to_token_idx.shape
    target_batch_shape = a_atom.shape[:-2]  # e.g., [B] or [B, N_sample]
    n_batch_dims_target = len(target_batch_shape)
    n_atom_dim_a = a_atom.shape[-2]  # N_atom dimension from a_atom

    # Ensure index has at least batch_dims + 1 dimensions
    temp_idx = atom_to_token_idx
    while temp_idx.dim() < n_batch_dims_target + 1:
        temp_idx = temp_idx.unsqueeze(0)

    # Match the number of batch dimensions by squeezing/unsqueezing index
    while temp_idx.dim() > n_batch_dims_target + 1:
        squeezed = False
        for i in range(1, temp_idx.dim() - 1): # Don't squeeze batch or atom dim
            if temp_idx.shape[i] == 1:
                temp_idx = temp_idx.squeeze(i)
                squeezed = True
                break
        if not squeezed:
            warnings.warn(
                f"Could not reduce index dimensions from {temp_idx.dim()} to {n_batch_dims_target + 1}"
            )
            break

    while temp_idx.dim() < n_batch_dims_target + 1:
        temp_idx = temp_idx.unsqueeze(1) # Add dimensions after potential batch dim 0

    # Match batch dimension sizes via expand
    current_batch_shape = temp_idx.shape[:-1]
    if target_batch_shape != current_batch_shape:
        can_expand = all(
            t == c or c == 1 for t, c in zip(target_batch_shape, current_batch_shape)
        )
        if can_expand:
            # Ensure the atom dimension matches a_atom's atom dimension if index's doesn't
            n_atom_dim_idx = temp_idx.shape[-1]
            if n_atom_dim_idx != n_atom_dim_a:
                 warnings.warn(
                    f"Atom dimension mismatch between a_atom ({n_atom_dim_a}) and "
                    f"atom_to_token_idx ({n_atom_dim_idx}) before expansion. "
                    f"Original idx shape: {original_idx_shape}. Attempting to expand index atom dim."
                 )
                 # Try expanding the atom dimension if it's 1
                 if n_atom_dim_idx == 1:
                     temp_idx = temp_idx.expand(*current_batch_shape, n_atom_dim_a)
                 else:
                     # If not 1, we likely can't safely expand. Keep original shape for error below.
                     pass # Keep temp_idx as is, let the final check catch it

            # Now expand batch dimensions
            target_idx_shape = target_batch_shape + (temp_idx.shape[-1],) # Use potentially expanded atom dim
            try:
                temp_idx = temp_idx.expand(target_idx_shape)
            except RuntimeError as e:
                warnings.warn(
                    f"Could not expand atom_to_token_idx from {temp_idx.shape} to {target_idx_shape}: {e}. "
                    f"Original idx shape: {original_idx_shape}."
                )
        else:
            warnings.warn(
                f"Cannot expand index batch dims {current_batch_shape} to target {target_batch_shape}. "
                f"Original idx shape: {original_idx_shape}."
            )

    # Final check before aggregation
    if temp_idx.shape[:-1] != a_atom.shape[:-2] or temp_idx.shape[-1] != n_atom_dim_a:
        warnings.warn(
            f"Shape mismatch persists before aggregation: "
            f"a_atom shape {a_atom.shape}, "
            f"atom_to_token_idx shape {temp_idx.shape}. "
            f"Original idx shape was {original_idx_shape}."
        )
        # Fallback: Use original index if shapes are incompatible after attempts
        # This might still error in aggregate_atom_to_token, but prevents using a badly reshaped index
        if atom_to_token_idx.shape[:-1] != a_atom.shape[:-2] or atom_to_token_idx.shape[-1] != n_atom_dim_a:
             warnings.warn("Falling back to original atom_to_token_idx due to persistent shape mismatch.")
             temp_idx = atom_to_token_idx # Revert to original if still mismatched

    # Ensure atom_to_token_idx doesn't exceed num_tokens to prevent out-of-bounds
    if temp_idx.numel() > 0 and temp_idx.max() >= num_tokens:
        warnings.warn(
            f"[AtomAttentionEncoder] atom_to_token_idx contains indices >= {num_tokens}. "
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

    # Process coordinates and style embedding
    q_l = _process_coordinate_encoding(encoder, params.c_l, params.r_l, ref_pos)
    c_l = _process_style_embedding(encoder, params.c_l, params.s, atom_to_token_idx)

    # Process through atom transformer
    # Unsqueeze p_lm to add a block dimension (assuming 1 block for this context)
    # AtomTransformer expects 3D (global) or 5D (local) pair embedding
    if p_lm.dim() == 4:
        p_lm_5d = p_lm.unsqueeze(
            1
        )  # Add block dimension: [B, N_queries, N_keys, C] -> [B, 1, N_queries, N_keys, C]
    else:
        # If it's already 3D or 5D, pass it as is (though create_pair_embedding seems to always make 4D)
        p_lm_5d = p_lm

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
        # Estimate num_tokens based on atom_to_token_idx if possible, else use q_l atom dim
        if atom_to_token_idx is not None and atom_to_token_idx.numel() > 0:
            num_tokens_est = int(atom_to_token_idx.max().item()) + 1
        else:
            num_tokens_est = q_l.shape[-2]  # Fallback: use atom dimension

        s_for_transformer = torch.zeros(
            *batch_dims, num_tokens_est, encoder.c_s, device=q_l.device, dtype=q_l.dtype
        )

    q_l = encoder.atom_transformer(
        q=q_l, s=s_for_transformer, p=p_lm_5d, chunk_size=params.chunk_size
    )

    # Project to token dimension with ReLU
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))

    # Get token count and aggregate to token level
    restype = safe_tensor_access(params.input_feature_dict, "restype")
    # Corrected logic:
    if restype is not None and hasattr(restype, 'dim'):
        if restype.dim() >= 3:
            num_tokens = restype.shape[-2]
        elif restype.dim() == 2:
            if restype.shape[0] > restype.shape[1] and restype.shape[1] != a_atom.shape[-1]:
                num_tokens = restype.shape[0]
            else:
                num_tokens = restype.shape[-1]
        elif restype.dim() == 1:
            num_tokens = restype.shape[0]
        else:
            warnings.warn(f"Could not determine num_tokens from restype shape {restype.shape}. Falling back to a_atom.")
            num_tokens = a_atom.shape[-2]
    else:
         warnings.warn(f"restype is None or not a Tensor. Falling back to a_atom shape for num_tokens.")
         num_tokens = a_atom.shape[-2]


    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)

    return a, q_l, c_l, p_lm
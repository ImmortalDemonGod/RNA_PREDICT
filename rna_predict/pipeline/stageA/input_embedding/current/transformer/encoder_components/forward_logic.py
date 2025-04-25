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
from .feature_processing import adapt_tensor_dimensions, extract_atom_features
from .pair_embedding import create_pair_embedding

import snoop

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
    if restype is not None and hasattr(restype, "dim"):  # Check if restype is a tensor
        if restype.dim() >= 3:  # e.g., [B, N_token, C]
            num_tokens = restype.shape[-2]
        elif restype.dim() == 2:  # e.g., [B, N_token] or [N_token, C]
            # Heuristic: Assume the larger dimension is the token dimension if ambiguous
            # Or assume [B, N_token] format is most likely
            if (
                restype.shape[0] > restype.shape[1]
                and restype.shape[1] != a_atom.shape[-1]
            ):  # Likely [N_token, C]
                num_tokens = restype.shape[0]
            else:  # Assume [B, N_token]
                num_tokens = restype.shape[-1]
        elif restype.dim() == 1:  # e.g., [N_token]
            num_tokens = restype.shape[0]
        else:  # Fallback if restype is scalar or invalid
            warnings.warn(
                f"Could not determine num_tokens from restype shape {restype.shape}. Falling back to a_atom."
            )
            num_tokens = a_atom.shape[-2]  # Fallback to atom dimension of a_atom
    else:  # Fallback if restype is None or not a tensor
        warnings.warn(
            "restype is None or not a Tensor. Falling back to a_atom shape for num_tokens."
        )
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
        warnings.warn(
            "Creating default atom_to_token_idx mapping all atoms to token 0."
        )
        atom_to_token_idx = torch.zeros(
            a_atom.shape[:-1], dtype=torch.long, device=a_atom.device
        )
        if num_tokens == 0:
            num_tokens = 1  # Avoid n_token=0 if using default index

    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)

    return a, q_l, c_l, None


def _process_coordinate_encoding(
    encoder: Any, q_l: torch.Tensor, r_l: Optional[torch.Tensor], ref_pos: Optional[torch.Tensor]
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

    # Handle None ref_pos
    if ref_pos is None:
        warnings.warn("ref_pos is None in _process_coordinate_encoding. Using r_l directly.")
        return q_l + encoder.linear_no_bias_r(r_l)

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

##@snoop
def _process_style_embedding(
    encoder: Any,
    c_l: torch.Tensor,
    s: Optional[torch.Tensor],
    atom_to_token_idx: Optional[torch.Tensor],
) -> torch.Tensor:
    print(f"[DEBUG][CALL] _process_style_embedding c_l.shape={getattr(c_l, 'shape', None)} s.shape={getattr(s, 'shape', None)} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    if s is None or atom_to_token_idx is None:  # Check if index is None
        return c_l

    # Broadcast token-level s to atom-level
    broadcasted_s = broadcast_token_to_atom(s, atom_to_token_idx)

    # Ensure compatible shape for layernorm
    if broadcasted_s.size(-1) != encoder.c_s:
        broadcasted_s = adapt_tensor_dimensions(broadcasted_s, encoder.c_s)

    # Apply layer norm and add to atom embedding
    x = encoder.linear_no_bias_s(encoder.layernorm_s(broadcasted_s))
    print(f"[DEBUG][PRE-ADD] c_l.shape={c_l.shape}, x.shape={x.shape}, broadcasted_s.shape={broadcasted_s.shape}")
    print(f"[DEBUG][ENCODER][_process_style_embedding] c_l.shape={getattr(c_l, 'shape', None)}, x.shape={getattr(x, 'shape', None)}")
    print(f"[DEBUG][ENCODER][_process_style_embedding] c_l type={type(c_l)}, x type={type(x)}")
    print(f"[DEBUG][ENCODER][_process_style_embedding] atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    # If c_l is a tensor, print a snippet of its values for provenance
    if hasattr(c_l, 'shape') and hasattr(c_l, 'flatten'):
        print(f"[DEBUG][ENCODER][_process_style_embedding] c_l.flatten()[:10]={c_l.flatten()[:10]}")
    # PATCH: Ensure c_l is atom-level before addition
    # If c_l is [batch, num_residues, emb_dim], broadcast to [batch, num_atoms, emb_dim]
    if c_l.shape[1] != x.shape[1] and atom_to_token_idx is not None:
        print(f"[PATCH][ENCODER][_process_style_embedding] Broadcasting c_l from residues to atoms using atom_to_token_idx")
        # Handle case where c_l has an extra dimension (sample dimension)
        # c_l shape: [batch, sample, num_atoms, emb_dim], atom_to_token_idx shape: [batch, num_atoms]
        if c_l.ndim == 4 and atom_to_token_idx.ndim == 2:
            # Create a 3D index tensor for each sample in the batch
            # First, add a sample dimension to atom_to_token_idx
            expanded_idx = atom_to_token_idx.unsqueeze(1)  # [batch, 1, num_atoms]
            # Then expand to match c_l's sample dimension
            expanded_idx = expanded_idx.expand(-1, c_l.shape[1], -1)  # [batch, sample, num_atoms]
        elif c_l.ndim == 4 and atom_to_token_idx.ndim == 3 and c_l.shape[1] < atom_to_token_idx.shape[1]:
            # Handle case where c_l has fewer samples than atom_to_token_idx
            # Expand c_l to match the sample dimension of atom_to_token_idx
            c_l = c_l.expand(-1, atom_to_token_idx.shape[1], -1, -1)
            expanded_idx = atom_to_token_idx

            # Clamp indices to be within valid range (0 to c_l.shape[2]-1)
            # This prevents out-of-bounds errors when using torch.gather
            max_idx = c_l.shape[2] - 1
            if expanded_idx.max() > max_idx:
                print(f"[DEBUG][_process_style_embedding] Clamping indices from max {expanded_idx.max().item()} to {max_idx}")
                expanded_idx = torch.clamp(expanded_idx, max=max_idx)

            # Finally, add the embedding dimension
            # Handle case where atom_to_token_idx has an extra dimension (e.g., [1, 1, 11, 1])
            if expanded_idx.dim() == 3:
                # Standard case: expanded_idx is [batch, sample, num_atoms]
                idx = expanded_idx.unsqueeze(-1).expand(-1, -1, -1, c_l.shape[-1])  # [batch, sample, num_atoms, emb_dim]
                # Now gather using the 4D index tensor
                c_l = torch.gather(c_l, 2, idx)
            elif expanded_idx.dim() == 4:
                # Special case: expanded_idx is already [batch, sample, num_atoms, 1]
                # Just expand the last dimension to match feature dimension
                idx = expanded_idx.expand(-1, -1, -1, c_l.shape[-1])  # [batch, sample, num_atoms, emb_dim]
                # Now gather using the 4D index tensor
                c_l = torch.gather(c_l, 2, idx)
            else:
                # Unexpected case, print debug info and try to handle gracefully
                print(f"[DEBUG][UNEXPECTED] expanded_idx.dim()={expanded_idx.dim()}, shape={expanded_idx.shape}")
                # Reshape to expected dimensions if possible
                if expanded_idx.dim() > 4:
                    # Too many dimensions, try to squeeze
                    expanded_idx = expanded_idx.squeeze()
                    if expanded_idx.dim() == 3:
                        idx = expanded_idx.unsqueeze(-1).expand(-1, -1, -1, c_l.shape[-1])
                        c_l = torch.gather(c_l, 2, idx)
                    elif expanded_idx.dim() == 4:
                        idx = expanded_idx.expand(-1, -1, -1, c_l.shape[-1])
                        c_l = torch.gather(c_l, 2, idx)
                    else:
                        raise ValueError(f"Cannot handle expanded_idx with shape {expanded_idx.shape} after squeezing")
                else:
                    # Too few dimensions, try to unsqueeze
                    while expanded_idx.dim() < 3:
                        expanded_idx = expanded_idx.unsqueeze(0)
                    idx = expanded_idx.unsqueeze(-1).expand(-1, -1, -1, c_l.shape[-1])
                    c_l = torch.gather(c_l, 2, idx)
        else:
            # Original case: c_l shape: [batch, num_atoms, emb_dim], atom_to_token_idx shape: [batch, num_atoms]
            # Clamp indices to be within valid range (0 to c_l.shape[1]-1)
            max_idx = c_l.shape[1] - 1
            clamped_idx = atom_to_token_idx.clone()
            if clamped_idx.max() > max_idx:
                print(f"[DEBUG][_process_style_embedding] Clamping indices from max {clamped_idx.max().item()} to {max_idx}")
                clamped_idx = torch.clamp(clamped_idx, max=max_idx)

            # Handle case where atom_to_token_idx has an extra dimension (e.g., [1, 1, 11])
            if clamped_idx.dim() == 3:
                # Special case: clamped_idx is [batch, sample, num_atoms]
                # First, we need to handle the case where c_l is [batch, num_atoms, emb_dim]
                # We need to add a sample dimension to c_l
                if c_l.dim() == 3:
                    c_l = c_l.unsqueeze(1)  # [batch, 1, num_atoms, emb_dim]
                # Handle case where c_l has fewer samples than clamped_idx
                if c_l.dim() == 4 and c_l.shape[1] < clamped_idx.shape[1]:
                    # Expand c_l to match the sample dimension of clamped_idx
                    c_l = c_l.expand(-1, clamped_idx.shape[1], -1, -1)
                # Now gather using the 4D index tensor
                idx = clamped_idx.unsqueeze(-1).expand(-1, -1, -1, c_l.shape[-1])  # [batch, sample, num_atoms, emb_dim]
                c_l = torch.gather(c_l, 2, idx)
            elif clamped_idx.dim() == 4:
                # Special case: clamped_idx is already [batch, sample, num_atoms, 1]
                # Just expand the last dimension to match feature dimension
                if c_l.dim() == 3:
                    c_l = c_l.unsqueeze(1)  # [batch, 1, num_atoms, emb_dim]
                # Handle case where c_l has fewer samples than clamped_idx
                if c_l.dim() == 4 and c_l.shape[1] < clamped_idx.shape[1]:
                    # Expand c_l to match the sample dimension of clamped_idx
                    c_l = c_l.expand(-1, clamped_idx.shape[1], -1, -1)
                idx = clamped_idx.expand(-1, -1, -1, c_l.shape[-1])  # [batch, sample, num_atoms, emb_dim]
                c_l = torch.gather(c_l, 2, idx)
            else:
                # Standard case: clamped_idx is [batch, num_atoms]
                idx = clamped_idx.unsqueeze(-1).expand(-1, -1, c_l.shape[-1])
                c_l = torch.gather(c_l, 1, idx)
        print(f"[PATCH][ENCODER][_process_style_embedding] New c_l.shape={c_l.shape}, x.shape={x.shape}")
    try:
        return c_l + x
    except Exception as e:
        print(f"[DEBUG][EXCEPTION-ADD] {e}")
        print(f"[DEBUG][EXCEPTION-ADD-SHAPE] c_l.shape={c_l.shape}, x.shape={x.shape}")
        # Try to reshape c_l to match x
        if c_l.dim() != x.dim():
            print(f"[DEBUG][EXCEPTION-ADD-DIM] c_l.dim()={c_l.dim()}, x.dim()={x.dim()}")
            if c_l.dim() < x.dim():
                # Add dimensions to c_l
                for _ in range(x.dim() - c_l.dim()):
                    c_l = c_l.unsqueeze(1)
            elif c_l.dim() > x.dim():
                # Add dimensions to x
                for _ in range(c_l.dim() - x.dim()):
                    x = x.unsqueeze(1)
            print(f"[DEBUG][EXCEPTION-ADD-SHAPE-AFTER] c_l.shape={c_l.shape}, x.shape={x.shape}")
        # Try broadcasting
        try:
            return c_l + x
        except Exception as e2:
            print(f"[DEBUG][EXCEPTION-ADD-AFTER] {e2}")
            # Try reshaping x to match c_l
            try:
                # If c_l has more dimensions, reshape x to match
                if c_l.dim() > x.dim():
                    # Reshape x to match c_l's dimensions
                    new_shape = list(x.shape)
                    while len(new_shape) < c_l.dim():
                        new_shape.insert(1, 1)  # Insert singleton dimension after batch
                    x = x.view(*new_shape)
                    print(f"[DEBUG][EXCEPTION-ADD-RESHAPE-X] New x.shape={x.shape}")
                    return c_l + x
                # If x has more dimensions, reshape c_l to match
                elif x.dim() > c_l.dim():
                    # Reshape c_l to match x's dimensions
                    new_shape = list(c_l.shape)
                    while len(new_shape) < x.dim():
                        new_shape.insert(1, 1)  # Insert singleton dimension after batch
                    c_l = c_l.view(*new_shape)
                    print(f"[DEBUG][EXCEPTION-ADD-RESHAPE-CL] New c_l.shape={c_l.shape}")
                    return c_l + x
                # Last resort: reshape c_l to exactly match x
                if c_l.shape != x.shape:
                    try:
                        # Try to expand c_l to match x's shape
                        c_l = c_l.expand_as(x)
                        return c_l + x
                    except Exception as e3:
                        print(f"[DEBUG][EXCEPTION-ADD-EXPAND] {e3}")
                        # Try to expand x to match c_l's shape
                        try:
                            x = x.expand_as(c_l)
                            return c_l + x
                        except Exception as e4:
                            print(f"[DEBUG][EXCEPTION-ADD-EXPAND-X] {e4}")
                            # Give up and return x
                            return x
            except Exception as e5:
                print(f"[DEBUG][EXCEPTION-ADD-FINAL] {e5}")
                # Give up and return x
                return x


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
    # --- Start Dimension Alignment v3 ---
    target_batch_shape = a_atom.shape[:-2]  # e.g., (B, S) or (B,)
    n_atom_dim_a = a_atom.shape[-2]
    original_idx_shape = atom_to_token_idx.shape
    temp_idx = atom_to_token_idx

    # 1. Ensure atom dimension matches (must match exactly)
    n_atom_dim_idx = temp_idx.shape[-1]
    if n_atom_dim_idx != n_atom_dim_a:
        if n_atom_dim_idx == 1 and temp_idx.ndim == len(target_batch_shape) + 1:
            warnings.warn(
                f"Atom dimension mismatch: a_atom has {n_atom_dim_a} atoms, index has 1. "
                f"Expanding index atom dimension. Original idx shape: {original_idx_shape}."
            )
            temp_idx = temp_idx.expand(*temp_idx.shape[:-1], n_atom_dim_a)
        else:
            raise ValueError(
                f"Irreconcilable atom dimension mismatch in _aggregate_to_token_level. "
                f"a_atom shape: {a_atom.shape} (N_atom={n_atom_dim_a}), "
                f"atom_to_token_idx shape: {temp_idx.shape} (N_atom={n_atom_dim_idx}). "
                f"Original index shape was {original_idx_shape}. Cannot aggregate."
            )

    # 2. Add Sample dimension (dim 1) if a_atom has it but temp_idx doesn't
    # a_atom shape: [B, S, N_atom, C], temp_idx shape: [B, N_atom] -> [B, 1, N_atom]
    if len(target_batch_shape) == 2 and temp_idx.ndim == 2:
        temp_idx = temp_idx.unsqueeze(1)
    # Handle cases where a_atom might be [B, N_atom, C] but idx is [1, B, N_atom] - less likely
    elif len(target_batch_shape) == 1 and temp_idx.ndim == 3 and temp_idx.shape[0] == 1:
         temp_idx = temp_idx.squeeze(0) # Remove leading singleton dim

    # 3. Expand remaining batch dimensions to match target
    target_idx_shape = target_batch_shape + (n_atom_dim_a,) # Full target shape including atom dim
    if temp_idx.shape != target_idx_shape:
        # Special case: allow singleton sample dimension to expand to multi-sample
        if (
            temp_idx.ndim == 3 and
            len(target_batch_shape) == 2 and
            temp_idx.shape[1] == 1 and
            target_batch_shape[1] > 1
        ):
            temp_idx = temp_idx.expand(target_idx_shape)
        # Special case for diffusion module with N_sample dimension
        # If a_atom is [B, N_sample, N_atom, C] and temp_idx is [B, N_atom],
        # we need to reshape temp_idx to [B, 1, N_atom] and then expand to [B, N_sample, N_atom]
        elif len(target_batch_shape) >= 2 and temp_idx.ndim == 2:
            # First add the missing dimension
            temp_idx = temp_idx.unsqueeze(1)
            # Then try to expand to match target shape
            try:
                temp_idx = temp_idx.expand(target_idx_shape)
            except RuntimeError as e:
                # If expansion fails, try a different approach
                print(f"[DEBUG][_aggregate_to_token_level] Expansion failed: {e}. Trying alternative approach.")
                # Create a new tensor with the right shape
                new_temp_idx = torch.zeros(target_idx_shape, dtype=temp_idx.dtype, device=temp_idx.device)
                # Fill it with the values from temp_idx
                for i in range(target_batch_shape[1]):
                    new_temp_idx[:, i] = temp_idx[:, 0]
                temp_idx = new_temp_idx
        else:
            try:
                temp_idx = temp_idx.expand(target_idx_shape)
            except RuntimeError as e:
                # Try a more careful approach for diffusion module
                if len(target_batch_shape) >= 2 and temp_idx.ndim >= 2:
                    print(f"[DEBUG][_aggregate_to_token_level] Expansion failed: {e}. Trying diffusion-specific approach.")
                    # Create a new tensor with the right shape
                    new_temp_idx = torch.zeros(target_idx_shape, dtype=temp_idx.dtype, device=temp_idx.device)
                    # Fill it with the values from temp_idx, broadcasting as needed
                    if temp_idx.ndim == 2:  # [B, N_atom]
                        for i in range(target_batch_shape[1]):  # For each sample
                            new_temp_idx[:, i] = temp_idx
                    elif temp_idx.ndim == 3 and temp_idx.shape[1] != target_batch_shape[1]:  # [B, S, N_atom] with S != target S
                        for i in range(target_batch_shape[1]):  # For each target sample
                            new_temp_idx[:, i] = temp_idx[:, 0]  # Use the first sample
                    temp_idx = new_temp_idx
                else:
                    raise ValueError(
                        f"Cannot expand index batch dimensions {temp_idx.shape[:-1]} to target {target_batch_shape}. "
                        f"Original idx shape: {original_idx_shape}. Aggregation impossible. Error: {e}"
                    ) from e
    # --- End Dimension Alignment v3 ---

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

    print("[DEBUG][_aggregate_to_token_level] a_atom.shape:", getattr(a_atom, 'shape', None))
    print("[DEBUG][_aggregate_to_token_level] atom_to_token_idx.shape:", getattr(temp_idx, 'shape', None))
    print("[DEBUG][_aggregate_to_token_level] n_token:", num_tokens)
    print("[DEBUG][_aggregate_to_token_level] a_atom dtype:", getattr(a_atom, 'dtype', None))
    print("[DEBUG][_aggregate_to_token_level] atom_to_token_idx dtype:", getattr(temp_idx, 'dtype', None))
    print("[DEBUG][_aggregate_to_token_level] a_atom (first 5):", a_atom.flatten()[:5])
    print("[DEBUG][_aggregate_to_token_level] atom_to_token_idx (first 10):", temp_idx.flatten()[:10])

    # Aggregate atom features to token level
    return aggregate_atom_to_token(
        x_atom=a_atom,
        atom_to_token_idx=temp_idx,
        n_token=num_tokens,
        reduce="mean",
    )


#@snoop
def _process_inputs_with_coords_impl(
    encoder: Any,
    params: ProcessInputsParams,
    atom_to_token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process inputs when coordinates are available.

    Args:
        encoder: The encoder module instance
        params: Parameters for processing inputs with coordinates
        atom_to_token_idx: Optional atom to token index

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

    # Create a default restype tensor if not found
    default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = atom_to_token_idx.shape[0] if atom_to_token_idx is not None else 1
    num_tokens = atom_to_token_idx.shape[1] if atom_to_token_idx is not None and atom_to_token_idx.dim() > 1 else 50
    default_restype = torch.zeros((batch_size, num_tokens), device=default_device, dtype=torch.long)
    restype = safe_tensor_access(params.input_feature_dict, "restype", default=default_restype)

    print(f"[DEBUG][PRE-CALL] c_l.shape={params.c_l.shape if params.c_l is not None else None} s.shape={params.s.shape if params.s is not None else None} atom_to_token_idx.shape={atom_to_token_idx.shape if atom_to_token_idx is not None else None}")

    # Process coordinates and style embedding
    q_l = _process_coordinate_encoding(encoder, params.c_l, params.r_l, ref_pos)
    print(f"[DEBUG][PRE-CALL-TYPE] c_l={type(params.c_l)} s={type(params.s)} atom_to_token_idx={type(atom_to_token_idx)}")
    try:
        print(f"[DEBUG][PRE-CALL-SHAPE] c_l.shape={getattr(params.c_l, 'shape', None)} s.shape={getattr(params.s, 'shape', None)} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
    except Exception as e:
        print(f"[DEBUG][PRE-CALL-SHAPE-ERROR] {e}")
    try:
        c_l = _process_style_embedding(encoder, params.c_l, params.s, atom_to_token_idx)
    except Exception as e:
        print(f"[DEBUG][EXCEPTION] {e}")
        print(f"[DEBUG][EXCEPTION-TYPE] c_l={type(params.c_l)} s={type(params.s)} atom_to_token_idx={type(atom_to_token_idx)}")
        print(f"[DEBUG][EXCEPTION-SHAPE] c_l.shape={getattr(params.c_l, 'shape', None)} s.shape={getattr(params.s, 'shape', None)} atom_to_token_idx.shape={getattr(atom_to_token_idx, 'shape', None)}")
        raise

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
        warnings.warn(
            "p_lm is None in process_inputs_with_coords. AtomTransformer might fail."
        )
        p_for_transformer = None  # Explicitly set to None
    else:
        # Raise error for unexpected dimensions like 1D, 2D, or > 5D
        raise ValueError(
            f"Unexpected p_lm dimensions ({p_lm.dim()}) received in process_inputs_with_coords. Shape: {p_lm.shape}"
        )

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
            num_tokens = (
                int(atom_to_token_idx.max().item()) + 1
                if atom_to_token_idx is not None
                else q_l.shape[-2]
            )

        s_for_transformer = torch.zeros(
            *batch_dims, num_tokens, encoder.c_s, device=q_l.device, dtype=q_l.dtype
        )

    q_l = encoder.atom_transformer(
        q=q_l,
        s=s_for_transformer,
        p=p_for_transformer,
        chunk_size=params.chunk_size,  # Use aligned p
    )

    # Project to token dimension with ReLU
    a_atom = F.relu(encoder.linear_no_bias_q(q_l))

    # Get number of tokens from restype
    if restype is not None:
        num_tokens = restype.shape[1]  # [B, N_tokens, ...]
    else:
        # Fallback to atom_to_token_idx if restype not available
        num_tokens = (
            int(atom_to_token_idx.max().item()) + 1
            if atom_to_token_idx is not None
            else q_l.shape[-2]
        )

    # Aggregate to token level
    # --- DEBUG PRINT ---
    print(f"[DEBUG AGG] a_atom shape: {a_atom.shape}")
    print(f"[DEBUG AGG] atom_to_token_idx shape: {atom_to_token_idx.shape}")
    print(f"[DEBUG AGG] num_tokens: {num_tokens}")
    # --- END DEBUG PRINT ---
    a = _aggregate_to_token_level(encoder, a_atom, atom_to_token_idx, num_tokens)

    return a, q_l, c_l, p_lm


#@snoop
def process_inputs_with_coords(
    encoder: Any,
    params: ProcessInputsParams,
    atom_to_token_idx: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _process_inputs_with_coords_impl(encoder, params, atom_to_token_idx)

"""
Atom pair transformation module for neural network operations.

This module contains functions for transforming tensors specifically for
atom pair operations in molecular representations.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple, TypeVar, Union

import torch

T = TypeVar("T", bound="AtomPairConfig")


@dataclass
class AtomPairConfig:
    """Configuration for atom pair operations.

    This class encapsulates configuration parameters for atom pair operations,
    making function signatures cleaner and more maintainable.
    """

    n_batch: int
    n_tokens: int
    n_atoms_per_token: int
    n_tokens_out: Optional[int] = None
    device: Optional[torch.device] = None
    atom_to_token_idx: Optional[torch.Tensor] = None

    # Maximum recommended number of parameters
    MAX_PARAMS: ClassVar[int] = 4

    @classmethod
    def from_tensors(
        cls: type[T],
        token_feats: torch.Tensor,
        atom_config: dict,
    ) -> T:
        """Create config from tensor dimensions.

        Args:
            token_feats: Token features tensor
            atom_config: Dictionary containing:
                - atom_to_token_idx: Mapping from atom to token indices
                - n_atoms_per_token: Number of atoms per token
                - n_tokens_out: Optional, number of output tokens
                - device: Optional, device for computation

        Returns:
            AtomPairConfig: Configuration object
        """
        return cls(
            n_batch=token_feats.shape[0],
            n_tokens=token_feats.shape[1],
            n_atoms_per_token=atom_config["n_atoms_per_token"],
            n_tokens_out=atom_config.get("n_tokens_out"),
            device=atom_config.get("device") or token_feats.device,
            atom_to_token_idx=atom_config["atom_to_token_idx"],
        )


def _validate_token_feats_shape(token_feats: torch.Tensor, expected_dim: int) -> None:
    """
    Validate the shape of token features tensor.

    Args:
        token_feats: Token features tensor
        expected_dim: Expected dimension

    Raises:
        ValueError: If the tensor does not have the expected dimension
    """
    if token_feats.ndim != expected_dim:
        raise ValueError(
            f"token_feats should be {expected_dim}D "
            f"(batch, n_tokens, channels), got shape {token_feats.shape}"
        )


def _map_tokens_to_atoms(
    token_feats: torch.Tensor, config: AtomPairConfig
) -> torch.Tensor:
    """
    Map token features to atoms using atom-to-token mapping.

    Args:
        token_feats: Token features tensor
        config: Configuration for the operation

    Returns:
        Atom features derived from token features

    Raises:
        ValueError: If atom_to_token_idx is not provided in config
    """
    if config.atom_to_token_idx is None:
        raise ValueError("atom_to_token_idx must be provided in config")

    # Reshape token features for broadcasting
    token_feats_reshaped = token_feats.reshape(
        config.n_batch, config.n_tokens, 1, token_feats.shape[-1]
    )

    # Create gather indices for mapping tokens to atoms
    n_atoms = config.atom_to_token_idx.shape[1]

    # Add extra dimension to match token_feats_reshaped shape
    gather_indices = (
        config.atom_to_token_idx.unsqueeze(-1)
        .unsqueeze(-1)
        .expand(-1, -1, 1, token_feats.shape[-1])
    )

    # Map tokens to atoms via gather operation
    atom_feats = token_feats_reshaped.gather(1, gather_indices)

    return atom_feats.squeeze(2)


def broadcast_token_to_local_atom_pair(
    token_feats: torch.Tensor,
    atom_config: Union[AtomPairConfig, dict],
) -> torch.Tensor:
    """Broadcast token features to local atom pairs.

    This function supports two calling conventions:
    1. With a pre-configured AtomPairConfig object
    2. With a dictionary containing all required parameters

    Args:
        token_feats: Token features of shape (batch, n_tokens, channels)
        atom_config: Either:
            - AtomPairConfig object containing all configuration
            - Dictionary with keys:
                - atom_to_token_idx: Tensor mapping atoms to tokens
                - n_atoms_per_token: Number of atoms per token
                - n_tokens_out: (Optional) Number of output tokens
                - device: (Optional) Device for computation

    Returns:
        Broadcasted token features of shape (batch, n_atoms, n_atoms, 2*channels)

    Raises:
        ValueError: If required parameters are missing
    """
    # Validate token features shape
    _validate_token_feats_shape(token_feats, 3)

    # Handle both calling conventions
    if isinstance(atom_config, AtomPairConfig):
        pair_config = atom_config
    else:
        # For dictionary case
        if "atom_to_token_idx" not in atom_config:
            raise ValueError(
                "atom_to_token_idx must be provided in atom_config dictionary"
            )
        if "n_atoms_per_token" not in atom_config:
            raise ValueError(
                "n_atoms_per_token must be provided in atom_config dictionary"
            )

        pair_config = AtomPairConfig.from_tensors(
            token_feats=token_feats,
            atom_config=atom_config,
        )

    # Map tokens to atoms
    atom_feats = _map_tokens_to_atoms(token_feats, pair_config)

    # Get number of atoms from atom_feats shape
    n_atoms = atom_feats.size(1)

    # Expand for pairwise interactions: expand dimensions to enable broadcasting to shape (batch, n_atoms, n_atoms, channels)
    atom_feats_i = atom_feats.unsqueeze(2).expand(
        -1, n_atoms, n_atoms, -1
    )  # (batch, n_atoms, n_atoms, channels)
    atom_feats_j = atom_feats.unsqueeze(1).expand(
        -1, n_atoms, n_atoms, -1
    )  # (batch, n_atoms, n_atoms, channels)

    # Combine pairwise features along the last dimension to form (batch, n_atoms, n_atoms, 2*channels)
    pair_feats = torch.cat([atom_feats_i, atom_feats_j], dim=-1)

    return pair_feats


def gather_pair_embedding_in_dense_trunk(
    x: torch.Tensor, indices: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """
    Gather pair-wise embeddings in dense trunks.

    Args:
        x: Input tensor of shape (batch, n_atoms, feat_dim)
        indices: Tuple of (idx_q, idx_k) containing query and key indices

    Returns:
        Gathered pair embeddings of shape (batch, *idx_shape, feat_dim*2)

    Raises:
        ValueError: If input tensor shape is invalid or indices don't match
    """
    # Validate inputs
    if x.ndim != 3:
        raise ValueError(
            f"Expected 3D tensor, got {x.ndim}D tensor with shape {x.shape}"
        )

    idx_q, idx_k = indices
    if idx_q.shape != idx_k.shape:
        raise ValueError(
            f"idx_q and idx_k must have the same shape. Got {idx_q.shape} and {idx_k.shape}"
        )

    # Get dimensions
    batch_size, n_atoms, feat_dim = x.shape

    # Create batch indices
    batch_indices = (
        torch.arange(batch_size, device=x.device).unsqueeze(-1).expand_as(idx_q)
    )

    # Reshape indices for gather operation
    flat_batch_q = batch_indices.reshape(-1)
    flat_idx_q = idx_q.reshape(-1)
    flat_batch_k = batch_indices.reshape(-1)
    flat_idx_k = idx_k.reshape(-1)

    # Gather embeddings
    gathered_q = x[flat_batch_q, flat_idx_q].reshape(*idx_q.shape, feat_dim)
    gathered_k = x[flat_batch_k, flat_idx_k].reshape(*idx_k.shape, feat_dim)

    # Combine embeddings
    combined = torch.cat([gathered_q, gathered_k], dim=-1)

    return combined

"""
Embedding operations for tensor utilities.

This module provides functions for handling tensor embeddings and conversions
between residue-level and atom-level representations.
"""

import torch
from typing import List, Optional, Union

from .types import ResidueAtomMap, logger
from .validation import get_dimensions, validate_atom_indices


def handle_empty_inputs(
    residue_atom_map: ResidueAtomMap,
    s_emb: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Handle empty inputs for residue_to_atoms function.

    Args:
        residue_atom_map: Mapping from residue indices to atom indices
        s_emb: Residue embeddings tensor

    Returns:
        Empty tensor with appropriate shape if inputs are empty, None otherwise

    Raises:
        ValueError: If residue_atom_map (ResidueAtomMap) is empty but s_emb has residues
    """
    if len(residue_atom_map) == 0:
        if s_emb.shape[-2] != 0:
            raise ValueError(f"Empty residue_atom_map provided, but s_emb has {s_emb.shape[-2]} residues.")
        # Return empty tensor with correct shape
        if s_emb.dim() == 2:  # [N_residue, C_s]
            return torch.empty((0, s_emb.shape[-1]), dtype=s_emb.dtype, device=s_emb.device)
        else:  # [B, N_residue, C_s]
            return torch.empty((s_emb.shape[0], 0, s_emb.shape[-1]), dtype=s_emb.dtype, device=s_emb.device)
    return None


class EmbeddingDimensions:
    """Dimensions for embeddings."""
    def __init__(
        self,
        n_atom: int,
        c_s: int,
    ) -> None:
        self.n_atom = n_atom
        self.c_s = c_s


class EmbeddingConfig:
    """Configuration for initializing embeddings."""
    def __init__(
        self,
        dimensions: EmbeddingDimensions,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        self.n_atom = dimensions.n_atom
        self.c_s = dimensions.c_s
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


def initialize_atom_embeddings(
    config: EmbeddingConfig,
) -> torch.Tensor:
    """
    Initialize atom embeddings tensor with zeros.

    Args:
        config: Configuration for embedding initialization

    Returns:
        Initialized atom embeddings tensor
    """
    if config.batch_size is not None:
        return torch.zeros(
            (config.batch_size, config.n_atom, config.c_s),
            dtype=config.dtype,
            device=config.device
        )
    return torch.zeros(
        (config.n_atom, config.c_s),
        dtype=config.dtype,
        device=config.device
    )


class EmbeddingAssignment:
    """Data for a single embedding assignment operation."""
    def __init__(
        self,
        res_idx: int,
        atom_indices: List[int],
    ) -> None:
        self.res_idx = res_idx
        self.atom_indices = atom_indices


class AssignEmbeddingsConfig:
    """Configuration for assigning embeddings."""
    def __init__(
        self,
        atom_embs: torch.Tensor,
        s_emb: torch.Tensor,
        assignment: EmbeddingAssignment,
        is_batched: bool,
    ) -> None:
        self.atom_embs = atom_embs
        self.s_emb = s_emb
        self.res_idx = assignment.res_idx
        self.atom_indices = assignment.atom_indices
        self.is_batched = is_batched


def assign_embeddings(
    config: AssignEmbeddingsConfig,
) -> None:
    """
    Assign residue embeddings to corresponding atoms.

    Args:
        config: Configuration for assigning embeddings
    """
    if config.is_batched:
        # For each batch, assign the residue embedding to all its atoms
        config.atom_embs[:, config.atom_indices, :] = config.s_emb[:, config.res_idx, :].unsqueeze(1)
    else:
        # Assign the residue embedding to all its atoms
        config.atom_embs[config.atom_indices, :] = config.s_emb[config.res_idx, :]


class ResidueToAtomsConfig:
    """Configuration for residue-to-atoms conversion."""
    def __init__(
        self,
        s_emb: torch.Tensor,
        residue_atom_map: ResidueAtomMap,
    ) -> None:
        self.s_emb = s_emb
        self.residue_atom_map = residue_atom_map

        # Derived properties
        _, self.n_atom, self.c_s, self.is_batched = get_dimensions(s_emb, residue_atom_map)
        self.batch_size = s_emb.shape[0] if self.is_batched else None
        self.dtype = s_emb.dtype
        self.device = s_emb.device


def create_empty_atom_tensor(
    config: ResidueToAtomsConfig,
) -> torch.Tensor:
    """
    Create an empty atom tensor with appropriate shape.

    Args:
        config: Configuration for residue-to-atoms conversion

    Returns:
        Empty tensor with appropriate shape
    """
    if config.is_batched:
        return torch.empty(
            (config.batch_size, 0, config.c_s),
            dtype=config.dtype,
            device=config.device
        )
    return torch.empty(
        (0, config.c_s),
        dtype=config.dtype,
        device=config.device
    )


# EmbeddingDimensions class is already defined above


class CreateEmbeddingsConfig:
    """Configuration for creating atom embeddings."""
    def __init__(
        self,
        s_emb: torch.Tensor,
        residue_atom_map: ResidueAtomMap,
        dimensions: EmbeddingDimensions,
        is_batched: bool,
    ) -> None:
        self.s_emb = s_emb
        self.residue_atom_map = residue_atom_map
        self.n_atom = dimensions.n_atom
        self.c_s = dimensions.c_s
        self.is_batched = is_batched
        self.batch_size = s_emb.shape[0] if is_batched else None
        self.dtype = s_emb.dtype
        self.device = s_emb.device


def create_atom_embeddings(
    config: CreateEmbeddingsConfig,
) -> torch.Tensor:
    """
    Create atom embeddings from residue embeddings.

    Args:
        config: Configuration for creating atom embeddings

    Returns:
        Atom-level embeddings tensor
    """
    # Create embedding configuration
    dimensions = EmbeddingDimensions(config.n_atom, config.c_s)
    embedding_config = EmbeddingConfig(
        dimensions=dimensions,
        batch_size=config.batch_size,
        dtype=config.dtype,
        device=config.device
    )

    # Initialize atom embeddings tensor
    atom_embs = initialize_atom_embeddings(embedding_config)

    # Assign residue embeddings to corresponding atoms
    for res_idx, atom_indices in enumerate(config.residue_atom_map):
        if not atom_indices:
            continue  # Skip residues with no atoms

        # Create embedding assignment
        assignment = EmbeddingAssignment(
            res_idx=res_idx,
            atom_indices=atom_indices,
        )

        # Create assignment configuration
        assign_config = AssignEmbeddingsConfig(
            atom_embs=atom_embs,
            s_emb=config.s_emb,
            assignment=assignment,
            is_batched=config.is_batched
        )
        assign_embeddings(assign_config)

    logger.debug(f"Expanded residue embeddings {config.s_emb.shape} to atom embeddings {atom_embs.shape}")
    return atom_embs


def residue_to_atoms(
    s_emb: torch.Tensor,
    residue_atom_map: ResidueAtomMap,
) -> torch.Tensor:
    """
    Expands residue-level embeddings to atom-level embeddings using a precomputed map.

    Assigns the embedding of residue `i` to all atoms belonging to residue `i` as defined
    by the `residue_atom_map`.

    Args:
        s_emb (torch.Tensor): Residue embeddings. Shape [N_residue, C_s] or [B, N_residue, C_s].
        residue_atom_map (ResidueAtomMap): A list where index `i` contains the list of
                                           global atom indices corresponding to residue `i`.
                                           The length of this list must equal N_residue.
                                           The union of all inner lists must cover all atoms
                                           from 0 to N_atom-1 exactly once and without overlaps.

    Returns:
        torch.Tensor: Atom-level embeddings. Shape [N_atom, C_s] or [B, N_atom, C_s].

    Raises:
        ValueError: If input shapes are inconsistent (e.g., `len(residue_atom_map)` != `s_emb.shape[-2]`)
                    or `residue_atom_map` is invalid (e.g., gaps or overlaps in atom indices).
    """
    # Handle empty inputs
    empty_result = handle_empty_inputs(residue_atom_map, s_emb)
    if empty_result is not None:
        return empty_result

    # Create configuration
    config = ResidueToAtomsConfig(s_emb, residue_atom_map)

    # Handle case with no atom indices
    if config.n_atom == 0:
        logger.warning("residue_atom_map contains no atom indices. Returning empty tensor.")
        return create_empty_atom_tensor(config)

    # Validate atom indices
    validate_atom_indices(config.residue_atom_map, config.n_atom)

    # Create atom embeddings
    dimensions = EmbeddingDimensions(config.n_atom, config.c_s)
    create_config = CreateEmbeddingsConfig(
        s_emb=config.s_emb,
        residue_atom_map=config.residue_atom_map,
        dimensions=dimensions,
        is_batched=config.is_batched
    )
    return create_atom_embeddings(create_config)

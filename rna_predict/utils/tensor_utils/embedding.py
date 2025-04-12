"""
Embedding operations for tensor utilities.

This module provides functions for handling tensor embeddings and conversions
between residue-level and atom-level representations.
"""

import torch
from typing import List, Optional, Tuple, Union

from .types import ResidueAtomMap, logger
from .validation import get_dimensions, validate_atom_indices


def handle_empty_inputs(
    residue_atom_map: ResidueAtomMap,
    s_emb: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    Return an empty atom embeddings tensor if the residue mapping is empty.
    
    If residue_atom_map is empty but the residue embeddings tensor contains residues,
    a ValueError is raised. For a 2D residue embeddings tensor (shape [N_residue, C_s]),
    an empty tensor with shape [0, C_s] is returned. For batched embeddings (shape [B, N_residue, C_s]),
    an empty tensor with shape [B, 0, C_s] is returned.
    
    Args:
        residue_atom_map: Mapping from residue indices to atom indices.
        s_emb: Tensor of residue embeddings.
    
    Returns:
        A tensor with the appropriate empty shape if residue_atom_map is empty and s_emb is empty;
        otherwise, None.
    
    Raises:
        ValueError: If residue_atom_map is empty but s_emb contains residues.
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
        """
        Initialize an EmbeddingDimensions instance.
        
        Args:
            n_atom: Number of atoms.
            c_s: Size (dimension) of the embedding.
        """
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
        """Initialize an EmbeddingConfig instance.
        
        Extracts the number of atoms and embedding size from the provided dimensions and
        sets optional configuration parameters for batch size, tensor data type, and
        computation device.
        
        Args:
            dimensions: An EmbeddingDimensions object specifying the number of atoms (n_atom)
                        and embedding size (c_s).
            batch_size: Optional; the batch size for processing embeddings.
            dtype: Optional; the torch data type for the embedding tensor.
            device: Optional; the target device for tensor computations.
        """
        self.n_atom = dimensions.n_atom
        self.c_s = dimensions.c_s
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device


def initialize_atom_embeddings(
    config: EmbeddingConfig,
) -> torch.Tensor:
    """
    Initializes an atom embeddings tensor with zeros.
    
    Creates a tensor filled with zeros based on the provided embedding configuration. When a batch size
    is specified, the tensor shape is (batch_size, n_atom, c_s); otherwise, it is (n_atom, c_s). The tensor
    is created with the data type and device defined in the configuration.
    
    Args:
        config (EmbeddingConfig): Configuration for initializing atom embeddings, including the number
            of atoms (n_atom), embedding dimension (c_s), optional batch size, data type, and device.
    
    Returns:
        torch.Tensor: A zero-initialized tensor for atom embeddings.
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
        """
        Initializes an embedding assignment.
        
        Stores the residue index and its corresponding atom indices for mapping a residue's
        embedding to atom-level positions.
        
        Args:
            res_idx (int): The index of the residue.
            atom_indices (List[int]): The list of atom indices associated with the residue.
        """
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
        """
        Initializes an embedding assignment configuration.
        
        Stores atom- and residue-level embeddings along with the residue index and corresponding atom indices extracted from 
        the provided assignment. The batched flag indicates whether the embeddings are processed in a batched manner.
        
        Args:
            atom_embs: Tensor containing atom-level embeddings.
            s_emb: Tensor containing residue-level embeddings.
            assignment: Object providing the residue index and corresponding atom indices.
            is_batched: Boolean flag indicating if the embeddings are batched.
        """
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
    
    Transfers a residue embedding to specified atom indices using the provided configuration.
    For batched inputs, the residue embedding is unsqueezed and assigned across all batches;
    for non-batched inputs, a direct assignment is performed.
    
    Args:
        config: An assignment configuration object containing:
            atom_embs: Tensor for atom embeddings to be updated.
            s_emb: Tensor containing residue embeddings.
            atom_indices: Indices in the atom embeddings where the residue embedding is assigned.
            res_idx: Index of the residue embedding to assign.
            is_batched: Flag indicating whether the embeddings are batched.
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
        """
        Initialize the embedding instance with residue embeddings and mapping.
        
        Assigns the residue embeddings and a mapping from residues to atom indices to the instance,
        and computes derived properties including the number of atoms, embedding dimension, batched flag,
        batch size (if applicable), data type, and device.
          
        Args:
            s_emb: A tensor containing residue-level embeddings.
            residue_atom_map: A mapping that assigns each residue to its corresponding atom indices.
        """
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
    Creates an empty tensor for atom embeddings based on the provided configuration.
    
    If the configuration indicates a batched operation, the returned tensor has shape (batch_size, 0, c_s); otherwise, it has shape (0, c_s). The tensor is created with the specified data type and device settings.
    
    Args:
        config (ResidueToAtomsConfig): Configuration for converting residue embeddings to atom embeddings, including batching, embedding size, and device details.
    
    Returns:
        torch.Tensor: An empty tensor with a shape determined by the configuration.
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
        """
        Initializes a residue-to-atom embedding configuration instance.
        
        Sets instance attributes based on the provided residue embeddings tensor, residue-to-atom 
        mapping, and embedding dimensions. If operating in batched mode, the batch size is derived 
        from the first dimension of the residue embeddings tensor. Also stores tensor metadata 
        (dtype and device) for subsequent embedding operations.
        
        Parameters:
            s_emb: A torch.Tensor containing residue embeddings.
            residue_atom_map: A mapping that associates residues with their corresponding atom indices.
            dimensions: An instance providing the number of atoms (n_atom) and the embedding size (c_s).
            is_batched: A boolean indicating whether the embeddings are organized in batches.
        """
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
    
    This function initializes a zero-filled atom embeddings tensor based on the
    provided configuration and then assigns residue embeddings to corresponding
    atoms using a residue-to-atom mapping. Residues without assigned atoms are
    skipped.
    
    Args:
        config: CreateEmbeddingsConfig containing:
            - n_atom (int): Total number of atoms.
            - c_s (int): Dimensionality of each embedding.
            - batch_size (int): Number of embeddings in a batch.
            - dtype: Data type of the tensor.
            - device: Device for tensor allocation.
            - s_emb (torch.Tensor): Residue-level embeddings.
            - residue_atom_map: Mapping from residue indices to lists of atom indices.
            - is_batched (bool): Indicates if embeddings are batched.
    
    Returns:
        torch.Tensor: Atom-level embeddings tensor with residue embeddings assigned.
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

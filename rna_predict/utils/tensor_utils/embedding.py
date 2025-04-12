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
    Checks and handles empty residue mapping for atom embedding assignment.
    
    If the residue-to-atom mapping is empty, this function verifies that the residue 
    embedding tensor contains no residues. If residues are present despite an empty mapping, 
    a ValueError is raised. When no residues are present, it returns an empty tensor with 
    the same dtype and device as s_emb; for a 2D tensor, the shape is (0, embedding_dim), 
    and for a 3D tensor, the shape is (batch_size, 0, embedding_dim). If the mapping is not 
    empty, the function returns None.
      
    Args:
        residue_atom_map: Mapping from residue indices to atom indices.
        s_emb: Tensor of residue embeddings, either of shape [N_residue, embedding_dim] or 
               [batch_size, N_residue, embedding_dim].
      
    Returns:
        An empty tensor with an appropriate shape if residue_atom_map is empty and s_emb 
        has no residues; otherwise, None.
      
    Raises:
        ValueError: If residue_atom_map is empty but s_emb contains one or more residues.
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
        Initializes the embedding dimensions.
        
        Sets the number of atoms and the dimensionality of the embedding vector.
        
        Args:
            n_atom (int): Number of atoms in the embedding.
            c_s (int): Size of the embedding vector.
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
        """
        Initializes an embedding configuration.
        
        Configures an instance with the specified embedding dimensions and optional
        parameters for batch size, data type, and device.
            
        Args:
            dimensions: An EmbeddingDimensions instance providing the number of atoms
                and the embedding vector size.
            batch_size: Optional batch size for subsequent embedding operations.
            dtype: Optional torch data type for the embedding tensors.
            device: Optional device specification (e.g., 'cpu' or 'cuda') for computations.
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
    Initializes a zero-filled tensor for atom embeddings.
    
    Creates a tensor with shape (batch_size, n_atom, c_s) if a batch size is provided;
    otherwise, produces a tensor with shape (n_atom, c_s). The tensor uses the data type
    and device specified in the configuration.
    
    Args:
        config: An EmbeddingConfig instance specifying embedding dimensions, optional
                batch size, data type, and device.
    
    Returns:
        A PyTorch tensor of initialized atom embeddings.
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
        Initialize an embedding assignment with a residue index and corresponding atom indices.
        
        Args:
            res_idx: The index of the residue for the assignment.
            atom_indices: A list of indices for atoms that receive the residue embedding.
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
        Initialize assignment configuration for mapping residue embeddings to atom embeddings.
        
        This constructor sets the atom and residue embeddings tensors and extracts the residue index and
        atom indices from the provided assignment. It also indicates whether the operation is batched.
        
        Parameters:
            atom_embs (torch.Tensor): Tensor containing atom-level embeddings.
            s_emb (torch.Tensor): Tensor containing residue-level embeddings.
            assignment (EmbeddingAssignment): Object holding residue and corresponding atom indices.
            is_batched (bool): Flag indicating if the inputs are organized in batches.
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
    Assign residue embeddings to corresponding atom positions.
    
    This function updates an atom embeddings tensor by transferring residue-level embeddings
    to all specified atom indices. When processing batched data (i.e. when config.is_batched
    is True), it expands each residue embedding along a new axis before assigning it per batch.
    For non-batched inputs, the residue embedding is directly applied to the designated atom positions.
    
    Args:
        config: Assignment configuration containing:
            atom_embs (torch.Tensor): Tensor of atom embeddings to be updated.
            s_emb (torch.Tensor): Tensor of residue embeddings.
            res_idx: Index used to select the residue embedding.
            atom_indices: Indices of atom embeddings to assign the residue embedding.
            is_batched (bool): Flag indicating whether the assignment involves batched data.
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
        Initialize configuration for converting residue embeddings to atom embeddings.
        
        Args:
            s_emb: A tensor of residue embeddings.
            residue_atom_map: A mapping that assigns residue indices to corresponding atom indices.
        
        This constructor stores the provided embeddings and mapping, and computes derived properties,
        including the number of atoms, the embedding dimension, and the batched status. For batched
        embeddings, it sets the batch size based on the first tensor dimension; otherwise, batch size is None.
        It also captures the tensor's data type and device.
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
    Create an empty atom tensor based on configuration.
    
    Returns a torch.Tensor with no atom embeddings. If the configuration specifies batched
    processing, the tensor shape is (config.batch_size, 0, config.c_s); otherwise, it is (0, config.c_s).
    The tensor is created with the data type and device defined in the configuration.
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
        Initializes configuration for converting residue to atom embeddings.
        
        Stores the residue embeddings, residue-to-atom mapping, and dimension parameters. If the data is batched,
        the batch size is extracted from the residue tensor, and the tensor's data type and device are recorded.
        
        Args:
            s_emb: A tensor containing residue-level embeddings.
            residue_atom_map: Mapping from residues to their corresponding atom indices.
            dimensions: An instance providing the number of atoms and embedding size.
            is_batched: Flag indicating whether the input tensor represents batched data.
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
    Generate atom-level embeddings by mapping residue embeddings to atoms.
    
    This function initializes an atom embeddings tensor using the provided
    embedding configuration (number of atoms, embedding dimensionality, batch size,
    data type, and device) and assigns residue embeddings to atom positions based on
    a residue-to-atom mapping. Residues with no corresponding atom indices are skipped,
    and the function supports both batched and non-batched inputs.
    
    Args:
        config (CreateEmbeddingsConfig): Configuration containing:
            - n_atom: Total number of atoms.
            - c_s: Dimensionality of each residue embedding.
            - batch_size: Number of samples in the batch.
            - dtype: Data type for the embeddings tensor.
            - device: Device on which to allocate the tensor.
            - residue_atom_map: Sequence mapping residue indices to lists of atom indices.
            - s_emb: Tensor with residue embeddings.
            - is_batched: Flag indicating whether the input embeddings are batched.
    
    Returns:
        torch.Tensor: The generated atom-level embeddings tensor.
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

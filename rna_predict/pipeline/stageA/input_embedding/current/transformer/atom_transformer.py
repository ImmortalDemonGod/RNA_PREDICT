"""
AtomTransformer module for RNA structure prediction.
"""

from typing import Optional, Tuple, cast

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.diffusion import (
    DiffusionTransformer,
)


class TransformerConfig:
    """
    Configuration class for AtomTransformer.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize transformer configuration.

        Args:
            c_atom: Embedding dimension for atom features
            c_s: Embedding dimension for token-level style/conditioning features
            c_atompair: Embedding dimension for atom pair features
            n_blocks: Number of blocks in the transformer
            n_heads: Number of attention heads
            n_queries: Local window size of query tensor for local attention
            n_keys: Local window size of key tensor for local attention
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency
        """
        # Set default values
        self.c_atom = 128
        self.c_s = 384
        self.c_atompair = 16
        self.n_blocks = 3
        self.n_heads = 4
        self.n_queries = 32
        self.n_keys = 128
        self.blocks_per_ckpt = None

        # Update with provided values
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


class AtomTransformer(torch.nn.Module):
    """
    Local transformer for atom embeddings with bias predicted from atom pair embeddings.
    Implements Algorithm 7 in AlphaFold3.
    """

    def __init__(self, config: Optional[TransformerConfig] = None, **kwargs) -> None:
        """
        Initialize the AtomTransformer.

        Args:
            config: Configuration object for the transformer
            **kwargs: Keyword arguments to create a config if not provided
        """
        super().__init__()

        # Create config if not provided
        if config is None:
            config = TransformerConfig(**kwargs)

        # Store configuration parameters
        self.n_blocks = config.n_blocks
        self.n_heads = config.n_heads
        self.n_queries = config.n_queries
        self.n_keys = config.n_keys
        self.c_atom = config.c_atom
        self.c_s = config.c_s
        self.c_atompair = config.c_atompair

        # Create the underlying diffusion transformer
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=config.n_blocks,
            n_heads=config.n_heads,
            c_a=config.c_atom,  # DiffusionTransformer expects atom dim for 'a' (input q)
            c_s=config.c_s,  # DiffusionTransformer expects style dim for 's' (input s)
            c_z=config.c_atompair,  # DiffusionTransformer expects pair dim for 'z' (input p)
            blocks_per_ckpt=config.blocks_per_ckpt,
        )

    def _check_tensor_type(self, p: torch.Tensor) -> None:
        """
        Check if the input is a tensor.

        Args:
            p: Input to validate

        Raises:
            ValueError: If p is not a tensor
        """
        if not isinstance(p, torch.Tensor):
            raise ValueError(f"Expected p to be a tensor, got {type(p)}")

    def _reshape_4d_tensor(self, p: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Reshape a 4D tensor to 3D by merging dimensions.

        Args:
            p: 4D tensor to reshape

        Returns:
            Tuple containing:
                - Reshaped tensor
                - New number of dimensions

        Raises:
            ValueError: If reshaping is not possible
        """
        # Try to reshape to 3D by merging dimensions
        if p.shape[1] * p.shape[2] == p.shape[1] * p.shape[2]:
            p_reshaped = p.reshape(p.shape[0], p.shape[1] * p.shape[2], p.shape[3])
            return p_reshaped, 3
        else:
            # If we can't reshape easily, raise error
            raise ValueError(
                f"AtomTransformer: 4D 'p' is not supported. Got shape={p.shape}."
            )

    def _check_dimensions(self, p: torch.Tensor, n_dims: int) -> None:
        """
        Check if the tensor has valid dimensions (3D or 5D).

        Args:
            p: Tensor to check
            n_dims: Number of dimensions

        Raises:
            ValueError: If dimensions are invalid
        """
        if n_dims not in [3, 5]:
            raise ValueError(
                f"AtomTransformer: 'p' must be 3D or 5D. Got shape={p.shape}, dim={n_dims}"
            )

    def _fix_last_dimension(self, p: torch.Tensor) -> torch.Tensor:
        """
        Fix the last dimension of a tensor to match c_atompair.

        Args:
            p: Tensor to fix

        Returns:
            Fixed tensor
        """
        if p.shape[-1] == 1:
            # Expand last dimension from 1 to c_atompair
            return p.expand(*p.shape[:-1], self.c_atompair)
        else:
            # Add dimension and expand
            return p.unsqueeze(-1).expand(*p.shape, self.c_atompair)

    def _validate_p_tensor(self, p: torch.Tensor) -> torch.Tensor:
        """
        Validate and fix pair embedding tensor dimensions.

        Args:
            p: Pair embedding tensor to validate

        Returns:
            Fixed tensor with correct dimensions

        Raises:
            ValueError: If p tensor has invalid dimensions that cannot be fixed
        """
        # Check tensor type
        self._check_tensor_type(p)

        n_dims = p.dim()

        # Handle 4D tensor by reshaping to 3D or 5D
        if n_dims == 4:
            p, n_dims = self._reshape_4d_tensor(p)

        # Check dimensions after potential reshaping
        self._check_dimensions(p, n_dims)

        # Fix last dimension if needed
        if p.shape[-1] != self.c_atompair:
            p = self._fix_last_dimension(p)

        return p

    def _add_batch_dimensions(self, q: torch.Tensor, s: torch.Tensor, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Add batch dimensions to tensors if needed.

        Args:
            q: Atom features tensor
            s: Style features tensor
            p: Pair features tensor

        Returns:
            Tuple containing:
                - Updated q tensor
                - Updated s tensor
                - Updated p tensor
                - Original q dimensions
        """
        # Store original dimensions
        q_dim = q.dim()

        # Add batch dimension if needed
        if q_dim == 2:
            q = q.unsqueeze(0)
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if p.dim() == 3:
            p = p.unsqueeze(0)

        return q, s, p, q_dim

    def _determine_attention_type(self, p: torch.Tensor, n_queries: Optional[int], n_keys: Optional[int], debug_logging: bool) -> Tuple[Optional[int], Optional[int], bool]:
        """
        Determine the type of attention to use based on tensor shapes and provided parameters.

        Args:
            p: Pair features tensor
            n_queries: Number of queries for local attention (if provided)
            n_keys: Number of keys for local attention (if provided)
            debug_logging: Whether to print debug messages

        Returns:
            Tuple containing:
                - Number of queries to use
                - Number of keys to use
                - Whether local attention should be used
        """
        current_p_dim = p.dim()

        # Check if the shape suggests local attention
        is_potentially_local = (
            (n_queries is not None and n_keys is not None) or
            (current_p_dim >= 4 and p.shape[-3] == self.n_queries and p.shape[-2] == self.n_keys)
        )

        # Debug print
        if debug_logging:
            print(f"DEBUG: p.shape={p.shape}, current_p_dim={current_p_dim}, n_queries={n_queries}, n_keys={n_keys}")

        # Use provided n_queries and n_keys if available, otherwise use class attributes or None
        n_q_to_use = n_queries or self.n_queries if is_potentially_local else None
        n_k_to_use = n_keys or self.n_keys if is_potentially_local else None

        # Debug print
        if debug_logging:
            print(f"DEBUG: Using n_queries={n_q_to_use}, n_keys={n_k_to_use}, is_potentially_local={is_potentially_local}")

        return n_q_to_use, n_k_to_use, is_potentially_local

    def forward(self, q: torch.Tensor, s: torch.Tensor, p: torch.Tensor, debug_logging: bool = False, **kwargs) -> torch.Tensor:
        """
        Process atom embeddings through the AtomTransformer, conditioned by token-level features.

        This method handles different tensor dimensions for the pair embedding 'p':
        - 5D tensor (trunk-based): p has shape [..., n_blocks, n_queries, n_keys, c_atompair]
        - 3D tensor (global attention): p has shape [..., N_atom, N_atom, c_atompair]

        Args:
            q: Atom embeddings of shape [..., N_atom, c_atom]
            s: Token-level style/conditioning embeddings of shape [..., N_token, c_s]
            p: Pair embeddings with varying dimensions based on use case
            debug_logging: Whether to print debug messages
            inplace_safe: Whether inplace operations are safe
            chunk_size: Size of chunks for memory optimization
            n_queries: Number of queries for local attention (optional)
            n_keys: Number of keys for local attention (optional)

        Returns:
            Updated atom embeddings of shape [..., N_atom, c_atom]

        Raises:
            ValueError: If p has invalid dimensions or wrong shape
        """
        # Validate and fix p tensor
        p = self._validate_p_tensor(p)

        # Add batch dimensions if needed
        q, s, p, q_dim = self._add_batch_dimensions(q, s, p)

        # Get optional parameters
        n_queries = kwargs.get('n_queries', None)
        n_keys = kwargs.get('n_keys', None)

        # Determine attention type
        n_q_to_use, n_k_to_use, _ = self._determine_attention_type(p, n_queries, n_keys, debug_logging)

        # Get optional parameters with defaults
        inplace_safe = kwargs.get('inplace_safe', False)
        chunk_size = kwargs.get('chunk_size', None)

        # Process through diffusion transformer
        result = self.diffusion_transformer(
            a=q,  # Pass atom features as 'a'
            s=s,  # Pass token-level style features as 's'
            z=p,  # Pass pair features as 'z'
            n_queries=n_q_to_use,  # Pass None for global, values for local
            n_keys=n_k_to_use,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        result = cast(torch.Tensor, result)

        # Remove batch dimension if it was added
        if q_dim == 2 and result.dim() > q_dim:
            result = result.squeeze(0)

        return result

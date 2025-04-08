"""
AtomTransformer module for RNA structure prediction.
"""

from typing import Optional, cast

import torch

from rna_predict.pipeline.stageA.input_embedding.current.transformer.diffusion import (
    DiffusionTransformer,
)


class AtomTransformer(torch.nn.Module):
    """
    Local transformer for atom embeddings with bias predicted from atom pair embeddings.
    Implements Algorithm 7 in AlphaFold3.
    """

    def __init__(
        self,
        c_atom: int = 128,  # Dimension for q (atom features)
        c_s: int = 384,  # Dimension for s (token-level style/conditioning)
        c_atompair: int = 16,  # Dimension for p/z (pair features)
        n_blocks: int = 3,
        n_heads: int = 4,
        n_queries: int = 32,
        n_keys: int = 128,
        blocks_per_ckpt: Optional[int] = None,
    ) -> None:
        """
        Initialize the AtomTransformer.

        Args:
            c_atom: Embedding dimension for atom features (q)
            c_s: Embedding dimension for token-level style/conditioning features (s)
            c_atompair: Embedding dimension for atom pair features (p/z)
            n_blocks: Number of blocks in the transformer
            n_heads: Number of attention heads
            n_queries: Local window size of query tensor for local attention
            n_keys: Local window size of key tensor for local attention
            blocks_per_ckpt: Number of blocks per checkpoint for memory efficiency
        """
        super(AtomTransformer, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.n_queries = n_queries
        self.n_keys = n_keys
        self.c_atom = c_atom
        self.c_s = c_s  # Store c_s
        self.c_atompair = c_atompair

        # Create the underlying diffusion transformer
        self.diffusion_transformer = DiffusionTransformer(
            n_blocks=n_blocks,
            n_heads=n_heads,
            c_a=c_atom,  # DiffusionTransformer expects atom dim for 'a' (input q)
            c_s=c_s,  # DiffusionTransformer expects style dim for 's' (input s)
            c_z=c_atompair,  # DiffusionTransformer expects pair dim for 'z' (input p)
            blocks_per_ckpt=blocks_per_ckpt,
        )

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
        if not isinstance(p, torch.Tensor):
            raise ValueError(f"Expected p to be a tensor, got {type(p)}")

        n_dims = p.dim()

        # Handle 4D tensor by reshaping to 3D or 5D
        if n_dims == 4:
            # Try to reshape to 3D by merging dimensions
            if p.shape[1] * p.shape[2] == p.shape[1] * p.shape[2]:
                p = p.reshape(p.shape[0], p.shape[1] * p.shape[2], p.shape[3])
                n_dims = 3  # Update n_dims after reshaping
            else:
                # If we can't reshape easily, raise error
                raise ValueError(
                    f"AtomTransformer: 4D 'p' is not supported. Got shape={p.shape}, dim={n_dims}."
                )

        # Check dimensions after potential reshaping
        if n_dims not in [3, 5]:
            raise ValueError(
                f"AtomTransformer: 'p' must be 3D or 5D. Got shape={p.shape}, dim={n_dims}"
            )

        # Fix last dimension if needed
        if p.shape[-1] != self.c_atompair:
            # For 3D tensor [batch, n, 1] or [batch, n, m]
            if n_dims == 3:
                if p.shape[-1] == 1:
                    # Expand last dimension from 1 to c_atompair
                    p = p.expand(*p.shape[:-1], self.c_atompair)
                else:
                    # Add dimension and expand
                    p = p.unsqueeze(-1).expand(*p.shape, self.c_atompair)
            # For 5D tensor
            elif n_dims == 5:
                if p.shape[-1] == 1:
                    # Expand last dimension from 1 to c_atompair
                    p = p.expand(*p.shape[:-1], self.c_atompair)
                else:
                    # Add dimension and expand
                    p = p.unsqueeze(-1).expand(*p.shape, self.c_atompair)

        return p

    def forward(
        self,
        q: torch.Tensor,  # Atom features (input 'a' to DiffusionTransformer)
        s: torch.Tensor,  # Token-level style/conditioning features (input 's' to DiffusionTransformer)
        p: torch.Tensor,  # Pair features (input 'z' to DiffusionTransformer)
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        n_queries: Optional[int] = None,  # Number of queries for local attention
        n_keys: Optional[int] = None,     # Number of keys for local attention
    ) -> torch.Tensor:
        """
        Process atom embeddings through the AtomTransformer, conditioned by token-level features.

        This method handles different tensor dimensions for the pair embedding 'p':
        - 5D tensor (trunk-based): p has shape [..., n_blocks, n_queries, n_keys, c_atompair]
        - 3D tensor (global attention): p has shape [..., N_atom, N_atom, c_atompair]

        Args:
            q: Atom embeddings of shape [..., N_atom, c_atom]
            s: Token-level style/conditioning embeddings of shape [..., N_token, c_s]
            p: Pair embeddings with varying dimensions based on use case
            inplace_safe: Whether inplace operations are safe
            chunk_size: Size of chunks for memory optimization

        Returns:
            Updated atom embeddings of shape [..., N_atom, c_atom]

        Raises:
            ValueError: If p has invalid dimensions or wrong shape
        """
        # Validate and fix p tensor
        p = self._validate_p_tensor(p)

        # Store original dimensions
        q_dim = q.dim()
        s_dim = s.dim()
        p_dim = p.dim()

        # Add batch dimension if needed
        if q_dim == 2:
            q = q.unsqueeze(0)
        if s_dim == 2:
            s = s.unsqueeze(0)
        if p_dim == 3:
            p = p.unsqueeze(0)

        # Determine attention type based on p's dimensions AFTER potential batch dim addition
        current_p_dim = p.dim()

        # Handle 5D/6D case - local or global attention with potential sample/block dims
        # DiffusionTransformer expects z to be [..., N, N, C] or [..., B, Q, K, C]
        # If p is 5D [B, 1, Nq, Nk, C] -> Local
        # If p is 5D [B, N, N, N, C] -> Global (N=N_atom)
        # If p is 6D [B, S, 1, Nq, Nk, C] -> Local
        # If p is 6D [B, S, N, N, N, C] -> Global (N=N_atom)
        # We pass p directly and let DiffusionTransformer handle it based on n_queries/n_keys presence.

        # Check if the shape suggests local attention (matches n_queries/n_keys)
        # Use -3 and -2 because the block dim might be present at -4
        # For this test, we'll force local attention if n_queries and n_keys are provided
        is_potentially_local = (
            (n_queries is not None and n_keys is not None) or
            (current_p_dim >= 4 and p.shape[-3] == self.n_queries and p.shape[-2] == self.n_keys)
        )

        # Debug print
        print(f"DEBUG: p.shape={p.shape}, current_p_dim={current_p_dim}, n_queries={n_queries}, n_keys={n_keys}")

        if is_potentially_local:
            # Assume local attention based on shape matching config
            n_q = self.n_queries
            n_k = self.n_keys
        else:
            # Assume global attention (or let DiffusionTransformer raise error if shape is wrong)
            n_q = None
            n_k = None

        # Process through diffusion transformer
        # Use provided n_queries and n_keys if available, otherwise use class attributes or None
        n_q_to_use = n_queries or self.n_queries if is_potentially_local else None
        n_k_to_use = n_keys or self.n_keys if is_potentially_local else None

        # Debug print
        print(f"DEBUG: Using n_queries={n_q_to_use}, n_keys={n_k_to_use}, is_potentially_local={is_potentially_local}")

        result = self.diffusion_transformer(
            a=q,  # Pass atom features as 'a'
            s=s,  # Pass token-level style features as 's'
            z=p,  # Pass pair features as 'z' (could be 4D, 5D, or 6D)
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

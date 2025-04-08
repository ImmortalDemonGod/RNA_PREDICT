"""
Coordinate processing components for atom attention.
"""

import torch

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)


class CoordinateProcessor:
    """Handles processing of coordinate-based features."""

    def __init__(self, c_atom: int, c_atompair: int, c_s: int, c_z: int):
        """
        Initialize the coordinate processor.

        Args:
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            c_s: Single embedding dimension
            c_z: Pair embedding dimension
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_s
        self.c_z = c_z

        # Set up coordinate-dependent components
        self._setup_components()

    def _setup_components(self) -> None:
        """Set up coordinate-dependent components."""
        # Style normalization and projection
        self.layernorm_s = LayerNorm(self.c_s)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_atom
        )

        # Pair embedding normalization and projection
        self.layernorm_z = LayerNorm(self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=self.c_z, out_features=self.c_atompair
        )

        # Position encoder
        self.linear_no_bias_r = LinearNoBias(in_features=3, out_features=self.c_atom)

    def process_coordinate_encoding(
        self, q_l: torch.Tensor, r_l: torch.Tensor, ref_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Process coordinate-based encoding.

        Args:
            q_l: Query tensor
            r_l: Reference tensor
            ref_pos: Reference positions

        Returns:
            Processed coordinate encoding
        """
        # Process reference positions
        r_l_processed = self.linear_no_bias_r(r_l)

        # Add coordinate encoding to query
        return q_l + r_l_processed

    def process_style_embedding(
        self,
        c_l: torch.Tensor,
        s: torch.Tensor,
        atom_to_token_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process style embedding with coordinate information.

        Args:
            c_l: Coordinate tensor
            s: Style tensor
            atom_to_token_idx: Mapping from atoms to tokens

        Returns:
            Processed style embedding
        """
        # Normalize and project style tensor
        s_normalized = self.layernorm_s(s)
        s_projected = self.linear_no_bias_s(s_normalized)

        # Broadcast to atom level and add to coordinate tensor
        s_broadcast = self.broadcast_to_atom_level(s_projected, atom_to_token_idx)
        return c_l + s_broadcast

    def process_pair_embedding(
        self, p_l: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """
        Process pair embedding with coordinate information.

        Args:
            p_l: Pair tensor
            z: Pair embedding tensor

        Returns:
            Processed pair embedding
        """
        # Normalize and project pair embedding
        z_normalized = self.layernorm_z(z)
        z_projected = self.linear_no_bias_z(z_normalized)

        # Add to pair tensor
        return p_l + z_projected

    def broadcast_to_atom_level(
        self, a_token: torch.Tensor, atom_to_token_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Broadcast token-level features to atom level.

        Args:
            a_token: Token-level features
            atom_to_token_idx: Mapping from atoms to tokens

        Returns:
            Atom-level broadcast features
        """
        return a_token[atom_to_token_idx]

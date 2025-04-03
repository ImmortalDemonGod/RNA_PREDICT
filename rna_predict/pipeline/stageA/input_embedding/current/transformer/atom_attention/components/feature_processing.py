"""
Feature processing components for atom attention.
"""

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.common import (
    InputFeatureDict,
    safe_tensor_access,
)
from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    aggregate_atom_to_token,
    broadcast_token_to_atom,
)


class FeatureProcessor:
    """Handles processing of atom features and embeddings."""

    def __init__(self, c_atom: int, c_atompair: int, c_s: int, c_z: int):
        """
        Initialize the feature processor.

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

        # Define expected feature dimensions
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 4 * 64,
        }

        # Set up encoders
        self._setup_encoders()

    def _setup_encoders(self) -> None:
        """Set up encoders for atom and distance features."""
        self.linear_no_bias_f = LinearNoBias(
            in_features=sum(self.input_feature.values()), out_features=self.c_atom
        )
        self.linear_no_bias_d = LinearNoBias(
            in_features=3, out_features=self.c_atompair
        )
        self.linear_no_bias_invd = LinearNoBias(
            in_features=3, out_features=self.c_atompair
        )
        self.linear_no_bias_v = LinearNoBias(
            in_features=1, out_features=self.c_atompair
        )

    def extract_atom_features(self, input_feature_dict: InputFeatureDict) -> torch.Tensor:
        """
        Extract and process atom features from input dictionary.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Processed atom features tensor
        """
        # Extract features
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")
        ref_mask = safe_tensor_access(input_feature_dict, "ref_mask")
        ref_element = safe_tensor_access(input_feature_dict, "ref_element")
        ref_atom_name_chars = safe_tensor_access(input_feature_dict, "ref_atom_name_chars")

        # Concatenate features
        features = torch.cat(
            [ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars], dim=-1
        )

        # Process through encoder
        return self.linear_no_bias_f(features)

    def create_pair_embedding(self, input_feature_dict: InputFeatureDict) -> torch.Tensor:
        """
        Create pair embeddings from atom features.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Pair embedding tensor of shape [num_atoms, num_atoms, c_atompair]
        """
        # Extract distance features
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")  # [N, 3]
        ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")  # [N, 1]

        # Get number of atoms
        num_atoms = ref_pos.shape[0]

        # Process distance features
        d = self.linear_no_bias_d(ref_pos)  # [N, c_atompair]

        # Calculate inverse distance features
        # First calculate inverse distances
        inv_pos = 1.0 / (ref_pos + 1e-6)  # [N, 3]
        invd = self.linear_no_bias_invd(inv_pos)  # [N, c_atompair]

        # Process charge features
        v = self.linear_no_bias_v(ref_charge)  # [N, c_atompair]

        # Combine features
        p = d + invd + v  # [N, c_atompair]

        # Create outer product to get pair features
        p_i = p.unsqueeze(1)  # [N, 1, c_atompair]
        p_j = p.unsqueeze(0)  # [1, N, c_atompair]
        p_ij = p_i + p_j  # [N, N, c_atompair]

        return p_ij

    def aggregate_to_token_level(
        self, a_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, num_tokens: int
    ) -> torch.Tensor:
        """
        Aggregate atom-level features to token level.

        Args:
            a_atom: Atom-level features
            atom_to_token_idx: Mapping from atoms to tokens
            num_tokens: Number of tokens

        Returns:
            Token-level aggregated features
        """
        return aggregate_atom_to_token(a_atom, atom_to_token_idx, num_tokens)

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
        return broadcast_token_to_atom(a_token, atom_to_token_idx) 
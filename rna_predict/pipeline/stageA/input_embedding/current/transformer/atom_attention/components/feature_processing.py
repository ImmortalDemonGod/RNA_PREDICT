"""
Feature processing components for atom attention.
"""

import torch
import logging

from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
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

logger = logging.getLogger("rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention.components.feature_processing")


class FeatureProcessor:
    """Handles processing of atom features and embeddings."""

    def __init__(self, c_atom: int, c_atompair: int, c_s: int, c_z: int, c_ref_element: int = 128, debug_logging: bool = False):
        """
        Initialize the feature processor.

        Args:
            c_atom: Atom embedding dimension
            c_atompair: Atom pair embedding dimension
            c_s: Single embedding dimension
            c_z: Pair embedding dimension
            c_ref_element: ref_element embedding dimension (config-driven)
            debug_logging: Whether to print debug logs
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_s
        self.c_z = c_z
        self.c_ref_element = c_ref_element
        self.debug_logging = debug_logging

        # --- EXPERIMENT: Force log handler and level for this logger ---
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(levelname)s][%(name)s] %(message)s')
        handler.setFormatter(formatter)
        if not logger.hasHandlers():
            logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        if self.debug_logging:
            logger.debug("TEST: FeatureProcessor constructed (forced handler)")
            logger.warning("TEST: FeatureProcessor WARNING (forced handler)")
            logger.error("TEST: FeatureProcessor ERROR (forced handler)")
            logger.debug("TEST: FeatureProcessor constructed")
            logger.debug(f"[FeatureProcessor] __init__ debug_logging={self.debug_logging}")
            logger.warning("TEST: FeatureProcessor WARNING")
            logger.error("TEST: FeatureProcessor ERROR")

        # Define expected feature dimensions (config-driven)
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": self.c_ref_element,
            "ref_atom_name_chars": 4 * 64,
        }
        if self.debug_logging:
            logger.debug(f"[DEBUG][FeatureProcessor] ref_element expected dim: {self.c_ref_element}")
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

    def extract_atom_features(
        self, input_feature_dict: InputFeatureDict
    ) -> torch.Tensor:
        """
        Extract and process atom features from input dictionary.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Processed atom features tensor
        """
        # Extract features
        # Safely access tensors with default values to avoid None issues
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")
        ref_mask = safe_tensor_access(input_feature_dict, "ref_mask")
        ref_element = safe_tensor_access(input_feature_dict, "ref_element")
        ref_atom_name_chars = safe_tensor_access(
            input_feature_dict, "ref_atom_name_chars"
        )

        # Ensure all tensors are not None before proceeding
        if ref_pos is None or ref_charge is None or ref_mask is None or ref_element is None or ref_atom_name_chars is None:
            # Create default tensors if any are missing
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            batch_size = 1
            n_atoms = 1

            # Try to get dimensions from any available tensor
            for tensor in [ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars]:
                if tensor is not None and tensor.dim() >= 2:
                    batch_size = tensor.shape[0]
                    n_atoms = tensor.shape[1]
                    device = tensor.device
                    break

            # Create default tensors for any missing ones
            if ref_pos is None:
                ref_pos = torch.zeros((batch_size, n_atoms, 3), device=device)
            if ref_charge is None:
                ref_charge = torch.zeros((batch_size, n_atoms, 1), device=device)
            if ref_mask is None:
                ref_mask = torch.ones((batch_size, n_atoms, 1), device=device)
            if ref_element is None:
                ref_element = torch.zeros((batch_size, n_atoms, self.c_ref_element), device=device)
            if ref_atom_name_chars is None:
                ref_atom_name_chars = torch.zeros((batch_size, n_atoms, 4 * 64), device=device)

        # Now we can safely check shapes
        if self.debug_logging:
            logger.debug(f"[DEBUG][FeatureProcessor] extract_atom_features: ref_element.shape={ref_element.shape}, expected={self.c_ref_element}")
        assert ref_element.shape[-1] == self.c_ref_element, (
            f"UNIQUE ERROR: ref_element last dim {ref_element.shape[-1]} does not match expected {self.c_ref_element}")
        # Concatenate features
        features = torch.cat(
            [ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars], dim=-1
        )

        # Process through encoder
        return self.linear_no_bias_f(features)

    def create_pair_embedding(
        self, input_feature_dict: InputFeatureDict
    ) -> torch.Tensor:
        """
        Create pair embeddings from atom features.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Pair embedding tensor of shape [num_atoms, num_atoms, c_atompair]
        """
        # Extract distance features
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")  # [N, 3]

        # Ensure ref_pos is available before using it
        if ref_pos is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ref_pos = torch.zeros((1, 1, 3), device=device)

        # Try to get ref_charge, but use a default if not available
        try:
            ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")  # [N, 1]
        except ValueError:
            # Create a default ref_charge tensor with zeros
            ref_charge = torch.zeros((ref_pos.shape[0], ref_pos.shape[1], 1), device=ref_pos.device)

        # Process distance features
        d = self.linear_no_bias_d(ref_pos)  # [N, c_atompair]

        # Calculate inverse distance features
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
        return broadcast_token_to_atom(a_token, atom_to_token_idx, debug_logging=self.debug_logging)

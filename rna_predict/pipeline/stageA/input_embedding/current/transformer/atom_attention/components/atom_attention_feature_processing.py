# Copied from feature_processing.py
"""
Feature processing components for atom attention.
"""

import torch
from typing import Optional, List, Tuple

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
            debug_logging: Whether to print debug logs (ignored in this implementation)
        """
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_s = c_s
        self.c_z = c_z
        self.c_ref_element = c_ref_element
        # debug_logging is accepted for interface compatibility, but ignored here

        # Define expected feature dimensions (config-driven)
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": self.c_ref_element,
            "ref_atom_name_chars": 4 * 64,
        }
        print(f"[DEBUG][FeatureProcessor] ref_element expected dim: {self.c_ref_element}")
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
        Extract atom features from input dictionary.

        Args:
            input_feature_dict: Dictionary containing atom features

        Returns:
            Tensor of atom features

        Raises:
            ValueError: If required features are missing
        """
        # Extract required features
        ref_pos = safe_tensor_access(input_feature_dict, "ref_pos")
        ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")
        ref_mask = safe_tensor_access(input_feature_dict, "ref_mask")
        ref_element = safe_tensor_access(input_feature_dict, "ref_element")
        ref_atom_name_chars = safe_tensor_access(input_feature_dict, "ref_atom_name_chars")

        # Ensure all tensors are not None before proceeding
        required_features = {
            "ref_pos": ref_pos,
            "ref_charge": ref_charge,
            "ref_mask": ref_mask,
            "ref_element": ref_element,
            "ref_atom_name_chars": ref_atom_name_chars
        }
        
        missing = [name for name, val in required_features.items() if val is None]
        if missing:
            raise ValueError(f"Missing required atom features: {', '.join(missing)}")

        # Now we can safely assert and use the tensors since we know they're not None
        assert ref_pos is not None  # For type checker
        assert ref_charge is not None  # For type checker
        assert ref_mask is not None  # For type checker
        assert ref_element is not None  # For type checker
        assert ref_atom_name_chars is not None  # For type checker

        print(f"[DEBUG][FeatureProcessor] extract_atom_features: ref_element.shape={ref_element.shape}, expected last dim={self.c_ref_element}")
        assert ref_element.shape[-1] == self.c_ref_element, (
            f"FeatureProcessor.extract_atom_features: ref_element.shape={ref_element.shape}, "
            f"expected last dim={self.c_ref_element} (from config)."
        )

        # All tensors are guaranteed to be not None at this point
        features = torch.cat([
            ref_pos,
            ref_charge,
            ref_mask,
            ref_element,
            ref_atom_name_chars
        ], dim=-1)

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

        # Try to get ref_charge, but use a default if not available
        try:
            ref_charge = safe_tensor_access(input_feature_dict, "ref_charge")  # [N, 1]
        except ValueError:
            # Create a default ref_charge tensor with zeros
            # First ensure ref_pos is not None
            if ref_pos is None:
                # Create a default ref_pos if it's None
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ref_pos = torch.zeros((1, 1, 3), device=device)
            ref_charge = torch.zeros((ref_pos.shape[0], 1), device=ref_pos.device)

        # Get number of atoms if ref_pos is not None
        if ref_pos is not None:
            ref_pos.shape[0]
        else:
            pass  # Default value if ref_pos is None

        # Process distance features
        d = self.linear_no_bias_d(ref_pos)  # [N, c_atompair]

        # Calculate inverse distance features
        # First calculate inverse distances
        # Ensure ref_pos is not None before using it
        if ref_pos is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ref_pos = torch.zeros((1, 3), device=device)
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
        self, a_token: torch.Tensor, atom_to_token_idx: torch.Tensor, num_atoms: Optional[int] = None
    ) -> torch.Tensor:
        """
        Broadcast token-level features to atom level.

        Args:
            a_token: Token-level features
            atom_to_token_idx: Mapping from atoms to tokens
            num_atoms: Number of atoms (not used, kept for backward compatibility)

        Returns:
            Atom-level features
        """
        # num_atoms parameter is not used by broadcast_token_to_atom
        return broadcast_token_to_atom(a_token, atom_to_token_idx)

    def process_atom_attention_features(
        self,
        pos: Optional[torch.Tensor] = None,
        charge: Optional[torch.Tensor] = None,
        element: Optional[torch.Tensor] = None,
        residue: Optional[torch.Tensor] = None,
        chain: Optional[torch.Tensor] = None,
        atom_type: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Process atom attention features into a standardized format.
        
        Args:
            pos: Position tensor
            charge: Charge tensor
            element: Element tensor
            residue: Residue tensor
            chain: Chain tensor
            atom_type: Atom type tensor
            mask: Mask tensor
            
        Returns:
            List of processed tensors
        """
        # Get device and dtype from first non-None tensor
        tensors = [t for t in [pos, charge, element, residue, chain, atom_type, mask] if t is not None]
        if not tensors:
            device = torch.device('cpu')
            dtype = torch.float32
        else:
            device = tensors[0].device
            dtype = tensors[0].dtype
        
        # Get maximum batch size and sequence length from non-None tensors
        shapes = [(t.shape[0], t.shape[1]) for t in tensors] if tensors else [(1, 1)]
        batch_size = max(shape[0] for shape in shapes)
        seq_len = max(shape[1] for shape in shapes)
        
        # Create default tensors for None inputs
        def create_tensor(t: Optional[torch.Tensor], default_shape: Tuple[int, ...]) -> torch.Tensor:
            if t is None:
                return torch.zeros(default_shape, dtype=dtype, device=device)
            return t

        pos = create_tensor(pos, (batch_size, seq_len, 3))
        charge = create_tensor(charge, (batch_size, seq_len))
        element = create_tensor(element, (batch_size, seq_len))
        residue = create_tensor(residue, (batch_size, seq_len))
        chain = create_tensor(chain, (batch_size, seq_len))
        atom_type = create_tensor(atom_type, (batch_size, seq_len))
        mask = create_tensor(mask, (batch_size, seq_len)) if mask is None else mask
        
        # Ensure all tensors have compatible shapes
        def expand_tensor(t: torch.Tensor, name: str) -> torch.Tensor:
            if t.shape[0] == 1 and batch_size > 1:
                t = t.expand(batch_size, -1, *t.shape[2:])
            if t.shape[1] == 1 and seq_len > 1:
                t = t.expand(-1, seq_len, *t.shape[2:])
            return t
        
        # Expand tensors to match largest dimensions
        pos = expand_tensor(pos, 'pos')
        charge = expand_tensor(charge, 'charge')
        element = expand_tensor(element, 'element')
        residue = expand_tensor(residue, 'residue')
        chain = expand_tensor(chain, 'chain')
        atom_type = expand_tensor(atom_type, 'atom_type')
        mask = expand_tensor(mask, 'mask')
        
        # Return list of tensors
        return [pos, charge, element, residue, chain, atom_type, mask]

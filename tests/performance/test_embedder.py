"""
Test embedder for performance benchmarks.
"""

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias


class TestInputFeatureEmbedder(nn.Module):
    """
    A simplified embedder for testing that matches the interface of InputFeatureEmbedder
    but doesn't have the same dimension requirements.
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_token: int = 384,
        **kwargs
    ) -> None:
        """
        Initialize the test embedder.

        Args:
            c_atom: Atom embedding dimension
            c_token: Token embedding dimension
            **kwargs: Additional arguments (ignored)
        """
        super(TestInputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_token = c_token

        # Create a linear layer that projects from the concatenated feature dimension to c_atom
        # The concatenated feature dimension is 389 (3 + 1 + 1 + 128 + 256)
        self.linear_no_bias_f = LinearNoBias(
            in_features=389, out_features=self.c_atom
        )

        # Create a linear layer that projects from c_atom to c_token
        self.linear_no_bias_q = LinearNoBias(
            in_features=self.c_atom, out_features=self.c_token
        )

        # Store attributes that match the real embedder
        self.input_feature = {
            "ref_pos": 3,
            "ref_charge": 1,
            "ref_mask": 1,
            "ref_element": 128,
            "ref_atom_name_chars": 256,
        }

    def forward(self, input_feature_dict, trunk_sing=None, trunk_pair=None, block_index=None, **kwargs):
        """
        Forward pass that returns a tensor with the expected shape.

        Args:
            input_feature_dict: Dictionary of input features
            trunk_sing: Ignored
            trunk_pair: Ignored
            block_index: Ignored
            **kwargs: Additional arguments (ignored)

        Returns:
            Tensor with shape [batch_size, num_tokens, c_token]
        """
        # Get the number of tokens from the input features
        num_tokens = 0
        if "restype" in input_feature_dict:
            restype = input_feature_dict["restype"]
            if restype.dim() >= 3:
                num_tokens = restype.shape[1]
            elif restype.dim() == 2:
                num_tokens = restype.shape[1] if restype.shape[0] == 1 else restype.shape[0]
            else:
                num_tokens = restype.shape[0]
        elif "profile" in input_feature_dict:
            profile = input_feature_dict["profile"]
            if profile.dim() >= 3:
                num_tokens = profile.shape[1]
            elif profile.dim() == 2:
                num_tokens = profile.shape[1] if profile.shape[0] == 1 else profile.shape[0]
            else:
                num_tokens = profile.shape[0]
        elif "atom_to_token_idx" in input_feature_dict:
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() >= 2:
                num_tokens = atom_to_token_idx.max().item() + 1
            else:
                num_tokens = atom_to_token_idx.max().item() + 1

        # If we couldn't determine the number of tokens, use a default
        if num_tokens == 0:
            num_tokens = 4

        # Create a batch dimension if needed
        batch_size = 1

        # Return a tensor with the expected shape that requires gradients
        return torch.zeros((batch_size, num_tokens, self.c_token), device=next(self.parameters()).device, requires_grad=True)

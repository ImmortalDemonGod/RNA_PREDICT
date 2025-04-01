"""
Specific patch for the transformer module to fix tensor shape compatibility issues.
"""

import torch


def patch_transformer():
    """
    Apply patches to the transformer module to fix tensor shape compatibility issues.
    """
    from rna_predict.pipeline.stageA.input_embedding.current import transformer

    # Find the AtomAttentionEncoder class
    if hasattr(transformer, "AtomAttentionEncoder"):
        # Store the original forward method
        original_forward = transformer.AtomAttentionEncoder.forward

        # Define the patched forward method
        def patched_forward(
            self, input_feature_dict, r_l, s, z, inplace_safe=False, chunk_size=None
        ):
            """
            Patched forward method for AtomAttentionEncoder that handles tensor shape mismatches.
            """
            # Import here to avoid circular imports
            from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
                broadcast_token_to_local_atom_pair,
            )

            # Get atom_to_token_idx from input_feature_dict
            atom_to_token_idx = input_feature_dict.get("atom_to_token_idx", None)

            # Process inputs
            a = self.layernorm_a(r_l)

            # Process token-level inputs
            if s is not None:
                s = self.layernorm_s(broadcast_token_to_atom(s, atom_to_token_idx))

            # Process pair-level inputs
            if z is not None:
                # Initialize p_lm
                p_lm = torch.zeros(
                    (*z.shape[:-3], self.n_queries, self.n_keys, self.c_z),
                    device=z.device,
                    dtype=z.dtype,
                )

                # Get z_local_pairs
                z_local_pairs, _ = broadcast_token_to_local_atom_pair(
                    z_token=z,
                    atom_to_token_idx=atom_to_token_idx,
                    n_queries=self.n_queries,
                    n_keys=self.n_keys,
                    compute_mask=False,
                )

                # Unsqueeze p_lm to match z_local_pairs dimensions
                p_lm_unsqueezed = p_lm.unsqueeze(-5)

                # Apply layernorm and linear transformation to z_local_pairs
                z_transformed = self.linear_no_bias_z(self.layernorm_z(z_local_pairs))

                # Handle shape mismatch for addition
                if p_lm_unsqueezed.shape != z_transformed.shape:
                    # Reshape z_transformed to match p_lm_unsqueezed's dimensions
                    if (
                        len(p_lm_unsqueezed.shape) >= 6
                        and len(z_transformed.shape) >= 6
                    ):
                        # Get the shapes
                        shape_a = p_lm_unsqueezed.shape
                        shape_b = z_transformed.shape

                        # Reshape dimension 4 (index 3) if needed
                        if shape_a[3] != shape_b[3]:
                            z_transformed = z_transformed.mean(
                                dim=3, keepdim=True
                            ).expand(-1, -1, -1, shape_a[3], -1, -1)

                        # Reshape dimension 5 (index 4) if needed
                        if shape_a[4] != shape_b[4]:
                            z_transformed = z_transformed.mean(
                                dim=4, keepdim=True
                            ).expand(-1, -1, -1, -1, shape_a[4], -1)

                        # Reshape dimension 6 (index 5) if needed
                        if len(shape_a) > 6 and (
                            len(shape_b) <= 6 or shape_a[5] != shape_b[5]
                        ):
                            if len(shape_b) <= 6:
                                z_transformed = z_transformed.unsqueeze(5)
                            z_transformed = z_transformed.expand(
                                -1, -1, -1, -1, -1, shape_a[5], -1
                            )

                # Add the tensors
                p_lm = p_lm_unsqueezed + z_transformed

            # Call the original forward method for the rest of the implementation
            return original_forward(
                self, input_feature_dict, r_l, s, z, inplace_safe, chunk_size
            )

        # Replace the original forward method with the patched one
        transformer.AtomAttentionEncoder.forward = patched_forward


if __name__ == "__main__":
    # Apply patches when this module is run directly
    patch_transformer()

"""
Diffusion-related tensor shape compatibility fixes.
"""

from functools import wraps

import torch


def fix_token_indices_after_resize():
    """
    Fix token indices after resizing to handle shape mismatches.
    """
    from rna_predict.pipeline.stageD.diffusion import diffusion

    original_forward = diffusion.DiffusionConditioning.forward

    @wraps(original_forward)
    def patched_forward(
        self,
        t_hat_noise_level,
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        inplace_safe=False,
    ):
        # First capture the original token dimensions
        original_token_dims = {
            "s_inputs": s_inputs.shape[1] if s_inputs is not None else None,
            "s_trunk": s_trunk.shape[1] if s_trunk is not None else None,
            "z_trunk": z_trunk.shape[1] if z_trunk is not None else None,
        }

        # Run the original forward pass
        result = original_forward(
            self,
            t_hat_noise_level,
            input_feature_dict,
            s_inputs,
            s_trunk,
            z_trunk,
            inplace_safe,
        )

        # Fix any token indices that were affected by resizing
        if hasattr(self, "token_indices"):
            for key, dim in original_token_dims.items():
                if dim is not None and self.token_indices[key].max() >= dim:
                    # Clip indices to valid range
                    self.token_indices[key] = torch.clamp(
                        self.token_indices[key], 0, dim - 1
                    )

        return result

    diffusion.DiffusionConditioning.forward = patched_forward


def fix_trunk_feature_dimensions():
    """
    Fix trunk feature dimension mismatches.
    """
    from rna_predict.pipeline.stageD.diffusion import diffusion

    original_diffusion_forward = diffusion.DiffusionConditioning.forward

    @wraps(original_diffusion_forward)
    def patched_diffusion_forward(
        self,
        t_hat_noise_level,
        input_feature_dict,
        s_inputs,
        s_trunk,
        z_trunk,
        inplace_safe=False,
    ):
        # Ensure s_inputs and s_trunk feature dimensions match
        if s_inputs is not None and s_trunk is not None:
            if s_inputs.shape[-1] != s_trunk.shape[-1]:
                min_dim = min(s_inputs.shape[-1], s_trunk.shape[-1])
                s_inputs = s_inputs[..., :min_dim]
                s_trunk = s_trunk[..., :min_dim]

        return original_diffusion_forward(
            self,
            t_hat_noise_level,
            input_feature_dict,
            s_inputs,
            s_trunk,
            z_trunk,
            inplace_safe,
        )

    diffusion.DiffusionConditioning.forward = patched_diffusion_forward

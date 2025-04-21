from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn

from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
    FourierEmbedding,
    RelativePositionEncoding,
)
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
    Transition,
)

from .diffusion_utils import (
    InputFeatureDict,
    create_zero_tensor_like,
    validate_tensor_shapes,
)


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3 for conditioning the diffusion process.

    This module handles the conditioning of both pair and single features
    for the diffusion process, combining trunk features with input features
    and noise embeddings.
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        """
        Initialize the diffusion conditioning module.

        Args:
            sigma_data: Standard deviation of the data
            c_z: Hidden dimension for pair embedding
            c_s: Hidden dimension for single embedding
            c_s_inputs: Input embedding dimension from InputEmbedder
            c_noise_embedding: Noise embedding dimension
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        self.c_noise_embedding = c_noise_embedding

        # Pair feature processing
        self.relpe = RelativePositionEncoding(c_z=c_z)
        print(f"[DEBUG][DiffusionConditioning] layernorm_z normalized_shape: {2 * self.c_z}")
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Single feature processing
        print(f"[DEBUG][DiffusionConditioning] layernorm_s normalized_shape: {self.c_s + self.c_s_inputs}")
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs, out_features=self.c_s
        )

        # Noise embedding processing
        self.fourier_embedding = FourierEmbedding(c=self.c_noise_embedding)
        print(f"[DEBUG][DiffusionConditioning] layernorm_n normalized_shape: {self.c_noise_embedding}")
        self.layernorm_n = LayerNorm(self.c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(
            in_features=self.c_noise_embedding, out_features=self.c_s
        )

        # Additional transitions
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

        # Assertions to catch mismatches early
        assert isinstance(self.c_z, int) and self.c_z > 0, (
            f"UNIQUE ERROR: c_z must be positive int, got {self.c_z}")
        assert isinstance(self.c_s, int) and self.c_s > 0, (
            f"UNIQUE ERROR: c_s must be positive int, got {self.c_s}")
        assert isinstance(self.c_s_inputs, int) and self.c_s_inputs > 0, (
            f"UNIQUE ERROR: c_s_inputs must be positive int, got {self.c_s_inputs}")
        assert isinstance(self.c_noise_embedding, int) and self.c_noise_embedding > 0, (
            f"UNIQUE ERROR: c_noise_embedding must be positive int, got {self.c_noise_embedding}")

    def _process_pair_features(
        self,
        z_trunk: torch.Tensor,
        input_feature_dict: InputFeatureDict,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Process pair features through the conditioning pipeline."""
        N_tokens = z_trunk.shape[-2]
        input_feature_dict["expected_n_tokens"] = N_tokens

        relpe_output = self.relpe(input_feature_dict)

        # Ensure feature dimensions match
        if relpe_output.shape[-1] != z_trunk.shape[-1]:
            if relpe_output.shape[-1] < z_trunk.shape[-1]:
                padding = create_zero_tensor_like(
                    relpe_output,
                    (
                        *relpe_output.shape[:-1],
                        z_trunk.shape[-1] - relpe_output.shape[-1],
                    ),
                )
                relpe_output = torch.cat([relpe_output, padding], dim=-1)
            else:
                relpe_output = relpe_output[..., : z_trunk.shape[-1]]

        # --- Start Dimension Alignment (Batch & Sample) ---
        # Ensure relpe_output has the same number of dimensions as z_trunk
        # If z_trunk is 5D ([B, S, N, N, C]) and relpe_output is 4D ([B, N, N, C]),
        # add the sample dimension at index 1.
        if z_trunk.ndim == 5 and relpe_output.ndim == 4:
            # Add sample dimension of size 1 at index 1
            relpe_output = relpe_output.unsqueeze(1) # Shape becomes [B, 1, N, N, C]
        elif relpe_output.ndim < z_trunk.ndim:
             # Fallback for other dimension mismatches (less common)
             # This might need adjustment based on expected scenarios
             while relpe_output.ndim < z_trunk.ndim:
                 relpe_output = relpe_output.unsqueeze(0) # Add leading dimensions

        # Expand batch dimension (dim 0) if necessary (only if added via fallback)
        # This check might be redundant if the primary case (5D vs 4D) handles batch correctly.
        if relpe_output.shape[0] == 1 and z_trunk.shape[0] > 1 and relpe_output.ndim == z_trunk.ndim:
             relpe_output = relpe_output.expand(z_trunk.shape[0], *relpe_output.shape[1:])
        # --- End Dimension Alignment ---

        # --- Start Sample Dimension Alignment Fix ---
        # Handle cases where both are 5D but sample dimension (dim 1) might mismatch
        if z_trunk.ndim == 5 and relpe_output.ndim == 5:
            # If sample dimensions don't match, assume relpe_output's dim 1 was added
            # artificially and needs expansion to match z_trunk's sample dimension.
            if relpe_output.shape[1] != z_trunk.shape[1]:
                # Expand relpe_output's sample dimension to match z_trunk
                try: # Add try-except for safety, although expand should work if shapes are compatible otherwise
                    relpe_output = relpe_output.expand(-1, z_trunk.shape[1], -1, -1, -1)
                except RuntimeError as e:
                     raise RuntimeError(
                         f"Failed to expand relpe_output sample dimension ({relpe_output.shape}) to match z_trunk ({z_trunk.shape}). Error: {e}"
                     )

        # Final check before concatenation
        if z_trunk.shape[:-1] != relpe_output.shape[:-1]:
             raise RuntimeError(
                 f"Shape mismatch before concatenation (excluding last dim). "
                 f"z_trunk: {z_trunk.shape}, relpe_output: {relpe_output.shape}"
             )
        # --- End Sample Dimension Alignment Fix ---

        print(f"[DEBUG PRE-CAT] z_trunk shape: {z_trunk.shape}, relpe_output shape: {relpe_output.shape}") # Keep for verification
        pair_z = torch.cat([z_trunk, relpe_output], dim=-1)
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))

        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)

        return pair_z

    def _process_single_features(
        self,
        s_trunk: torch.Tensor,
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        """Process single features through the conditioning pipeline."""
        print(f"[STAGED DEBUG] _process_single_features: s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}, c_s={self.c_s}, c_s_inputs={self.c_s_inputs}, expected_in_features={self.c_s + self.c_s_inputs}")

        # Validate and adapt tensor shapes if needed
        adapted_s_trunk, adapted_s_inputs = validate_tensor_shapes(s_trunk, s_inputs, self.c_s, self.c_s_inputs)
        print(f"[STAGED DEBUG] After validate_tensor_shapes: adapted_s_trunk.shape={adapted_s_trunk.shape}, adapted_s_inputs.shape={adapted_s_inputs.shape}")

        # Concatenate the adapted tensors
        single_s = torch.cat([adapted_s_trunk, adapted_s_inputs], dim=-1)
        print(f"[STAGED DEBUG] After cat: single_s.shape={single_s.shape}, expected={self.c_s + self.c_s_inputs}")
        single_s = self.layernorm_s(single_s)
        print(f"[STAGED DEBUG] After layernorm_s: single_s.shape={single_s.shape}")
        assert single_s.shape[-1] == self.linear_no_bias_s.in_features, (
            f"UNIQUE ERROR: single_s.shape[-1] ({single_s.shape[-1]}) does not match linear_no_bias_s.in_features ({self.linear_no_bias_s.in_features}) [c_s={self.c_s}, c_s_inputs={self.c_s_inputs}]")
        single_s = self.linear_no_bias_s(single_s)
        print(f"[STAGED DEBUG] After linear_no_bias_s: single_s.shape={single_s.shape}")

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

        print(f"[STAGED DEBUG] After transitions: single_s.shape={single_s.shape}")
        return single_s

    def _ensure_input_feature_dict(
        self,
        input_feature_dict: Dict[str, Union[torch.Tensor, int, float, Dict[Any, Any]]],
        t_hat_noise_level: torch.Tensor,
    ) -> InputFeatureDict:
        """Ensure the input feature dictionary has required keys and correct types."""
        result: InputFeatureDict = {
            "ref_charge": torch.tensor(0.0, device=t_hat_noise_level.device),
            "ref_pos": torch.tensor(0.0, device=t_hat_noise_level.device),
            "expected_n_tokens": 0,
        }

        if "ref_charge" not in input_feature_dict:
            if "ref_pos" in input_feature_dict and isinstance(
                input_feature_dict["ref_pos"], torch.Tensor
            ):
                result["ref_charge"] = create_zero_tensor_like(
                    input_feature_dict["ref_pos"],
                    input_feature_dict["ref_pos"].shape[:-1],
                )
            else:
                result["ref_charge"] = create_zero_tensor_like(
                    t_hat_noise_level, (1, 0)
                )
        else:
            ref_charge = input_feature_dict["ref_charge"]
            if isinstance(ref_charge, torch.Tensor):
                result["ref_charge"] = ref_charge
            elif isinstance(ref_charge, (int, float)):
                result["ref_charge"] = torch.tensor(
                    float(ref_charge), device=t_hat_noise_level.device
                )
            else:
                raise ValueError(f"Invalid type for ref_charge: {type(ref_charge)}")

        if "ref_pos" in input_feature_dict and isinstance(
            input_feature_dict["ref_pos"], torch.Tensor
        ):
            result["ref_pos"] = input_feature_dict["ref_pos"]

        return result

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: Dict[str, Union[torch.Tensor, int, float, Dict[Any, Any]]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the diffusion conditioning module.

        Args:
            t_hat_noise_level: Noise level tensor [..., N_sample]
            input_feature_dict: Input meta feature dictionary
            s_inputs: Single embedding from InputFeatureEmbedder [..., N_tokens, c_s_inputs]
            s_trunk: Single feature embedding from PairFormer [..., N_tokens, c_s]
            z_trunk: Pair feature embedding from PairFormer [..., N_tokens, N_tokens, c_z]
            inplace_safe: Whether to use inplace operations

        Returns:
            Tuple of processed single and pair embeddings
        """
        print(f"[STAGED DEBUG] DiffusionConditioning.forward: c_s={self.c_s}, c_s_inputs={self.c_s_inputs}, expected_in_features={self.c_s + self.c_s_inputs}")
        print(f"[STAGED DEBUG] s_trunk.shape={s_trunk.shape}, s_inputs.shape={s_inputs.shape}")

        # Ensure required keys are present with correct types
        processed_input_dict = self._ensure_input_feature_dict(
            input_feature_dict, t_hat_noise_level
        )

        # Process pair features
        if z_trunk is None:
            batch_dims = s_trunk.shape[:-2]
            z_trunk = create_zero_tensor_like(
                s_trunk, (*batch_dims, s_trunk.shape[-2], s_trunk.shape[-2], self.c_z)
            )
        pair_z = self._process_pair_features(
            z_trunk, processed_input_dict, inplace_safe
        )

        # Process single features
        if s_inputs is None:
            batch_dims = s_trunk.shape[:-2]
            s_inputs = create_zero_tensor_like(
                s_trunk, (*batch_dims, s_trunk.shape[-2], self.c_s_inputs)
            )

        # --- Start Shape Alignment Fix ---
        # Ensure s_trunk and s_inputs have compatible shapes before processing
        # Expected shapes: s_trunk [B, N_sample, N, C_s], s_inputs [B, N_sample, N, C_s_inputs] or [B, N, C_s_inputs]

        processed_s_inputs = s_inputs
        processed_s_trunk = s_trunk

        # 1. Ensure s_inputs is 4D if s_trunk is 4D
        if processed_s_trunk.ndim == 4 and processed_s_inputs.ndim == 3:
            processed_s_inputs = processed_s_inputs.unsqueeze(1) # Add N_sample dim

        # 2. Ensure N_sample dimension (dim 1) matches
        if processed_s_trunk.ndim == 4 and processed_s_inputs.ndim == 4:
            n_sample_trunk = processed_s_trunk.shape[1]
            n_sample_inputs = processed_s_inputs.shape[1]

            if n_sample_trunk != n_sample_inputs:
                if n_sample_inputs == 1:
                    # Expand s_inputs to match s_trunk's N_sample
                    processed_s_inputs = processed_s_inputs.expand(
                        -1, n_sample_trunk, -1, -1
                    )
                elif n_sample_trunk == 1:
                     # Expand s_trunk to match s_inputs' N_sample (less likely but possible)
                     processed_s_trunk = processed_s_trunk.expand(
                         -1, n_sample_inputs, -1, -1
                     )
                else:
                    # If both have N_sample > 1 but they differ, it's an unexpected state
                    raise RuntimeError(
                        f"N_sample dimension mismatch cannot be resolved by expansion. "
                        f"s_trunk: {processed_s_trunk.shape}, s_inputs: {processed_s_inputs.shape}"
                    )
        elif processed_s_trunk.ndim != processed_s_inputs.ndim:
             # If dimensions still don't match after unsqueeze attempt
             raise RuntimeError(
                 f"Dimension mismatch after unsqueeze attempt. "
                 f"s_trunk: {processed_s_trunk.shape}, s_inputs: {processed_s_inputs.shape}"
             )

        # --- End Shape Alignment Fix ---

        # Call _process_single_features with aligned tensors
        single_s = self._process_single_features(processed_s_trunk, processed_s_inputs, inplace_safe)

        return single_s, pair_z

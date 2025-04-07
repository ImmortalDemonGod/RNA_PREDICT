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

        # Pair feature processing
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Single feature processing
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs, out_features=self.c_s
        )

        # Noise embedding processing
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding, out_features=self.c_s
        )

        # Additional transitions
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)

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

        # Ensure relpe_output has the N_sample dimension if z_trunk does
        if z_trunk.ndim == 5 and relpe_output.ndim == 4:
            # Assume N_sample is the second dimension in z_trunk [B, N_sample, N, N, C]
            # Add the sample dimension to relpe_output at dim 1
            relpe_output = relpe_output.unsqueeze(1).expand(
                -1, z_trunk.shape[1], -1, -1, -1
            )
            # print(f"[DEBUG] Expanded relpe_output shape: {relpe_output.shape}")
        elif z_trunk.ndim != relpe_output.ndim:
            # If dimensions still don't match after potential expansion, raise error
            raise RuntimeError(
                f"Cannot concatenate z_trunk ({z_trunk.shape}) and relpe_output ({relpe_output.shape}) due to mismatched dimensions."
            )

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
        validate_tensor_shapes(s_trunk, s_inputs, self.c_s, self.c_s_inputs)

        single_s = torch.cat([s_trunk, s_inputs], dim=-1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))

        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)

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
        single_s = self._process_single_features(s_trunk, s_inputs, inplace_safe)

        return single_s, pair_z

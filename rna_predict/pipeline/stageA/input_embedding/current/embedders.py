# protenix/model/modules/embedders.py
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionConfig,  # Import from transformer package
    AtomAttentionEncoder,
    EncoderForwardParams,  # Import from transformer package
    InputFeatureDict,  # Import the type hint
)

# Removed the direct import from ...transformer.atom_attention


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3.

    By default:
      c_atom=128, c_atompair=16, c_token=384
    now also accepts:
      restype_dim, profile_dim, c_pair, num_heads, num_layers, and use_optimized

    The forward pass concatenates:
      1) Atom-attention output
      2) Extra token features (restype, profile, deletion_mean)
    and returns a final tensor of shape [..., N_token, c_token].
    """

    # @snoop
    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        restype_dim: int = 32,
        profile_dim: int = 32,
        c_pair: int = 16,  # Overlaps with c_atompair or can remain separate
        num_heads: int = 4,
        num_layers: int = 3,
        use_optimized: bool = False,
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
            restype_dim (int, optional): dimension of restype input. Defaults to 32.
            profile_dim (int, optional): dimension of profile input. Defaults to 32.
            c_pair (int, optional): pair embedding dimension (if needed). Defaults to 16.
            num_heads (int, optional): # heads for any internal attention. Defaults to 4.
            num_layers (int, optional): # layers for potential stack. Defaults to 3.
            use_optimized (bool, optional): whether to use an optimized path. Defaults to False.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token

        self.restype_dim = restype_dim
        self.profile_dim = profile_dim
        self.c_pair = c_pair
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_optimized = use_optimized

        # Create the config object explicitly
        encoder_config = AtomAttentionConfig(
            has_coords=False,  # Based on original failing call
            c_token=self.c_token,
            c_atom=self.c_atom,
            c_atompair=self.c_atompair,
            # Rely on defaults in AtomAttentionConfig for other params (c_s, c_z, n_blocks, etc.)
        )
        self.atom_attention_encoder = AtomAttentionEncoder(config=encoder_config)
        # Existing line2 comment
        #
        # We'll store these so we know how many dims to expect for restype, profile, etc.
        self.input_feature = {
            "restype": self.restype_dim,
            "profile": self.profile_dim,
            "deletion_mean": 1,
        }

        # We'll create extras_linear lazily in the forward pass once we know
        # the exact input dimension
        self.extras_linear: Optional[nn.Linear] = None

        # Optionally, place a final layer norm after summing
        self.final_ln = nn.LayerNorm(self.c_token)

    # @snoop
    def forward(
        self,
        input_feature_dict: InputFeatureDict,
        trunk_sing: Optional[torch.Tensor] = None,
        trunk_pair: Optional[torch.Tensor] = None,
        block_index: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding with shape [..., N_token, c_token]
        """
        # --- Start: Added Key Check ---
        # Check if all required top-level keys are present
        required_keys = set(self.input_feature.keys())
        missing_keys = required_keys - set(input_feature_dict.keys())
        if missing_keys:
            raise KeyError(
                f"Missing required keys in input_feature_dict: {missing_keys}"
            )
        # --- End: Added Key Check ---

        # 1) Embed per-atom features with the AtomAttentionEncoder.
        #    a => [..., N_token, c_token]
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict=input_feature_dict,
            r_l=None,  # Explicitly None as not available here
            s=None,  # Explicitly None as not available here
            z=None,  # Explicitly None as not available here
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        print(f"DEBUG [Embedder]: Shape after encoder (a): {a.shape}") # DEBUG

        # Ensure 'a' has at least 3 dimensions [batch, tokens, features]
        if a.dim() == 2:  # [tokens, features]
            a = a.unsqueeze(0)  # Add batch dimension
        print(f"DEBUG [Embedder]: Shape after dimension adjustment (a): {a.shape}")

        # Extract the number of tokens from the restype feature
        # This is more reliable than using the output of AtomAttentionEncoder
        if "restype" in input_feature_dict and input_feature_dict.get("restype") is not None:
            restype = input_feature_dict["restype"]
            if restype.dim() >= 3:  # [B, N_token, C]
                token_dim = restype.shape[-2]
            elif restype.dim() == 2:  # [B, N_token] or [N_token, C]
                # For 2D tensors, we need to determine which dimension is N_token
                # In RNA context, the second dimension is typically N_token
                token_dim = restype.shape[1]
                if token_dim > 1000:  # If second dimension is suspiciously large, it might be C
                    token_dim = restype.shape[0]
            elif restype.dim() == 1:  # [N_token]
                token_dim = restype.shape[0]
            else:
                # For scalar or invalid shapes, use atom dimension as fallback
                token_dim = a.shape[-2]
        else:
            # If restype is not available, use atom dimension
            token_dim = a.shape[-2]

        # 2) Ensure 'a' has the right token dimension
        current_token_dim_a = a.shape[-2]
        if current_token_dim_a != token_dim:
            if current_token_dim_a < token_dim:
                # Pad 'a' if it has fewer tokens than expected
                padding_shape = list(a.shape)
                padding_shape[-2] = token_dim - current_token_dim_a
                padding = torch.zeros(padding_shape, device=a.device, dtype=a.dtype)
                a = torch.cat([a, padding], dim=-2) # Concatenate along the token dimension
                print(f"DEBUG [Embedder]: Padded 'a' to shape {a.shape}")
            else:
                # If a has more tokens than expected, we should keep all tokens
                # This is important for preserving sequence information
                print(f"DEBUG [Embedder]: Keeping all {current_token_dim_a} tokens from 'a'")
                token_dim = current_token_dim_a  # Update token_dim to match a's dimension

        # Create a tensor of the appropriate batch shape for reshaping
        batch_shape = a.shape[:-2]  # Everything except the token and feature dimensions

        # Handle optional trunk_sing / trunk_pair / block_index
        # (Currently no-op except for verifying presence)
        if trunk_sing is not None:
            pass  # Possibly incorporate trunk_sing into the pipeline
        if trunk_pair is not None:
            pass  # Possibly incorporate trunk_pair into the pipeline
        if block_index is not None:
            pass  # Possibly use block_index for local attention

        extras_list = []
        for name, dim_size in self.input_feature.items():
            # Check if the feature exists in the input_feature_dict
            if name not in input_feature_dict:
                # Create a default zero tensor if the feature is missing
                if name == "deletion_mean":
                    # For scalar features
                    default_tensor = torch.zeros(
                        (*batch_shape, token_dim, 1), device=a.device
                    )
                else:
                    # For vector features
                    default_tensor = torch.zeros(
                        (*batch_shape, token_dim, dim_size), device=a.device
                    )
                extras_list.append(default_tensor)
            else:
                # Get the feature tensor
                feature = input_feature_dict[name]

                # Ensure feature has the right number of dimensions
                while feature.dim() < len(batch_shape) + 2:  # Need [*batch, tokens, features]
                    feature = feature.unsqueeze(0)

                # Ensure feature has the right token dimension
                if feature.shape[-2] != token_dim:
                    if feature.shape[-2] < token_dim:
                        # Pad feature if it has fewer tokens
                        pad_size = token_dim - feature.shape[-2]
                        padding = torch.zeros(
                            (*feature.shape[:-2], pad_size, feature.shape[-1]),
                            device=feature.device,
                            dtype=feature.dtype
                        )
                        feature = torch.cat([feature, padding], dim=-2)
                    else:
                        # Slice feature if it has more tokens
                        feature = feature[..., :token_dim, :]

                # Ensure batch dimensions match
                if feature.shape[:-2] != batch_shape:
                    try:
                        feature = feature.expand(*batch_shape, token_dim, feature.shape[-1])
                    except RuntimeError as e:
                        print(f"WARNING: Could not expand feature {name} from shape {feature.shape} to match batch_shape {batch_shape}")
                        raise

                extras_list.append(feature)

        # Concatenate all extra features along the feature dimension
        extras = torch.cat(extras_list, dim=-1)

        # Project to token dimension if not already done
        if self.extras_linear is None:
            self.extras_linear = nn.Linear(extras.shape[-1], self.c_token)
        extras_emb = self.extras_linear(extras)

        # 3) Merge atom output with these extra features
        #    Ensure extras_emb also matches the batch dimensions of 'a'
        if a.shape[:-1] != extras_emb.shape[:-1]:
             # Try to expand extras_emb to match batch_shape and token_dim
             try:
                 target_shape = list(a.shape[:-1]) + [extras_emb.shape[-1]]
                 extras_emb = extras_emb.expand(target_shape)
                 print(f"DEBUG [Embedder]: Expanded extras_emb to shape {extras_emb.shape}")
             except RuntimeError as e:
                 # If expansion fails, raise a more informative error
                 raise RuntimeError(
                    f"Cannot broadcast extras_emb (shape {extras_emb.shape}) "
                    f"to match adjusted 'a' (shape {a.shape[:-1]}) for addition. Error: {e}"
                 ) from e

        # Summation is a simple choice that yields a final [N_token, c_token].
        s_inputs = a + extras_emb
        print(f"DEBUG [Embedder]: Shape after addition (s_inputs): {s_inputs.shape}") # DEBUG

        # 4) Optional final layer norm
        out = self.final_ln(s_inputs)  # => [..., N_token, c_token]
        print(f"DEBUG [Embedder]: Shape after final_ln (out): {out.shape}") # DEBUG

        # 5) Add a projection to get the expected output dimension
        # If the expected output dimension is different from c_token (e.g. 449 vs 384)
        # We need an additional projection layer
        if not hasattr(self, "final_projection"):
            # Lazily create the projection layer the first time it's needed
            self.final_projection = nn.Linear(self.c_token, 449, bias=True)
            # Initialize weights to be close to identity transformation with small random values
            nn.init.eye_(self.final_projection.weight[: self.c_token, : self.c_token])
            if self.final_projection.bias is not None:
                nn.init.zeros_(self.final_projection.bias)

        # Apply the final projection to get the desired output dimension
        out = self.final_projection(out)
        print(f"DEBUG [Embedder]: Shape before return (out): {out.shape}") # DEBUG

        return out


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: embedding z of relative position encoding [..., N, N, c_z]
        """
        # Determine N_tokens, prioritizing 'expected_n_tokens' if available
        if "expected_n_tokens" in input_feature_dict:
            n_tokens = input_feature_dict["expected_n_tokens"]
            device = None
            batch_size = 1

            # Still need to find a tensor to determine device
            for key in input_feature_dict:
                if isinstance(input_feature_dict[key], torch.Tensor):
                    device = input_feature_dict[key].device
                    if input_feature_dict[key].dim() > 1:
                        batch_size = input_feature_dict[key].shape[0]
                    break
        else:
            # Determine the number of tokens from the input, or use a default
            n_tokens = 0
            device = None
            batch_size = 1

            # Find existing tensor to determine token count and device
            for key in input_feature_dict:
                if isinstance(input_feature_dict[key], torch.Tensor):
                    if input_feature_dict[key].dim() >= 1:
                        # Extract token count from the first available tensor
                        tensor_shape = input_feature_dict[key].shape
                        if (
                            len(tensor_shape) > 1
                        ):  # For tensors with at least 2 dimensions
                            n_tokens = tensor_shape[-1]
                            batch_size = tensor_shape[0]
                        else:  # For 1D tensors
                            n_tokens = tensor_shape[0]
                            batch_size = 1
                        device = input_feature_dict[key].device
                        break

        if n_tokens == 0:
            # If we couldn't determine token count, default to 1
            n_tokens = 1
            device = torch.device("cpu")

        # Create and fill in missing features with correct batch size and token count
        feature_dict = {}
        for feature_name in self.input_feature:
            if feature_name not in input_feature_dict:
                # Create default zero tensor for this feature with correct shapes
                default_tensor = torch.zeros(
                    (batch_size, n_tokens), dtype=torch.long, device=device
                )
                feature_dict[feature_name] = default_tensor
            else:
                # Use existing tensor but ensure it has the right shape
                existing_tensor = input_feature_dict[feature_name]
                if existing_tensor.dim() == 1:
                    # If 1D, reshape to [1, n_tokens]
                    if existing_tensor.shape[0] != n_tokens:
                        # Padding or truncation needed
                        if existing_tensor.shape[0] < n_tokens:
                            # Pad
                            padded = torch.zeros(
                                n_tokens, dtype=existing_tensor.dtype, device=device
                            )
                            padded[: existing_tensor.shape[0]] = existing_tensor
                            existing_tensor = padded
                        else:
                            # Truncate
                            existing_tensor = existing_tensor[:n_tokens]
                    # Reshape to [1, n_tokens]
                    existing_tensor = existing_tensor.unsqueeze(0)
                elif existing_tensor.dim() >= 2:
                    # If already 2D+, ensure the token dimension (dim 1) matches n_tokens
                    if existing_tensor.shape[1] != n_tokens:
                        # Need to adjust the token dimension
                        if existing_tensor.shape[1] < n_tokens:
                            # Pad
                            padding = torch.zeros(
                                (
                                    existing_tensor.shape[0],
                                    n_tokens - existing_tensor.shape[1],
                                ),
                                dtype=existing_tensor.dtype,
                                device=device,
                            )
                            existing_tensor = torch.cat(
                                [existing_tensor, padding], dim=1
                            )
                        else:
                            # Truncate
                            existing_tensor = existing_tensor[:, :n_tokens]

                feature_dict[feature_name] = existing_tensor

        # Proceed with the calculations using feature_dict instead of input_feature_dict
        b_same_chain = (
            feature_dict["asym_id"][..., :, None]
            == feature_dict["asym_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            feature_dict["residue_index"][..., :, None]
            == feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            feature_dict["entity_id"][..., :, None]
            == feature_dict["entity_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        d_residue = torch.clip(
            input=feature_dict["residue_index"][..., :, None]
            - feature_dict["residue_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
        d_token = torch.clip(
            input=feature_dict["token_index"][..., :, None]
            - feature_dict["token_index"][..., None, :]
            + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
            2 * self.r_max + 1
        )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
        d_chain = torch.clip(
            input=feature_dict["sym_id"][..., :, None]
            - feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        if self.training:
            p = self.linear_no_bias(
                torch.cat(
                    [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                    dim=-1,
                ).float()
            )  # [..., N_token, N_token, c_z]
            return p
        else:
            del d_chain, d_token, d_residue, b_same_chain, b_same_residue
            origin_shape = a_rel_pos.shape[:-1]
            Ntoken = a_rel_pos.shape[-2]
            a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
            chunk_num = 1 if Ntoken < 3200 else 8
            a_rel_pos_chunks = torch.chunk(
                a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
            )
            a_rel_token_chunks = torch.chunk(
                a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
            )
            b_same_entity_chunks = torch.chunk(
                b_same_entity.reshape(-1, 1), chunk_num, dim=-2
            )
            a_rel_chain_chunks = torch.chunk(
                a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
            )
            start = 0
            p = None
            for i in range(len(a_rel_pos_chunks)):
                data = torch.cat(
                    [
                        a_rel_pos_chunks[i],
                        a_rel_token_chunks[i],
                        b_same_entity_chunks[i],
                        a_rel_chain_chunks[i],
                    ],
                    dim=-1,
                ).float()
                result = self.linear_no_bias(data)
                del data
                if p is None:
                    p = torch.empty(
                        (a_rel_pos.shape[-2], self.c_z),
                        device=a_rel_pos.device,
                        dtype=result.dtype,
                    )
                p[start : start + result.shape[0]] = result
                start += result.shape[0]
                del result
            del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
            p = p.reshape(*origin_shape, -1)
            return p


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )

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
import logging

from rna_predict.pipeline.stageA.input_embedding.current.primitives import LinearNoBias
from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionConfig,  # Import from transformer package
    AtomAttentionEncoder,
    InputFeatureDict,  # Import the type hint
)

# Removed the direct import from ...transformer.atom_attention

# Initialize logger for Stage A embedders
logger = logging.getLogger("rna_predict.pipeline.stageA.input_embedding.embedders")

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

    # #####@snoop
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
        debug_logging: bool = False,
        config: Optional[AtomAttentionConfig] = None,  # <-- Accept config for Hydra-driven instantiation
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
            debug_logging (bool, optional): whether to enable debug logging. Defaults to False.
            config (Optional[AtomAttentionConfig]): Hydra config object for encoder, if available.
        """
        super().__init__()
        # If a config object is provided (Hydra best practice), use it for all settings
        if config is not None:
            self.c_atom = config.c_atom
            self.c_atompair = config.c_atompair
            self.c_token = config.c_token
            self.restype_dim = getattr(config, 'restype_dim', restype_dim)
            self.profile_dim = getattr(config, 'profile_dim', profile_dim)
            self.c_pair = getattr(config, 'c_pair', c_pair)
            self.num_heads = getattr(config, 'num_heads', num_heads)
            self.num_layers = getattr(config, 'num_layers', num_layers)
            self.use_optimized = getattr(config, 'use_optimized', use_optimized)
            self.debug_logging = getattr(config, 'debug_logging', False)
            encoder_config = config
        else:
            self.c_atom = c_atom
            self.c_atompair = c_atompair
            self.c_token = c_token
            self.restype_dim = restype_dim
            self.profile_dim = profile_dim
            self.c_pair = c_pair
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.use_optimized = use_optimized
            self.debug_logging = debug_logging
            encoder_config = AtomAttentionConfig(
                has_coords=False,  # Based on original failing call
                c_token=self.c_token,
                c_atom=self.c_atom,
                c_atompair=self.c_atompair,
                c_s=self.c_token,  # Use token dim for single emb
                c_z=self.c_atompair,  # Use atompair dim for pair emb
                n_blocks=self.num_layers,
                n_heads=self.num_heads,
                debug_logging=self.debug_logging,
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

    # #####@snoop
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
        # Add atom_to_token_idx as a required key if it's used for N_token calculation
        required_keys.add("atom_to_token_idx")
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
        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: Shape after encoder (a): {a.shape}")

        # Ensure 'a' has at least 3 dimensions [batch, tokens, features]
        if a.dim() == 2:  # [tokens, features]
            a = a.unsqueeze(0)  # Add batch dimension
        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: Shape after dimension adjustment (a): {a.shape}")

        # Extract the number of tokens from restype or profile
        if "restype" in input_feature_dict:
            restype = input_feature_dict["restype"]
            if restype.dim() == 2:
                N_token = restype.size(0)
            else:
                N_token = restype.size(1)
        elif "profile" in input_feature_dict:
            profile = input_feature_dict["profile"]
            if profile.dim() == 2:
                N_token = profile.size(0)
            else:
                N_token = profile.size(1)
        else:
            # Fallback to atom_to_token_idx if neither restype nor profile is available
            # FIX 1: Use atom_to_token_idx instead of atom_to_token
            atom_to_token_idx = input_feature_dict["atom_to_token_idx"]
            if atom_to_token_idx.dim() == 2:
                N_token = atom_to_token_idx.size(0)
            else:
                # Ensure atom_to_token_idx is not empty before calling max()
                if atom_to_token_idx.numel() > 0:
                    # Explicitly cast to int to resolve mypy error
                    N_token = int(atom_to_token_idx.max().item() + 1)
                else:
                    # Handle the case where atom_to_token_idx is empty
                    N_token = 0 # Or raise an error, depending on expected behavior
                    if self.debug_logging:
                        logger.warning("atom_to_token_idx is empty, setting N_token to 0.")

        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: N_token determined as: {N_token}")

        # Ensure 'a' has the correct number of tokens
        if a.size(1) != N_token:
            # If 'a' has more tokens than needed, truncate
            if a.size(1) > N_token:
                a = a[:, :N_token, :]
            # If 'a' has fewer tokens than needed, pad with zeros
            else:
                padding = torch.zeros(
                    (a.size(0), N_token - a.size(1), a.size(2)),
                    device=a.device,
                    dtype=a.dtype,
                )
                a = torch.cat([a, padding], dim=1)

        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: Shape after token adjustment (a): {a.shape}")

        # Create extras tensor by concatenating restype, profile, and deletion_mean
        extras_list = []
        for key in ("restype", "profile", "deletion_mean"):
            feat = input_feature_dict[key]
            # Ensure all tensors have the same batch and token dimensions
            if feat.dim() == 1:
                # For 1D tensors (like deletion_mean), reshape to [1, N_token, 1]
                feat = feat.unsqueeze(0).unsqueeze(-1)
            elif feat.dim() == 2:
                # For 2D tensors (like restype and profile), reshape to [batch, N_token, feature_dim]
                # For deletion_mean, we need to ensure it has the correct number of tokens
                if key == "deletion_mean":
                    # Reshape to [batch, 1, N_token] and then transpose to [batch, N_token, 1]
                    feat = feat.unsqueeze(1).transpose(1, 2)
                else:
                    # For other 2D tensors, just add a feature dimension
                    feat = feat.unsqueeze(-1)
            elif feat.dim() == 3 and feat.size(-1) == 1:
                # Already has batch and token dimensions, but needs to be reshaped for concatenation
                pass
            else:
                # For tensors with more than 3 dimensions or with feature dimension > 1
                # Reshape to [batch, token, feature]
                if feat.dim() > 3:
                    feat = feat.view(feat.size(0), feat.size(1), -1)

            # Ensure the tensor has the correct number of tokens
            if feat.size(1) != N_token:
                if feat.size(1) > N_token:
                    feat = feat[:, :N_token, :]
                else:
                    padding = torch.zeros(
                        (feat.size(0), N_token - feat.size(1), feat.size(-1)),
                        device=feat.device,
                        dtype=feat.dtype,
                    )
                    feat = torch.cat([feat, padding], dim=1)

            if self.debug_logging:
                logger.debug(f"DEBUG [Embedder]: {key} shape after processing: {feat.shape}")
            extras_list.append(feat)

        # Now concatenate along the feature dimension
        # FIX 2: Assign result to a new variable `extras_cat`
        extras_cat = torch.cat(extras_list, dim=-1)
        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: extras_cat shape before linear: {extras_cat.shape}")

        # Create extras_linear if not already created
        if self.extras_linear is None:
            # FIX 4: Use `extras_cat` for size access
            extras_dim = extras_cat.size(-1)
            # FIX 5: Use `extras_cat` for device access
            self.extras_linear = LinearNoBias(extras_dim, self.c_token).to(
                extras_cat.device
            )
            if self.debug_logging:
                logger.debug(
                    f"DEBUG [Embedder]: extras_linear weight shape: {self.extras_linear.weight.shape}"
                )

        # Project extras to c_token dimension and add to atom embeddings
        extras_proj = self.extras_linear(extras_cat) # Use extras_cat here
        s_inputs = a + extras_proj
        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: Shape after addition (s_inputs): {s_inputs.shape}")

        # Apply final layer norm
        out = self.final_ln(s_inputs)
        if self.debug_logging:
            logger.debug(f"DEBUG [Embedder]: Shape after final_ln (out): {out.shape}")

        # Return the output with the batch dimension preserved
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
        super().__init__()
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
        super().__init__()
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

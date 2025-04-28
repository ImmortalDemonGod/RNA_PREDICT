# protenix/model/generator.py
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

from typing import Any, Callable, Optional

import torch
import logging

from rna_predict.pipeline.stageA.input_embedding.current.utils import (
    centre_random_augmentation,
)


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
        debug_logging: bool = False,
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
            debug_logging (bool, optional): Whether to print debug statements. Defaults to False.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        if debug_logging:
            print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
        debug_logging: bool = False,
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            p (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
            debug_logging (bool, optional): Whether to print debug statements. Defaults to False.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        if debug_logging:
            print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    debug_logging: bool = False,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    logger = logging.getLogger(__name__)
    # Ensure noise_schedule is a 1D tensor
    noise_schedule = noise_schedule.flatten()

    # Get number of atoms from input_feature_dict
    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)

    # Determine device/dtype from inputs
    ref_tensor = s_inputs if s_inputs is not None else s_trunk
    device = ref_tensor.device
    dtype = ref_tensor.dtype

    # Determine TRUE batch shape (leading dimensions before N_sample, N_atom, 3)
    # Use the shape of a reliable input tensor like s_trunk or s_inputs.
    ref_shape = ref_tensor.shape

    # FIXED: Simplified batch shape determination to avoid 5D tensors
    # We only want a single batch dimension to avoid creating 5D tensors
    # that would cause errors in the diffusion module's forward method
    true_batch_shape = ref_shape[:1]  # Just take the first dimension as batch

    # Log the decision for debugging
    if debug_logging:
        logger.debug(f"[DEBUG][FIXED] Using simplified true_batch_shape={true_batch_shape} from ref_shape={ref_shape}")

    if debug_logging:
        logger.debug(f"[DEBUG] Determined true_batch_shape: {true_batch_shape} from ref_shape {ref_shape} and N_sample {N_sample}")

    # Ensure true_batch_shape is a tuple
    if not isinstance(true_batch_shape, tuple):
        true_batch_shape = (true_batch_shape,)


    def _chunk_sample_diffusion(
        chunk_n_sample: int, inplace_safe: bool
    ) -> torch.Tensor:
        """Process a chunk of samples."""
        # Initialize noise using the true_batch_shape and chunk_n_sample
        # FIXED: Ensure we don't create a 5D tensor by flattening any extra batch dimensions
        x_l_shape = (true_batch_shape[0], chunk_n_sample, N_atom, 3)
        if debug_logging:
            logger.debug(f"[DEBUG][FIXED] Initializing noise x_l with shape: {x_l_shape}")

        x_l = noise_schedule[0] * torch.randn(
            size=x_l_shape,
            device=device,
            dtype=dtype,
        )

        # Process each step in the noise schedule
        for step, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            # Calculate gamma and t_hat
            gamma = float(gamma0) if c_tau > gamma_min else 0.0
            t_hat = c_tau_last * (gamma + 1.0)

            # Add noise for predictor step
            delta_noise_level = torch.sqrt(
                torch.clamp(t_hat**2 - c_tau_last**2, min=0.0)
            )
            x_noisy = x_l + noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )

            # Reshape t_hat for broadcasting: needs shape [B, chunk_n_sample] or similar
            # FIXED: Ensure t_hat has the right shape to match our simplified x_l_shape
            t_hat_target_shape = (true_batch_shape[0], chunk_n_sample)
            t_hat = t_hat.reshape(1).expand(true_batch_shape[0]).unsqueeze(-1).expand(*t_hat_target_shape).to(dtype)
            if debug_logging:
                logger.debug(f"[DEBUG][Generator Loop {step}] Reshaped t_hat shape: {t_hat.shape} to broadcast with x_noisy {x_noisy.shape}")

            # Denoise step
            if debug_logging:
                logger.debug(f"[DEBUG][Generator Loop {step}] Before denoise_net - x_noisy: {x_noisy.shape}, t_hat: {t_hat.shape}")
            if debug_logging:
                logger.debug("[DEBUG-STAGED-PIPELINE] About to call denoise_net with args:")
                logger.debug("  x_noisy: %s", x_noisy.shape if x_noisy is not None else None)
                logger.debug("  t_hat: %s", t_hat.shape if t_hat is not None else None)
                logger.debug("  input_feature_dict: %s", list(input_feature_dict.keys()) if isinstance(input_feature_dict, dict) else None)
                logger.debug("  s_inputs: %s", s_inputs.shape if s_inputs is not None else None)
                logger.debug("  s_trunk: %s", s_trunk.shape if s_trunk is not None else None)
                logger.debug("  z_trunk: %s", z_trunk.shape if z_trunk is not None else None)
                logger.debug("  chunk_size: %s", attn_chunk_size)
                logger.debug("  inplace_safe: %s", inplace_safe)
            if x_noisy is None or t_hat is None:
                raise ValueError("[ERR-STAGED-PIPELINE-001] x_noisy or t_hat is None before denoise_net call")

            # Call denoise_net, expect (coords, loss) tuple
            denoise_result = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                debug_logging=debug_logging,  # Enable debug logging for diffusion module
            )

            # Unpack the tuple
            if not isinstance(denoise_result, tuple) or len(denoise_result) != 2:
                 raise TypeError(f"denoise_net expected to return (coords, loss) tuple, but got {type(denoise_result)}")
            x_denoised, _ = denoise_result # Unpack tuple, ignore loss

            # Update x_l using Euler step
            # Add epsilon for stability, ensure t_hat broadcasts correctly [B, chunk_n_sample, 1, 1]
            delta = (x_noisy - x_denoised) / (
                t_hat.view(*t_hat.shape, 1, 1) + 1e-8
            )
            # Ensure dt broadcasts correctly [B, chunk_n_sample, 1, 1]
            dt = (c_tau - t_hat).view(*t_hat.shape, 1, 1)
            x_l = x_noisy + step_scale_eta * dt * delta

        if debug_logging:
            logger.debug(
                f"[DEBUG][sample_diffusion] Returning _chunk_sample_diffusion output shape: {x_l.shape}"
            )
        return x_l

    # Process all samples or in chunks
    if diffusion_chunk_size is None:
        return _chunk_sample_diffusion(N_sample, inplace_safe)
    else:
        # Process in chunks
        n_chunks = (N_sample + diffusion_chunk_size - 1) // diffusion_chunk_size
        results = []

        for i in range(n_chunks):
            start_idx = i * diffusion_chunk_size
            end_idx = min((i + 1) * diffusion_chunk_size, N_sample)
            chunk_size = end_idx - start_idx

            chunk_result = _chunk_sample_diffusion(chunk_size, inplace_safe)
            results.append(chunk_result)

        # Concatenate results along the sample dimension (which follows the true batch dimensions)
        sample_dim_index = len(true_batch_shape)
        final_result = torch.cat(results, dim=sample_dim_index)
        if debug_logging:
            logger.debug(
                f"[DEBUG][sample_diffusion] Returning chunked output shape: {final_result.shape}"
            )
        return final_result


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    debug_logging: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
        device (torch.device): device to use for all allocations
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    logging.getLogger(__name__)
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    dtype = label_dict["coordinate"].dtype
    if debug_logging:
        print(f"[DEVICE-DEBUG][StageD] sample_diffusion_training: device={device}")
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(device=device, dtype=dtype)

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma_size_list = list(batch_size_shape) + [N_sample]
    sigma_size = torch.Size(sigma_size_list)
    sigma = noise_sampler(size=sigma_size, device=device).to(dtype)
    if debug_logging:
        print(f"[DEVICE-DEBUG][StageD] sample_diffusion_training: sigma.device={sigma.device}")
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype, device=device) * sigma[..., None, None]
    if debug_logging:
        print(f"[DEVICE-DEBUG][StageD] sample_diffusion_training: noise.device={noise.device}")

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        # Call denoise_net, expect (coords, loss) tuple
        denoise_result = denoise_net(
            x_noisy=x_gt_augment + noise,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            debug_logging=debug_logging, # Enable debug logging for diffusion module
        )
        # Unpack the tuple
        if not isinstance(denoise_result, tuple) or len(denoise_result) != 2:
             raise TypeError(f"denoise_net expected to return (coords, loss) tuple, but got {type(denoise_result)}")
        x_denoised, _ = denoise_result # Ignore loss for training sample generation
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = (x_gt_augment + noise)[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            # Call denoise_net, expect (coords, loss) tuple
            denoise_result_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                debug_logging=debug_logging, # Enable debug logging for diffusion module
            )
            # Unpack the tuple
            if not isinstance(denoise_result_i, tuple) or len(denoise_result_i) != 2:
                 raise TypeError(f"denoise_net expected to return (coords, loss) tuple, but got {type(denoise_result_i)}")
            x_denoised_i, _ = denoise_result_i # Ignore loss
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
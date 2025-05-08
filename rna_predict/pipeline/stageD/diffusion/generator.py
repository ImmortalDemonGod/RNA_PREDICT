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

logger = logging.getLogger(__name__)

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
        """
        Initializes a sampler for noise levels used during diffusion model training.
        
        Args:
            p_mean: Mean of the Gaussian distribution for noise sampling.
            p_std: Standard deviation of the Gaussian distribution for noise sampling.
            sigma_data: Scaling factor applied to sampled noise levels.
            debug_logging: If True, enables debug logging during initialization.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        if debug_logging:
            print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device
    ) -> torch.Tensor:
        """
        Samples noise levels from a log-normal distribution for training.
        
        Args:

            size (torch.Size): the target size
            device (torch.device): target device (required).

        Returns:
            A tensor of sampled noise levels with the specified shape and device.
        
        Raises:
            ValueError: If device is not provided.
        """
        # Device is now a required parameter
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
        device: torch.device,
        N_step: int = 200,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generates a schedule of noise levels for diffusion inference steps.
        
        Args:

            device (torch.device): target device (required).
            N_step (int, optional): number of time steps. Defaults to 200.
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            A tensor of shape [N_step + 1] containing noise levels for each diffusion step, with the final value set to zero.
        """
        # Device is now a required parameter
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
    """
    Generates denoised protein structure coordinates using a diffusion process.
    
    Performs iterative denoising of randomly initialized coordinates according to a provided noise schedule, using a denoising network and input features. Supports chunked processing for memory efficiency. Returns the final denoised coordinates for all generated samples.
    
    Args:
        N_sample: Number of samples to generate.
        gamma0: Initial gamma parameter controlling noise scaling.
        gamma_min: Minimum gamma threshold for noise scaling.
        noise_scale_lambda: Scaling factor for added noise at each step.
        step_scale_eta: Scaling factor for the Euler update step.
        diffusion_chunk_size: If set, processes samples in chunks of this size.
        inplace_safe: If True, enables in-place operations where safe.
        attn_chunk_size: If set, chunks attention computation in the denoising network.
        debug_logging: If True, enables detailed debug logging.
    
    Returns:
        A tensor of denoised coordinates with shape [..., N_sample, N_atom, 3].
    """
    logger = logging.getLogger(__name__)
    # Ensure noise_schedule is a 1D tensor
    noise_schedule = noise_schedule.flatten()

    def _chunk_sample_diffusion(
        chunk_n_sample: int, inplace_safe: bool
    ) -> torch.Tensor:
        """Process a chunk of samples."""
        # --- BEGIN: Ensure all required variables are in local scope ---
        # Get number of atoms from input_feature_dict
        N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
        # Determine device/dtype from inputs
        ref_tensor = s_inputs if s_inputs is not None else s_trunk
        device = ref_tensor.device
        dtype = ref_tensor.dtype
        # Determine TRUE batch shape (leading dimensions before N_sample, N_atom, 3)
        ref_shape = ref_tensor.shape
        true_batch_shape = ref_shape[:-1]  # Keep every leading dim up to the feature dimension
        sample_dim = len(true_batch_shape)  # Track the sample dimension for consistent handling
        if debug_logging:
            logger.debug(f"[DEBUG] Using full true_batch_shape={true_batch_shape} from ref_shape={ref_shape}")
            logger.debug(f"[DEBUG] Sample dimension index: {sample_dim}")
            logger.debug(f"[DEBUG] Determined true_batch_shape: {true_batch_shape} from ref_shape {ref_shape} and N_sample {N_sample}")
        # --- END: Ensure all required variables are in local scope ---
        # Print relevant shapes for diagnosis before constructing x_l
        if debug_logging:
            logger.debug(f"[DEBUG][Generator] true_batch_shape: {true_batch_shape}")
            logger.debug(f"[DEBUG][Generator] chunk_n_sample: {chunk_n_sample}")
            logger.debug(f"[DEBUG][Generator] N_atom: {N_atom}")
        # PATCH: Always construct x_l_shape as [B, chunk_n_sample, N_atom, 3]
        B = 1 if len(true_batch_shape) == 1 else true_batch_shape[0]
        # If true_batch_shape is (B, N_atom), set B = true_batch_shape[0], N_atom = true_batch_shape[1]
        # But N_atom is already determined above
        x_l_shape = (B, chunk_n_sample, N_atom, 3)
        if debug_logging:
            logger.debug(f"[PATCHED][Generator] For diffusion, forcing x_l_shape = [B, N_sample, N_atom, 3] = {x_l_shape}")
        x_l = torch.randn(size=x_l_shape, device=device, dtype=dtype) * noise_schedule[0]

        # Process each step in the noise schedule
        from rna_predict.pipeline.stageD.run_stageD import log_mem
        import gc
        for step, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule[:-1], noise_schedule[1:])
        ):
            gc.collect()
            log_mem(f"Before diffusion step {step}")
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

            # Print relevant shapes for diagnosis
            if debug_logging:
                logger.debug(f"[DEBUG][Generator Loop {step}] x_noisy.shape: {x_noisy.shape}")
                logger.debug(f"[DEBUG][Generator Loop {step}] t_hat before reshape: {t_hat}")
                logger.debug(f"[DEBUG][Generator Loop {step}] t_hat type: {type(t_hat)}")

            # Handle t_hat properly based on its type
            if isinstance(t_hat, (int, float)):
                # If t_hat is a scalar, convert to tensor with shape [B, chunk_n_sample]
                t_hat = torch.full((B, chunk_n_sample), t_hat, device=device, dtype=dtype)
            elif isinstance(t_hat, torch.Tensor) and t_hat.numel() == 1:
                # If t_hat is a tensor with a single element, expand it
                t_hat = t_hat.expand(B, chunk_n_sample).to(dtype)
            else:
                # Otherwise, try to reshape it
                try:
                    t_hat = t_hat.reshape(B, chunk_n_sample).to(dtype)
                except RuntimeError as e:
                    if debug_logging:
                        logger.error(f"[ERROR] Failed to reshape t_hat with shape {t_hat.shape} to {(B, chunk_n_sample)}: {e}")
                    # Create a new tensor with the correct shape filled with the first value of t_hat
                    t_hat_value = t_hat.item() if t_hat.numel() == 1 else t_hat.flatten()[0].item()
                    t_hat = torch.full((B, chunk_n_sample), t_hat_value, device=device, dtype=dtype)
            if debug_logging:
                logger.debug(f"[PATCHED][Generator Loop {step}] t_hat shape for diffusion: {t_hat.shape} (should be [B, N_sample])")
            # Defensive assertion: t_hat should have shape [B, N_sample]
            assert t_hat.shape == (B, chunk_n_sample), (
                f"[GENERATOR PATCH] t_hat shape {t_hat.shape} does not match expected [B, N_sample]={[B, chunk_n_sample]} for diffusion module"
            )

            # --- SYSTEMATIC DEBUGGING: Print input_feature_dict keys and types before denoise_net call ---
            if debug_logging:
                logger.debug(f"[DEBUG][Generator] input_feature_dict keys: {list(input_feature_dict.keys())}")
                for k, v in input_feature_dict.items():
                    logger.debug(f"[DEBUG][Generator] input_feature_dict[{k}]: type={type(v)}, is_tensor={isinstance(v, torch.Tensor)}")
                logger.debug(f"[DEBUG][Generator] atom_to_token_idx: {input_feature_dict.get('atom_to_token_idx', None)}")
            # Defensive: Ensure atom_to_token_idx is present and a Tensor
            if not ("atom_to_token_idx" in input_feature_dict and isinstance(input_feature_dict["atom_to_token_idx"], torch.Tensor)):
                raise ValueError("[GENERATOR PATCH] input_feature_dict missing 'atom_to_token_idx' or it is not a Tensor before denoise_net call")

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
            )

            # Unpack the tuple
            if not isinstance(denoise_result, tuple) or len(denoise_result) != 2:
                 raise TypeError(f"denoise_net expected to return (coords, loss) tuple, but got {type(denoise_result)}")
            x_denoised, _ = denoise_result # Unpack tuple, ignore loss

            gc.collect()
            log_mem(f"After diffusion step {step}")

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

        # Concatenate results along the sample dimension (dim=1)
        devices = [r.device for r in results]
        if len(set(devices)) > 1:
            print(f"[ERROR][sample_diffusion] Device mismatch: {[str(d) for d in devices]}")
            for i, r in enumerate(results):
                print(f"  Result {i} shape: {r.shape}, device: {r.device}")
        final_result = torch.cat(results, dim=1)
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
    device: Optional[torch.device] = None,
    debug_logging: bool = False,
) -> tuple[torch.Tensor, ...]:
    """
    Performs diffusion-based training by adding noise to ground-truth coordinates and denoising.
    
    Randomly augments ground-truth coordinates, samples noise levels, adds Gaussian noise, and applies the denoising network. Supports optional chunked processing for memory efficiency. Returns the augmented coordinates, denoised outputs, and noise levels used.
    """
    # Initialize logger if needed for future use
    # logger = logging.getLogger(__name__)
    if device is None:
        raise ValueError(
            "sample_diffusion_training now requires an explicit `device`. "
            "Pass `device=label_dict['coordinate'].device` or cfg.device."
        )
    # Areate N_sample versions of the input structure by randomly rotating and translating
    x_gt_augment = centre_random_augmentation(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
    ).to(device=device, dtype=label_dict["coordinate"].dtype)

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma_size_list = list(label_dict["coordinate"].shape[:-2]) + [N_sample]
    sigma_size = torch.Size(sigma_size_list)
    sigma = noise_sampler(size=sigma_size, device=device).to(label_dict["coordinate"].dtype)
    if hasattr(self, 'debug_logging') and self.debug_logging:
        logger.info(f"[DEVICE-DEBUG][StageD] sample_diffusion_training: sigma.device={sigma.device}")
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=label_dict["coordinate"].dtype, device=device) * sigma[..., None, None]
    if hasattr(self, 'debug_logging') and self.debug_logging:
        logger.info(f"[DEVICE-DEBUG][StageD] sample_diffusion_training: noise.device={noise.device}")

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
        if debug_logging:
            print(f"[DEBUG][StageD] sample_diffusion_training: no_chunks={no_chunks}, diffusion_chunk_size={diffusion_chunk_size}, N_sample={N_sample}")
        if no_chunks == 0:
            # Defensive: If there are no chunks, return empty tensor with correct shape
            x_denoised = torch.empty(0, device=device, dtype=label_dict["coordinate"].dtype)
        else:
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
                )
                # Unpack the tuple
                if not isinstance(denoise_result_i, tuple) or len(denoise_result_i) != 2:
                     raise TypeError(f"denoise_net expected to return (coords, loss) tuple, but got {type(denoise_result_i)}")
                x_denoised_i, _ = denoise_result_i # Ignore loss
                x_denoised.append(x_denoised_i)
            x_denoised = torch.cat(x_denoised, dim=-3)

    return x_gt_augment, x_denoised, sigma
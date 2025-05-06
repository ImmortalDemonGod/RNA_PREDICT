import torch
from rna_predict.pipeline.stageD.diffusion.generator import TrainingNoiseSampler, sample_diffusion_training

# Create a mock denoise_net function
def mock_denoise_net(x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk):
    # Return a tuple (coords, loss)
    return x_noisy - 0.1, torch.tensor(0.0, device=x_noisy.device)

# Create test data
noise_sampler = TrainingNoiseSampler()
label_dict = {
    "coordinate": torch.zeros((1, 4, 3), dtype=torch.float32),
    "coordinate_mask": torch.ones((1, 4), dtype=torch.bool),
}
input_feature_dict = {
    "atom_to_token_idx": torch.zeros((1, 4), dtype=torch.long)
}
s_inputs = torch.randn(1, 4, 8)
s_trunk = torch.randn(1, 4, 16)
z_trunk = torch.randn(1, 4, 4, 16)

# Test with device
print("Testing with device...")
try:
    x_gt_aug, x_denoised, sigma = sample_diffusion_training(
        noise_sampler=noise_sampler,
        denoise_net=mock_denoise_net,
        label_dict=label_dict,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        N_sample=3,
        device=torch.device("cpu")
    )
    print(f"Success! Shapes: x_gt_aug={x_gt_aug.shape}, x_denoised={x_denoised.shape}, sigma={sigma.shape}")
except Exception as e:
    print(f"Error with device: {e}")

# Test without device
print("\nTesting without device...")
try:
    x_gt_aug, x_denoised, sigma = sample_diffusion_training(
        noise_sampler=noise_sampler,
        denoise_net=mock_denoise_net,
        label_dict=label_dict,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        N_sample=3,
        device=None
    )
    print(f"ERROR: Should have failed without device")
except ValueError as e:
    print(f"Expected error without device: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

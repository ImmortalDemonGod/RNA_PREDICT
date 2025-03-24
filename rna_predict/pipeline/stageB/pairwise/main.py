import torch
import sys

# 1) For building input embeddings
from rna_predict.pipeline.stageB.pairwise.protenix_integration import ProtenixIntegration

# 2) For running diffusion in Stage D
from rna_predict.pipeline.stageD.run_stageD import run_stageD_diffusion

def demo_run_protenix_embeddings():
    """
    Demonstrates building single + pair embeddings using ProtenixIntegration
    (Stage B/C synergy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_token = 12
    N_atom = N_token * 4  # Ensure N_atom is exactly 4 times N_token
    # Generate random positions and normalize to [-1, 1] range
    ref_pos = torch.randn(N_atom, 3, device=device)
    # Normalize to [-1, 1] range by finding max absolute value and dividing
    max_abs_val = torch.max(torch.abs(ref_pos))
    ref_pos = ref_pos / max_abs_val if max_abs_val > 0 else ref_pos
    
    input_features = {
        "ref_pos": ref_pos,
        "ref_charge": torch.randint(-2, 3, (N_atom,), device=device).float(),
        "ref_element": 2 * torch.rand(N_atom, 128, device=device) - 1,  # Uniform distribution in [-1, 1]
        "ref_atom_name_chars": 0.1 * torch.randn(N_atom, 16, device=device),  # Smaller scale for better stability
        "atom_to_token": torch.repeat_interleave(torch.arange(N_token, device=device), 4),
        "restype": 2 * torch.rand(N_token, 32, device=device) - 1,  # Uniform distribution in [-1, 1]
        "profile": 2 * torch.rand(N_token, 32, device=device) - 1,  # Uniform distribution in [-1, 1]
        "deletion_mean": torch.randn(N_token,device=device),
        "residue_index": torch.arange(N_token, device=device),
    }

    integrator = ProtenixIntegration(
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        device=device
    )

    embeddings = integrator.build_embeddings(input_features)
    s_inputs = embeddings["s_inputs"]
    z_init = embeddings["z_init"]
    print("[Embedding Demo] s_inputs shape:", s_inputs.shape)
    print("[Embedding Demo] z_init shape:", z_init.shape)

def demo_run_diffusion():
    """
    Demonstrates Stage D diffusion usage with partial coords and trunk embeddings
    for global refinement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suppose partial_coords is from Stage C, or random
    partial_coords = torch.randn(1, 10, 3, device=device)
    trunk_embeddings = {
        "sing": torch.randn(1, 10, 384, device=device),
        "pair": torch.randn(1, 10, 10, 32, device=device)
    }
    diffusion_config = {
        "c_x": 3,      # input coords dimension
        "c_s": 384,    # single embedding dimension
        "c_z": 32,     # pair embedding dimension
        "num_layers": 4
    }

    # 1) Inference mode
    coords_final = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="inference",
        device=device
    )
    print("[Diffusion Demo] coords_final shape:", coords_final.shape)

    # 2) Training mode
    x_denoised, loss, sigma = run_stageD_diffusion(
        partial_coords,
        trunk_embeddings,
        diffusion_config,
        mode="train",
        device=device
    )
    print(f"[Diffusion Demo] Train step => x_denoised shape: {x_denoised.shape}, loss={loss.item():.4f}, sigma={sigma:.2f}")

def main():
    print("=== Running Protenix Integration Demo (Embeddings) ===")
    demo_run_protenix_embeddings()

    print("\n=== Running Stage D Diffusion Demo ===")
    demo_run_diffusion()

if __name__ == "__main__":
    main()
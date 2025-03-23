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
    N_atom = 40
    N_token = 12

    input_features = {
        "ref_pos": torch.randn(N_atom,3, device=device),
        "ref_charge": torch.randint(-2,3,(N_atom,), device=device).float(),
        "ref_element": torch.randn(N_atom,128,device=device),
        "ref_atom_name_chars": torch.randn(N_atom,16,device=device),
        "atom_to_token": torch.randint(0,N_token,(N_atom,),device=device),
        "restype": torch.randn(N_token,32,device=device),
        "profile": torch.randn(N_token,32,device=device),
        "deletion_mean": torch.randn(N_token,device=device),
        "residue_index": torch.arange(N_token, device=device)
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
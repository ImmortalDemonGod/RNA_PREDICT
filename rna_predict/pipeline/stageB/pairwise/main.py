import torch

# 1) For building input embeddings
from rna_predict.pipeline.stageB.pairwise.protenix_integration import (
    ProtenixIntegration,
)

# 2) For running diffusion in Stage D


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
        "ref_element": 2 * torch.rand(N_atom, 128, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "ref_atom_name_chars": 0.1
        * torch.randn(N_atom, 16, device=device),  # Smaller scale for better stability
        "atom_to_token": torch.repeat_interleave(
            torch.arange(N_token, device=device), 4
        ),
        "restype": 2 * torch.rand(N_token, 32, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "profile": 2 * torch.rand(N_token, 32, device=device)
        - 1,  # Uniform distribution in [-1, 1]
        "deletion_mean": torch.randn(N_token, device=device),
        "residue_index": torch.arange(N_token, device=device),
    }

    integrator = ProtenixIntegration(
        c_token=449,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        device=device,
    )

    embeddings = integrator.build_embeddings(input_features)
    s_inputs = embeddings["s_inputs"]
    z_init = embeddings["z_init"]
    print("[Embedding Demo] s_inputs shape:", s_inputs.shape)
    print("[Embedding Demo] z_init shape:", z_init.shape)


def main():
    print("=== Running Protenix Integration Demo (Embeddings) ===")
    demo_run_protenix_embeddings()


if __name__ == "__main__":
    main()

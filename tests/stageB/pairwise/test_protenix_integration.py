import torch

from rna_predict.pipeline.stageB.pairwise.protenix_integration import (
    ProtenixIntegration,
)


def test_residue_index_squeeze_fix():
    """
    Ensures build_embeddings() works when 'residue_index'
    starts as shape [N_token,1]. Should no longer raise a RuntimeError.
    """
    integrator = ProtenixIntegration(device=torch.device("cpu"))
    N_token = 5

    # Minimal set of input features with residue_index => shape (5,1)
    input_features = {
        "residue_index": torch.arange(N_token).unsqueeze(-1),  # [N_token,1]
        "ref_pos": torch.randn(4 * N_token, 3),
        "ref_charge": torch.randn(4 * N_token),
        "ref_element": torch.randn(4 * N_token, 128),
        "ref_atom_name_chars": torch.zeros(4 * N_token, 16),
        "atom_to_token": torch.repeat_interleave(torch.arange(N_token), 4),
        "restype": torch.zeros(N_token, 32),
        "profile": torch.zeros(N_token, 32),
        "deletion_mean": torch.zeros(N_token),
    }

    # Verify no dimension error
    embeddings = integrator.build_embeddings(input_features)

    assert "s_inputs" in embeddings, "Missing single-token embedding"
    assert "z_init" in embeddings, "Missing pair embedding"

    s_inputs = embeddings["s_inputs"]
    z_init = embeddings["z_init"]

    # Confirm shapes
    assert (
        s_inputs.shape[0] == N_token
    ), f"Expected s_inputs shape (N_token, _), got {s_inputs.shape}"
    assert z_init.dim() == 3, f"Expected z_init dimension=3, got {z_init.dim()}"
    assert (
        z_init.shape[0] == N_token and z_init.shape[1] == N_token
    ), f"Expected z_init shape (N_token, N_token, c_z), got {z_init.shape}"

    print(
        "test_residue_index_squeeze_fix passed: no expand() error with (N_token,1) residue_index!"
    )

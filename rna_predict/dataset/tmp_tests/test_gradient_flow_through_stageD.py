import pytest
import torch
from hydra import initialize, compose
from pathlib import Path
from rna_predict.pipeline.stageD.run_stageD import run_stageD
from rna_predict.utils.tensor_utils.types import STANDARD_RNA_ATOMS
from rna_predict.utils.tensor_utils import derive_residue_atom_map

# Config directory for Hydra
CONF_DIR = Path(__file__).resolve().parents[3] / "rna_predict" / "conf"

@pytest.mark.parametrize("batch_size, n_res, c_s", [
    (2, 10, 8),
    (1, 20, 8),
])
def test_gradient_flow_through_stageD(batch_size, n_res, c_s):
    # Set up a minimal config using Hydra (adjust config group as needed)
    with initialize(config_path="../../../rna_predict/conf", version_base=None):
        cfg = compose(config_name="default")
        # Optionally override config values for a minimal test
        cfg.model.stageD.debug_logging = True
        cfg.model.stageD.model_architecture.c_s = c_s
        cfg.model.stageD.model_architecture.c_token = c_s  # if needed for shape match
        cfg.model.stageD.model_architecture.c_s_inputs = c_s  # if needed
        cfg.model.stageD.model_architecture.test_residues_per_batch = n_res
        cfg.device = "cpu"

    # Get correct dims from config
    c_z = cfg.model.stageD.model_architecture.c_z
    c_s_inputs = cfg.model.stageD.model_architecture.c_s_inputs

    # Create dummy sequence and residue-level features
    sequence = ["A"] * n_res
    n_atoms = sum(len(STANDARD_RNA_ATOMS[res]) for res in sequence)
    s_trunk = torch.randn(batch_size, n_res, c_s, requires_grad=True)
    z_trunk = torch.randn(batch_size, n_res, n_res, c_z, requires_grad=True)
    s_inputs = torch.randn(batch_size, n_res, c_s_inputs, requires_grad=True)
    coords = torch.randn(batch_size, n_atoms, 3, requires_grad=True)

    # Derive atom_metadata using real mapping logic
    residue_atom_map = derive_residue_atom_map(sequence, partial_coords=coords)
    residue_indices = []
    for res_idx, atom_idxs in enumerate(residue_atom_map):
        residue_indices.extend([res_idx] * len(atom_idxs))
    atom_metadata = {"residue_indices": torch.tensor(residue_indices, dtype=torch.long)}

    # input_feature_dict can be empty or minimal for this test
    input_feature_dict = {}

    # Instrumentation: print requires_grad before forward
    print(f"[INSTRUMENT] coords.requires_grad before forward: {coords.requires_grad}")

    # Run Stage D pipeline (adjust call signature as needed)
    output = run_stageD(
        cfg,
        coords=coords,
        s_trunk=s_trunk,
        z_trunk=z_trunk,
        s_inputs=s_inputs,
        input_feature_dict=input_feature_dict,
        atom_metadata=atom_metadata,
    )

    # Check output shape (adjust as appropriate)
    assert output is not None
    if isinstance(output, dict) and "coordinates" in output:
        out_tensor = output["coordinates"]
    else:
        out_tensor = output
    assert isinstance(out_tensor, torch.Tensor)
    assert out_tensor.shape[0] == batch_size

    # Instrumentation: print requires_grad after forward
    print(f"[INSTRUMENT] out_tensor.requires_grad after forward: {out_tensor.requires_grad}")

    # Gradient flow check: backward pass
    loss = out_tensor.sum()
    loss.backward()
    # Instrumentation: print coords.grad after backward
    print(f"[INSTRUMENT] coords.grad after backward: {coords.grad}")

    # All input tensors should have gradients
    missing_grads = []
    for t in [s_trunk, z_trunk, s_inputs, coords]:
        print(f"[DEBUG][GRADCHECK] t.shape={t.shape}, t.requires_grad={t.requires_grad}, t.grad is None? {t.grad is None}")
        if t.grad is None:
            missing_grads.append(t)

    # Assert that all tensors have gradients
    assert len(missing_grads) == 0, f"GRADIENT FLOW FAILURE: {len(missing_grads)} input tensors missing gradients"
    print("GRADIENT FLOW CONFIRMED: All input tensors have non-None gradients.")

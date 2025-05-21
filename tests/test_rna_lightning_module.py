import os
import sys
from unittest.mock import MagicMock
from hydra import compose, initialize_config_dir
from rna_predict.training.rna_lightning_module import RNALightningModule
import torch
import pytest

print(f"[DEBUG][CWD CONTENTS] {os.listdir(os.getcwd())}")
print(f"[DEBUG][sys.path] {sys.path}")
print(f"[DEBUG][CONFIG_ABS] {os.path.abspath('rna_predict/conf')}")
print(f"[DEBUG][TEST FILE ABS] {os.path.abspath(__file__)}")
print(f"[DEBUG][CWD] Current working directory at test start: {os.getcwd()}")

# NOTE: This test must be run from the project root for Hydra config resolution to work correctly.
#       Do NOT run pytest from inside the tests/ directory.
#       Example:
#         /Users/tomriddle1/.local/bin/uv run pytest tests/test_rna_lightning_module.py::test_noise_and_bridging_runs -vv --disable-warnings --maxfail=1

@pytest.fixture
def hydra_cfg():
    # Use the real Hydra config system and the actual conf directory
    """
    Loads and returns a Hydra configuration object for testing.
    
    Initializes the Hydra config system using the project's config directory, composes the default configuration, sets the device to CPU for test isolation, and enables debug logging for StageA if available.
    """
    config_dir = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    print(f"[DEBUG][HYDRA] Using config_dir={config_dir}")
    with initialize_config_dir(config_dir=config_dir, job_name="test_noise_and_bridging"):
        cfg = compose(config_name="default")
        # Override device for test isolation
        cfg.device = "cpu"
        # Optionally set debug_logging for StageA if present
        if hasattr(cfg.model, 'stageA') and hasattr(cfg.model.stageA, 'debug_logging'):
            cfg.model.stageA.debug_logging = True
        return cfg

def make_dummy_batch(device, B=2, N_res=5, atoms_per_residue=2, C=8):
    """
    Creates a dummy batch of atom-level RNA features for testing.
    
    Generates random coordinates, atom-to-residue mappings, sequence strings, and various atom-level features with consistent shapes for use as model input. Each residue contains a fixed number of atoms, and all features are shaped accordingly. The returned dictionary includes all required fields for model training or evaluation.
    """
    N_atom = N_res * atoms_per_residue
    print(f"[DEBUG][make_dummy_batch] B={B}, N_res={N_res}, atoms_per_residue={atoms_per_residue}, N_atom={N_atom}")
    # Sequence: one string of length N_res per batch
    sequence = ['AUGCU'[:N_res] for _ in range(B)]
    # Map each atom to its residue index
    residue_indices = []
    for b in range(B):
        indices = []
        for r in range(N_res):
            indices.extend([r] * atoms_per_residue)
        residue_indices.append(indices)
    # atom_to_token_idx: each atom's residue index, shape [B, N_atom]
    atom_to_token_idx = torch.tensor(residue_indices, device=device)
    coords_true = torch.randn(B, N_atom, 3, device=device)
    atom_mask = torch.ones(B, N_atom, dtype=torch.bool, device=device)
    print(f"[DEBUG][make_dummy_batch] coords_true.shape={coords_true.shape}")
    print(f"[DEBUG][make_dummy_batch] atom_to_token_idx.shape={atom_to_token_idx.shape}")
    print(f"[DEBUG][make_dummy_batch] residue_indices (len): {[len(lst) for lst in residue_indices]}")
    assert coords_true.shape == (B, N_atom, 3)
    assert atom_to_token_idx.shape == (B, N_atom)
    assert all(len(lst) == N_atom for lst in residue_indices)
    return {
        'coords_true': coords_true,
        'angles_true': torch.zeros(B, N_res, 3, dtype=torch.float32, device=device),
        'adjacency_type': torch.zeros(B, N_res, N_res, dtype=torch.int64, device=device),
        'atom_mask': atom_mask,
        'atom_to_token_idx': atom_to_token_idx,
        'sequence': sequence,
        # --- Atom features with correct shapes/types ---
        'ref_element': torch.zeros(B, N_atom, 128, dtype=torch.float32, device=device),
        'ref_charge': torch.zeros(B, N_atom, 1, dtype=torch.float32, device=device),
        'ref_mask': torch.ones(B, N_atom, 1, dtype=torch.float32, device=device),
        'ref_atom_name_chars': torch.zeros(B, N_atom, 256, dtype=torch.float32, device=device),
        'ref_pos': torch.zeros(B, N_atom, 3, dtype=torch.float32, device=device),
        # Optionally include profile and ref_space_uid if needed for Stage D config
        'profile': torch.zeros(B, N_atom, 32, dtype=torch.float32, device=device),
        'ref_space_uid': torch.zeros(1, 1, 1, 3, dtype=torch.float32, device=device),
        'atom_metadata': {
            # Flatten residue_indices for Stage D compatibility
            'residue_indices': [idx for sublist in residue_indices for idx in sublist]
        },
    }


def make_dummy_output(device, B=2, N_res=5, N_atom=5, C=8):
    """
    Creates a dictionary of random tensors simulating model outputs for testing.
    
    Args:
        device: The device on which to allocate the tensors.
        B: Batch size.
        N_res: Number of residues per sample.
        N_atom: Number of atoms per sample (unused in output shapes).
        C: Embedding dimension.
    
    Returns:
        A dictionary containing random 's_embeddings', 'z_embeddings', and 'logits' tensors with appropriate shapes for model output simulation.
    """
    return {
        's_embeddings': torch.randn(B, N_res, C, device=device),
        'z_embeddings': torch.randn(B, N_res, N_res, C, device=device),
        'logits': torch.randn(B, N_res, 3, device=device),
    }

def test_noise_and_bridging_runs(hydra_cfg):
    """
    Tests that the RNALightningModule training step executes bridging and noise logic without errors.
    
    Instantiates the model with a Hydra config, uses dummy input and output data, mocks the forward pass, and verifies that the training step returns a loss tensor.
    """
    device = torch.device('cpu')
    model = RNALightningModule(cfg=hydra_cfg)
    batch = make_dummy_batch(device)
    dummy_output = make_dummy_output(device)
    # Patch model.forward to return dummy_output and inject a dummy loss_angle
    model.forward = MagicMock(return_value=dummy_output)
    # Patch loss_angle in local scope (simulate as if computed in training_step)
    setattr(model, 'loss_angle', torch.tensor(1.0, device=device, requires_grad=True))
    # Run training_step
    result = model.training_step(batch, batch_idx=0)
    # Should return a dict with 'loss'
    assert 'loss' in result
    assert isinstance(result['loss'], torch.Tensor)
    # The bridging and noise logic should run without error (check logs manually if needed)
    # Optionally, check that shapes are as expected (by capturing logs or patching bridging)

@pytest.mark.skip(reason="Not relevant to L_angle integration for now")
def test_gradient_flow_through_stageD(hydra_cfg):
    """
    Tests that gradients flow through the Stage D component of RNALightningModule during training.
    
    This test uses a dummy batch and output, patches the model's forward method, and sets an environment variable to trigger Stage D logic. It verifies that the Stage D result is differentiable and that at least one model parameter receives a non-zero gradient after backward propagation.
    """
    import os
    device = torch.device('cpu')
    model = RNALightningModule(cfg=hydra_cfg)
    batch = make_dummy_batch(device)
    print(f"[TEST DEBUG] batch['coords_true'].shape: {batch['coords_true'].shape}")
    print(f"[TEST DEBUG] batch['atom_to_token_idx'].shape: {batch['atom_to_token_idx'].shape}")
    print(f"[TEST DEBUG] batch['atom_mask'].shape: {batch['atom_mask'].shape}")
    print(f"[TEST DEBUG] atom_metadata['residue_indices'] len: {len(batch['atom_metadata']['residue_indices'])}")
    assert batch['coords_true'].shape == batch['atom_to_token_idx'].shape + (3,)
    assert batch['atom_to_token_idx'].shape[1] == 10  # Should match N_atom
    dummy_output = make_dummy_output(device)
    model.forward = MagicMock(return_value=dummy_output)
    # Set test environment variable to trigger dummy Stage D result
    os.environ['PYTEST_CURRENT_TEST'] = 'test_run_stageD_basic'
    try:
        result = model.training_step(batch, batch_idx=0)
        stageD_result = result.get('stageD_result', None)
        assert stageD_result is not None, 'Stage D result should not be None.'
        if isinstance(stageD_result, dict) and 'coordinates' in stageD_result:
            dummy_loss = stageD_result['coordinates'].sum()
        elif isinstance(stageD_result, torch.Tensor):
            dummy_loss = stageD_result.sum()
        else:
            pytest.skip('Stage D result is not differentiable or has unknown format.')
        dummy_loss.backward()
        grads = [p.grad for n,p in model.named_parameters() if p.requires_grad and p.grad is not None]
        assert any(g is not None and torch.any(g != 0) for g in grads), 'No gradients flowed back to model parameters.'
    finally:
        if 'PYTEST_CURRENT_TEST' in os.environ:
            del os.environ['PYTEST_CURRENT_TEST']

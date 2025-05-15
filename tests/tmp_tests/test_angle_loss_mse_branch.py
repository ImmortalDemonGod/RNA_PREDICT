import os
import torch
import pytest
from hydra import initialize_config_dir, compose
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.dataset.preprocessing.angle_utils import angles_rad_to_sin_cos

@pytest.fixture(scope="function")
def cfg():
    # Load default config with StageD enabled
    with initialize_config_dir(config_dir="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", version_base=None, job_name="test_angle_loss_mse_branch"):
        cfg = compose(config_name="default")
    # Ensure StageD is on to test angle-loss branch
    cfg.run_stageD = True
    cfg.model.stageD.debug_logging = False
    return cfg

@pytest.fixture(scope="function")
def dummy_batch(cfg):
    # Build a minimal batch with zero true angles
    L = 3  # residues
    # angles_true shape [B, L, num_angles]
    num_angles = cfg.model.stageB.torsion_bert.num_angles
    batch = {
        'sequence': ['A' * L],
        'adjacency': torch.eye(L),
        'angles_true': torch.zeros((1, L, num_angles)),
        'attention_mask': torch.ones((1, L), dtype=torch.bool),
        # dummy stageD inputs
        'coords_true': torch.zeros((1, L * cfg.atoms_per_residue, 3)),
        'atom_mask': torch.ones((1, L * cfg.atoms_per_residue), dtype=torch.bool),
        'atom_to_token_idx': torch.arange(L, dtype=torch.long).repeat_interleave(cfg.atoms_per_residue).reshape(1, -1),
        'ref_element': list('A' * L),
        'ref_atom_name_chars': list('A' * L),
        'profile': torch.zeros((1, L * cfg.atoms_per_residue, 32)),
        'ref_space_uid': torch.zeros((1,1,1,3)),
        'atom_metadata': {'residue_indices': list(range(L * cfg.atoms_per_residue))}
    }
    return batch

def test_angle_loss_mse_branch(cfg, dummy_batch):
    # Force skip StageD execution after angle loss
    os.environ['PYTEST_CURRENT_TEST'] = 'test_noise_and_bridging_runs'
    device = torch.device(cfg.device)
    model = RNALightningModule(cfg)
    model.to(device)
    # Monkeypatch forward to produce zero predicted angles
    def fake_forward(batch):
        true = batch['angles_true']
        # convert true angles to sin/cos pairs
        true_sincos = angles_rad_to_sin_cos(true)
        # create gradient path through stageB_torsion parameters
        accum = sum(p.sum() for p in model.stageB_torsion.parameters())
        # broadcast accum to match sincos shape
        grad_pad = torch.zeros_like(true_sincos)
        pred_sincos = true_sincos + accum * grad_pad
        return {'torsion_angles': pred_sincos, 's_embeddings': None, 'z_embeddings': None}
    model.forward = fake_forward
    # Run training_step
    result = model.training_step(dummy_batch, batch_idx=0)
    assert 'loss' in result
    loss = result['loss']
    # Loss must be zero scalar
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    # Backward should propagate (grad_fn not None)
    loss.backward()
    grads = [p.grad for p in model.stageB_torsion.parameters()]
    assert any(g is not None for g in grads), "StageB torsion parameters should receive gradient"
    # Clean up
    del os.environ['PYTEST_CURRENT_TEST']

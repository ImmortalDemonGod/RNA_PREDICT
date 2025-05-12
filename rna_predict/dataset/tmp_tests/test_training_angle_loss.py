import pytest
import torch
from hydra import initialize, compose

from rna_predict.training.rna_lightning_module import RNALightningModule

@ pytest.fixture(scope="function")
def cfg():
    # Load Hydra config with Stage D disabled and debug_logging off
    # Use relative config_path per project guidelines
    # Initialize Hydra using config_path relative to this test file
    with initialize(config_path="../../conf", job_name="test_training_angle_loss"):
        cfg = compose(config_name="default", overrides=["run_stageD=false", "model.stageD.debug_logging=false"])
    return cfg

@ pytest.fixture(scope="function")
def dummy_batch(cfg):
    # Simple 4-mer RNA
    seq = "ACGU"
    L = len(seq)
    # Number of predicted torsion angles per residue
    num_angles = cfg.model.stageB.torsion_bert.num_angles
    # Build batch dict
    batch = {
        "sequence": [seq],                           # list of length 1
        "adjacency": torch.eye(L),                   # [L, L]
        "angles_true": torch.zeros((1, L, num_angles)),
        "attention_mask": torch.ones((1, L), dtype=torch.bool),
        # Stage D inputs (not used for angle loss)
        "coords_true": torch.zeros((1, L, 3)),
        "atom_mask": torch.ones((1, L * cfg.atoms_per_residue), dtype=torch.bool),
        "atom_to_token_idx": torch.arange(L, dtype=torch.long).repeat_interleave(cfg.atoms_per_residue).reshape(1, -1),
    }
    return batch

@ pytest.mark.parametrize("seed", [0, 42])
def test_angle_loss_and_gradients(cfg, dummy_batch, seed):
    torch.manual_seed(seed)
    model = RNALightningModule(cfg)
    # Move model to configured device
    model.to(torch.device(cfg.device))

    # Run training_step
    result = model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(result, dict), "training_step should return a dict"
    assert "loss" in result, "Result must contain 'loss'"

    loss = result["loss"]
    # Loss should be a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0, "Loss must be a scalar"
    # For zero true angles and model dummy predictor, MSE should be zero
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    # Backward should work without error
    loss.backward()
    # Check that at least one parameter of stageB_torsion has been assigned a grad (even if zero)
    grads = [p.grad for p in model.stageB_torsion.parameters()]
    assert any(g is not None for g in grads), "StageB TorsionBERT parameters should have gradients"
    # Clear gradients
    model.zero_grad()

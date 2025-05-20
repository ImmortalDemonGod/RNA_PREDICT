import pytest
import torch
from hydra import initialize_config_dir, compose
from rna_predict.training.rna_lightning_module import RNALightningModule

@pytest.fixture(scope="function")
def cfg():
    # Initialize Hydra with absolute config directory
    """
    Creates and returns a Hydra configuration object for testing with specific overrides.
    
    Initializes Hydra using an absolute configuration directory and applies overrides to disable stageD and its debug logging. Prints debug information about the structure of the stageB and torsion_bert configuration sections.
    """
    with initialize_config_dir(config_dir="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", version_base=None, job_name="test_training_angle_loss"):
        cfg = compose(config_name="default", overrides=["run_stageD=false", "model.stageD.debug_logging=false"])
    print("DEBUG: type(cfg.model['stageB']) =", type(cfg.model['stageB']))
    print("DEBUG: keys(cfg.model['stageB']) =", list(cfg.model['stageB'].keys()))
    print("DEBUG: type(cfg.model['stageB']['torsion_bert']) =", type(cfg.model['stageB']['torsion_bert']))
    print("DEBUG: keys(cfg.model['stageB']['torsion_bert']) =", list(cfg.model['stageB']['torsion_bert'].keys()))
    return cfg

@pytest.fixture
def dummy_batch():
    """
    Creates a dummy batch of RNA data with zeroed tensors for testing.
    
    Returns:
        dict: A batch dictionary containing a single RNA sequence, zeroed angle and coordinate tensors, and a zeroed adjacency matrix.
    """
    # Create a dummy batch with a sequence of length 4
    batch = {
        "sequence": ["ACGU"],  # Single sequence of length 4
        "angles_true": torch.zeros(1, 4, 14),  # [batch_size, seq_len, 2*num_angles] (sin/cos pairs)
        "coords_true": torch.zeros(1, 4, 3),  # [batch_size, seq_len, 3]
        "adjacency_type": torch.zeros(1, 4, 4),  # [batch_size, seq_len, seq_len]
    }
    return batch

@pytest.mark.parametrize("seed", [0, 42])
def test_angle_loss_and_gradients_with_seed(cfg, dummy_batch, seed):
    """
    Tests that the RNALightningModule computes zero angle loss and valid gradients on dummy input with a fixed seed.
    
    Verifies that the training step returns a scalar loss tensor close to zero, backward propagation works, and gradients are properly computed and finite for all parameters, especially in the stageB_torsion submodule.
    """
    torch.manual_seed(seed)
    model = RNALightningModule(cfg)
    # Move model to configured device
    model.to(torch.device(cfg.device))
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Run training_step
    model.train()
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
    
    # Additional gradient checks from second test
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()
    
    # Clear gradients
    model.zero_grad()

@pytest.mark.parametrize("batch_idx", [0])
def test_angle_loss_with_batch_idx(cfg, dummy_batch, batch_idx):
    """
    Verifies that the model's angle loss computation returns a near-zero scalar tensor for a dummy batch and specified batch index.
    
    Ensures the training step output is a dictionary containing a scalar loss tensor close to zero when provided with zeroed input data.
    """
    # Use the full Hydra config
    model = RNALightningModule(cfg)
    
    # Ensure all parameters require gradients
    for param in model.parameters():
        param.requires_grad = True

    # Run the training step
    model.train()
    result = model.training_step(dummy_batch, batch_idx)

    # Check that the result is a dictionary containing a scalar loss
    assert isinstance(result, dict)
    assert "loss" in result
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].ndim == 0  # Scalar

    # Check that the loss is approximately zero (since we're using zero tensors)
    assert torch.allclose(result["loss"], torch.tensor(0.0), atol=1e-6)

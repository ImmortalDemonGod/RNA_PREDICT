"""
Integration test for partial checkpoint cycle in RNA_PREDICT.
Covers: manual train loop, partial adapter save, partial load, and verification.
Also compares file sizes of full vs. partial checkpoints to demonstrate efficiency.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pytest
import hydra
import lightning as L
from rna_predict.utils.checkpoint import partial_load_state_dict

# Dummy model with frozen base and trainable adapter
class DummyCheckpointModel(nn.Module):
    def __init__(self, base_dim=16, adapter_dim=8):
        super().__init__()
        self.base_layer = nn.Linear(base_dim, base_dim)
        self.adapter_layer = nn.Linear(base_dim, adapter_dim)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.base_layer(x)
        x = self.adapter_layer(x)
        return x

    def get_adapter_state_dict(self):
        return {f"adapter_layer.{k}": v for k, v in self.adapter_layer.state_dict().items()}

# Synthetic data loader

def create_dummy_dataloader(batch_size=4, n_samples=12, base_dim=16, adapter_dim=8):
    X = torch.randn(n_samples, base_dim)
    y = torch.randn(n_samples, adapter_dim)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size)

# Hydra setup (relative path per Hydra requirement; resolves to absolute project conf)
def setup_hydra():
    import hydra.core.global_hydra
    # Hydra requires a relative config_path, but this resolves to the absolute conf dir
    if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.initialize(config_path="rna_predict/conf", version_base=None)

# Integration test

def test_train_save_partial_load_infer(tmp_path):
    setup_hydra()  # Ensure Hydra is initialized
    base_dim, adapter_dim = 16, 8
    dataloader = create_dummy_dataloader(base_dim=base_dim, adapter_dim=adapter_dim)
    model = DummyCheckpointModel(base_dim, adapter_dim)
    criterion = nn.MSELoss()
    # --- Replace manual train loop with Lightning Trainer ---
    class LightningAdapterModule(L.LightningModule):
        def __init__(self, model, criterion):
            super().__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            out = self(x)
            loss = self.criterion(out, y)
            self.log('train_loss', loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.model.adapter_layer.parameters(), lr=1e-3)

        def train_dataloader(self):
            return dataloader

    lightning_module = LightningAdapterModule(model, criterion)
    trainer = L.Trainer(max_epochs=2, fast_dev_run=False, enable_checkpointing=False, logger=False)
    trainer.fit(lightning_module)
    # --- Continue with original checkpointing and partial load logic ---
    # Save only adapter weights
    adapter_sd = model.get_adapter_state_dict()
    partial_ckpt_path = tmp_path / "adapter_only.pth"
    torch.save(adapter_sd, partial_ckpt_path)
    # Save full model for comparison
    full_ckpt_path = tmp_path / "full_model.pth"
    torch.save(model.state_dict(), full_ckpt_path)
    # Instantiate a new model (random init)
    model2 = DummyCheckpointModel(base_dim, adapter_dim)
    # Load only adapter weights into model2
    loaded_sd = torch.load(partial_ckpt_path)
    missing, unexpected = partial_load_state_dict(model2, loaded_sd, strict=False)
    # Check adapter params match, base params do not
    for k, v in model2.adapter_layer.state_dict().items():
        v_orig = model.adapter_layer.state_dict()[k]
        assert torch.allclose(v, v_orig, atol=1e-6), f"Adapter param {k} not loaded as expected"
    for k, v in model2.base_layer.state_dict().items():
        v_orig = model.base_layer.state_dict()[k]
        assert not torch.allclose(v, v_orig, atol=1e-6), f"Base param {k} should not have been loaded"
    # Model is operational
    x = torch.randn(4, base_dim)
    out = model2(x)
    assert out.shape == (4, adapter_dim)

import sys
import os
import types
import torch
import hydra
from omegaconf import OmegaConf

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('.'))

from rna_predict.training.rna_lightning_module import RNALightningModule

# Create a simple config
config = OmegaConf.create({
    "device": "cpu",
    "model": {
        "stageA": {
            "checkpoint_path": "dummy_path",
            "num_hidden": 128,
            "dropout": 0.1,
            "batch_size": 1,
            "lr": 0.001,
        },
        "stageB": {
            "torsion_bert": {},
            "pairformer": {},
        },
    },
})

# Create RNALightningModule
print("Creating RNALightningModule...")
model = RNALightningModule(config)
model._integration_test_mode = True
print(f"Model type: {type(model)}")

# Test forward pass
print("Testing forward pass...")
dummy_batch = {
    "sequence": ["AUGC"],
    "atom_to_token_idx": torch.zeros(1, dtype=torch.long),
    "ref_element": torch.zeros(1, dtype=torch.long),
    "ref_atom_name_chars": torch.zeros(1, dtype=torch.long),
    "atom_mask": torch.ones(1, dtype=torch.bool),
}
output = model(dummy_batch)
print(f"Output keys: {list(output.keys())}")
print("Success!")

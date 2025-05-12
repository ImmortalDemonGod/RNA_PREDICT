import pytest
import torch
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from rna_predict.dataset.loader import RNADataset
import rna_predict.dataset.preprocessing.angles as angles_module

@ pytest.fixture(scope="function")
def cfg(tmp_path):
    # Minimal Hydra config with necessary data and global extraction_backend
    cfg_dict = {
        "device": "cpu",
        "data": {
            "index_csv": str(tmp_path / "index.csv"),
            "max_residues": 2,
            "max_atoms": 2,
            "angle_backend": "mdanalysis"
        },
        "extraction_backend": "dssr"
    }
    cfg = OmegaConf.create(cfg_dict)
    # Create a minimal index CSV for RNADataset
    df = pd.DataFrame([
        {"id": "x", "sequence_path": "", "target_id": "", "pdb_path": "nonexistent.pdb"}
    ])
    df.to_csv(cfg.data.index_csv, index=False)
    return cfg


def test_load_angles_uses_extraction_backend(monkeypatch, cfg):
    # Monkeypatch extract_rna_torsions to capture the backend used
    called = {}
    def fake_extract(path, chain_id, backend, angle_set, **kwargs):
        called['backend'] = backend
        return torch.zeros((cfg.data.max_residues, 7), dtype=torch.float32)
    monkeypatch.setattr(angles_module, "extract_rna_torsions", fake_extract)

    # Instantiate dataset with angle loading enabled
    ds = RNADataset(cfg.data.index_csv, cfg, load_ang=True)
    # Access first item to trigger _load_angles
    sample = ds[0]
    assert called.get('backend') == cfg.extraction_backend, \
        f"Expected backend '{cfg.extraction_backend}', got '{called.get('backend')}'"

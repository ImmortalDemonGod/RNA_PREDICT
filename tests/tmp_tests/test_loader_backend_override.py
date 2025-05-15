import pytest
import torch
import pandas as pd
from omegaconf import OmegaConf
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
            "batch_size": 1,
            "coord_dtype": "float32",
            "coord_fill_value": "nan",
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
    def fake_extract(*args, **kwargs):
        # Capture backend used in extraction
        called['backend'] = kwargs.get('backend')
        return torch.zeros((cfg.data.max_residues, 7), dtype=torch.float32)
    monkeypatch.setattr(angles_module, "extract_rna_torsions", fake_extract)
    # Stub atom parsing to skip file I/O
    import rna_predict.dataset.loader as loader_module
    monkeypatch.setattr(loader_module, "parse_pdb_atoms", lambda pdb_file: {})

    # Instantiate dataset with angle loading enabled
    ds = RNADataset(cfg.data.index_csv, cfg, load_ang=True)
    # Access first item to trigger _load_angles
    ds[0]
    assert called.get('backend') == cfg.extraction_backend, \
        f"Expected backend '{cfg.extraction_backend}', got '{called.get('backend')}'"

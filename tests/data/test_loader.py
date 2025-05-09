import os
import tempfile
import shutil
import pandas as pd
import pytest
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from hydra import compose, initialize

from rna_predict.dataset.loader import RNADataset
from rna_predict.dataset.collate import rna_collate_fn

# --- Fixtures ---
@pytest.fixture(scope="module")
def temp_index_csv():
    tmpdir = tempfile.mkdtemp()
    csv_path = os.path.join(tmpdir, "index.csv")
    # Minimal CSV with required columns
    df = pd.DataFrame([
        {
            "id": "sample1",
            "sequence_path": "dummy_seq.fasta",
            "target_id": "sample1",
            "pdb_path": "",
        },
        {
            "id": "sample2",
            "sequence_path": "dummy_seq.fasta",
            "target_id": "sample2",
            "pdb_path": "",
        },
    ])
    df.to_csv(csv_path, index=False)
    yield csv_path
    shutil.rmtree(tmpdir)

@pytest.fixture(scope="module")
def dummy_cfg():
    # Hydra config_path must be RELATIVE TO THIS TEST FILE LOCATION
    print(f"[DEBUG] CWD for Hydra: {os.getcwd()}")
    with initialize(config_path="../../rna_predict/conf", version_base=None):
        cfg = compose(config_name="default.yaml")
        print("[DEBUG] Loaded Hydra config for test_loader.py:\n", OmegaConf.to_yaml(cfg))
        # Optionally override values for test (if needed)
        cfg.data.max_residues = 16
        cfg.data.max_atoms = 8
        cfg.data.batch_size = 2
        cfg.data.coord_dtype = "float32"
        cfg.data.coord_fill_value = "nan"
        return cfg

# --- Tests ---
def test_rnadataset_getitem(temp_index_csv, dummy_cfg):
    ds = RNADataset(temp_index_csv, dummy_cfg)
    sample = ds[0]
    # Required keys
    assert "sequence_id" in sample
    assert "sequence" in sample
    assert "coords_true" in sample
    assert "attention_mask" in sample
    assert "atom_mask" in sample
    assert "atom_to_token_idx" in sample
    assert "ref_element" in sample
    assert "atom_names" in sample
    assert "ref_atom_name_chars" in sample
    assert "residue_indices" in sample
    # Shapes
    seq_len = len(sample["sequence"])
    assert sample["coords_true"].shape[0] == seq_len
    assert sample["coords_true"].shape[-1] == 3
    assert sample["atom_mask"].shape[0] == seq_len
    assert sample["atom_mask"].shape[1] == dummy_cfg.data.max_atoms
    assert sample["atom_to_token_idx"].shape[0] == seq_len
    assert sample["atom_to_token_idx"].shape[1] == dummy_cfg.data.max_atoms
    assert seq_len <= dummy_cfg.data.max_residues

def test_rnadataset_len(temp_index_csv, dummy_cfg):
    ds = RNADataset(temp_index_csv, dummy_cfg)
    assert len(ds) == 2

def test_rna_collate_fn_basic(temp_index_csv, dummy_cfg):
    ds = RNADataset(temp_index_csv, dummy_cfg)
    loader = DataLoader(ds, batch_size=2, collate_fn=rna_collate_fn)
    batch = next(iter(loader))
    # Collated batch should have batched tensor shapes
    assert batch["coords_true"].shape[0] == 2
    # For fallback, each sample's length is 10; batch shape is [2, 10, 3]
    assert batch["coords_true"].shape[1] == len(batch["sequence"][0])
    assert batch["coords_true"].shape[-1] == 3
    assert len(batch["sequence"][0]) <= dummy_cfg.data.max_residues
    # Check batch keys
    assert "sequence_id" in batch
    assert "sequence" in batch
    assert "coords_true" in batch
    assert "attention_mask" in batch
    assert "atom_mask" in batch
    assert "atom_to_token_idx" in batch
    assert "ref_element" in batch
    assert "atom_names" in batch
    assert "ref_atom_name_chars" in batch
    assert "residue_indices" in batch

def test_rna_collate_fn_single_item(temp_index_csv, dummy_cfg):
    ds = RNADataset(temp_index_csv, dummy_cfg)
    loader = DataLoader(ds, batch_size=1, collate_fn=rna_collate_fn)
    batch = next(iter(loader))
    assert batch["coords_true"].shape[0] == 1
    assert batch["coords_true"].shape[1] == len(batch["sequence"][0])
    assert batch["coords_true"].shape[-1] == 3
    assert len(batch["sequence"][0]) <= dummy_cfg.data.max_residues
    # Keys present
    assert "sequence_id" in batch
    assert "sequence" in batch
    assert "coords_true" in batch
    assert "attention_mask" in batch
    assert "atom_mask" in batch
    assert "atom_to_token_idx" in batch
    assert "ref_element" in batch
    assert "atom_names" in batch
    assert "ref_atom_name_chars" in batch
    assert "residue_indices" in batch

def test_rna_collate_fn_empty_batch_raises():
    # Directly test collate_fn with empty batch
    with pytest.raises(ValueError):
        rna_collate_fn([])

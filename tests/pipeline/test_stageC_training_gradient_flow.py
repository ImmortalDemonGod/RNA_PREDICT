import torch
import pytest
from hydra import compose, initialize
from rna_predict.conf.config_schema import register_configs
from rna_predict.dataset.loader import RNADataset
from rna_predict.dataset.collate import rna_collate_fn
from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC_rna_mpnerf

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required for this test")
def test_stageC_training_gradient_flow_repro():
    # Register configs as in train.py
    register_configs()
    # Compose config as Hydra would do in training
    with initialize(version_base=None, config_path="../../rna_predict/conf"):
        cfg = compose(config_name="default.yaml", overrides=["device=mps", "data.index_csv=rna_predict/dataset/examples/kaggle_minimal_index.csv", "model.stageC.device=mps"])
    # Load a real batch from the dataset
    dataset = RNADataset(index_csv=cfg.data.index_csv, cfg=cfg, load_adj=False, load_ang=False, verbose=False)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=rna_collate_fn, shuffle=False)
    batch = next(iter(dl))
    # Use the sequence and dummy torsions (simulate Stage B output)
    sequence = batch["sequence"][0]
    n_res = len(sequence)
    n_torsions = 7
    torsions = torch.randn(n_res, n_torsions, requires_grad=True, device="mps")
    # Run Stage C as in training
    output = run_stageC_rna_mpnerf(cfg, sequence, torsions)
    coords = output["coords"] if isinstance(output, dict) else output
    print("[DEBUG-TRAIN-REPRO] torsions.requires_grad:", torsions.requires_grad)
    print("[DEBUG-TRAIN-REPRO] torsions.grad_fn:", torsions.grad_fn)
    print("[DEBUG-TRAIN-REPRO] coords.requires_grad:", getattr(coords, 'requires_grad', None))
    print("[DEBUG-TRAIN-REPRO] coords.grad_fn:", getattr(coords, 'grad_fn', None))
    loss = coords.sum()
    loss.backward()
    print("[DEBUG-TRAIN-REPRO] torsions.grad:", torsions.grad)
    assert coords.requires_grad, "Stage C output coords must be differentiable!"
    assert coords.grad_fn is not None, "Stage C output coords must have a grad_fn!"
    assert torsions.grad is not None, "No gradient flowed to input torsions!"
    assert torch.any(torsions.grad != 0), "Gradient is zero everywhere!"
    print("[TEST] Stage C training gradient flow reproduction: PASSED")

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS device required for this test")
def test_stageB_to_stageC_gradient_flow_repro():
    # Register configs as in train.py
    register_configs()
    # Compose config as Hydra would do in training
    with initialize(version_base=None, config_path="../../rna_predict/conf"):
        cfg = compose(config_name="default.yaml", overrides=["device=mps", "data.index_csv=rna_predict/dataset/examples/kaggle_minimal_index.csv", "model.stageC.device=mps"])
    # Load a real batch from the dataset
    dataset = RNADataset(index_csv=cfg.data.index_csv, cfg=cfg, load_adj=False, load_ang=False, verbose=False)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=rna_collate_fn, shuffle=False)
    batch = next(iter(dl))
    sequence = batch["sequence"][0]
    # Instantiate Stage B as in the pipeline
    stageB = StageBTorsionBertPredictor(cfg.model.stageB.torsion_bert)
    # Forward pass through Stage B
    outB = stageB(sequence)
    torsion_angles = outB["torsion_angles"]
    print("[DEBUG-TEST] torsion_angles.requires_grad:", getattr(torsion_angles, 'requires_grad', None))
    print("[DEBUG-TEST] torsion_angles.grad_fn:", getattr(torsion_angles, 'grad_fn', None))
    print("[DEBUG-TEST] torsion_angles.is_leaf:", getattr(torsion_angles, 'is_leaf', None))
    torsion_angles.retain_grad()
    # Forward pass through Stage C
    output = run_stageC_rna_mpnerf(cfg, sequence, torsion_angles)
    coords = output["coords"] if isinstance(output, dict) else output
    print("[DEBUG-TEST] coords.requires_grad:", getattr(coords, 'requires_grad', None))
    print("[DEBUG-TEST] coords.grad_fn:", getattr(coords, 'grad_fn', None))
    loss = coords.sum()
    loss.backward()
    print("[DEBUG-TEST] torsion_angles.grad:", getattr(torsion_angles, 'grad', None))
    assert coords.requires_grad, "Stage C output coords must be differentiable!"
    assert coords.grad_fn is not None, "Stage C output coords must have a grad_fn!"
    assert torsion_angles.grad is not None, "No gradient flowed to input torsion_angles!"
    assert torch.any(torsion_angles.grad != 0), "Gradient is zero everywhere!"
    print("[TEST] Stage B to C training gradient flow reproduction: PASSED")

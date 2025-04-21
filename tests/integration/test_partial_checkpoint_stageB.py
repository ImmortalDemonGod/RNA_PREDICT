"""
Integration test for partial checkpointing on Stage B (PairformerStack) in RNA_PREDICT.
- Only saves and loads the state dict of a single block (simulating adapter/LoRA scenario).
- Verifies that only the targeted block's weights are loaded, others remain at init.
- Confirms model is operational after partial load.
"""
import torch
import torch.nn as nn
from rna_predict.pipeline.stageB.pairwise.pairformer import PairformerStack, PairformerStackConfig
from rna_predict.utils.checkpoint import partial_load_state_dict

def minimal_pairformer_config():
    # Minimal config for a tiny PairformerStack
    class DummyCfg:
        n_blocks = 2
        n_heads = 2
        c_z = 4
        c_s = 0
        dropout = 0.1
        blocks_per_ckpt = 1
    # Fill in defaults for PairformerBlockConfig fields
    for k, v in PairformerStackConfig.__dataclass_fields__.items():
        if not hasattr(DummyCfg, k):
            setattr(DummyCfg, k, v.default)
    return DummyCfg()

def test_partial_checkpoint_stageB(tmp_path):
    cfg = minimal_pairformer_config()
    model = PairformerStack(cfg)
    # Step 1: Randomize all parameters in both blocks
    for block in model.blocks:
        for p in block.parameters():
            nn.init.normal_(p, mean=0.0, std=1.0)
    # Step 2: Overwrite only the first block's params to a constant
    for p in model.blocks[0].parameters():
        nn.init.constant_(p, 42.0)
    # Save only the first block's state dict
    adapter_sd = {f"blocks.0.{k}": v for k, v in model.blocks[0].state_dict().items()}
    partial_ckpt_path = tmp_path / "block0_only.pth"
    torch.save(adapter_sd, partial_ckpt_path)
    # Save full model for comparison
    full_ckpt_path = tmp_path / "full_model.pth"
    torch.save(model.state_dict(), full_ckpt_path)
    # Instantiate a new model (random init)
    model2 = PairformerStack(cfg)
    # Step 3: Load only block0 weights into model2
    loaded_sd = torch.load(partial_ckpt_path)
    missing, unexpected = partial_load_state_dict(model2, loaded_sd, strict=False)
    # Step 4: Assert block0 params match constant, block1 params do not match original
    for k, v in model2.blocks[0].state_dict().items():
        assert torch.allclose(v, torch.full_like(v, 42.0)), f"Block0 param {k} not loaded as constant"
    for k, v in model2.blocks[1].state_dict().items():
        v_orig = model.blocks[1].state_dict()[k]
        # Since both are random, they should almost never match
        assert not torch.allclose(v, v_orig, atol=1e-6), f"Block1 param {k} should not have been loaded"
    # Step 5: Model is operational
    z = torch.randn(2, 2, 2, cfg.c_z)
    s = None if cfg.c_s == 0 else torch.randn(2, 2, cfg.c_s)
    pair_mask = torch.ones(2, 2, dtype=torch.bool)
    out = model2(s, z, pair_mask)
    assert isinstance(out, tuple) and len(out) == 2
    # Step 6: Compare file sizes
    partial_size = partial_ckpt_path.stat().st_size
    full_size = full_ckpt_path.stat().st_size
    print(f"Partial block0 checkpoint size: {partial_size} bytes; Full model size: {full_size} bytes")
    assert partial_size < full_size, "Partial checkpoint should be smaller than full checkpoint"

import torch
from omegaconf import OmegaConf
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC_rna_mpnerf

def test_stageC_gradient_flow():
    # Minimal dummy RNA sequence and torsions
    sequence = "ACG"
    n_res = len(sequence)
    n_torsions = 7
    # Dummy torsion angles (requires_grad=True)
    torsions = torch.randn(n_res, n_torsions, requires_grad=True)
    # Minimal valid config for StageCConfig dataclass, nested under model.stageC per validator
    cfg = OmegaConf.create({
        "model": {
            "stageC": {
                "enabled": True,
                "method": "mp_nerf",
                "do_ring_closure": False,
                "place_bases": True,
                "sugar_pucker": "C3'-endo",
                "device": "cpu",
                "angle_representation": "cartesian",
                "use_metadata": False,
                "use_memory_efficient_kernel": False,
                "use_deepspeed_evo_attention": False,
                "use_lma": False,
                "inplace_safe": False,
                "chunk_size": None,
                "debug_logging": False,
            }
        }
    })
    print("[DEBUG-TEST] torsions.requires_grad:", torsions.requires_grad)
    print("[DEBUG-TEST] torsions.grad_fn:", torsions.grad_fn)
    # Run Stage C
    output = run_stageC_rna_mpnerf(cfg, sequence, torsions)
    coords = output["coords"] if isinstance(output, dict) else output
    print("[DEBUG-TEST] coords.requires_grad:", getattr(coords, 'requires_grad', None))
    print("[DEBUG-TEST] coords.grad_fn:", getattr(coords, 'grad_fn', None))
    # Compute a simple loss
    loss = coords.sum()
    loss.backward()
    print("[DEBUG-TEST] torsions.grad:", torsions.grad)
    # Assert gradient flows to input
    assert torsions.grad is not None, "No gradient flowed to input torsions!"
    assert torch.any(torsions.grad != 0), "Gradient is zero everywhere!"
    print("[TEST] Stage C gradient flow: PASSED")

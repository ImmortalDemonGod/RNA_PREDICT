import torch
from omegaconf import OmegaConf
import pytest

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor

def make_lora_cfg(enabled=True, r=4, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    # Use dict, not SimpleNamespace, for OmegaConf compatibility
    return {
        'enabled': enabled,
        'r': r,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'target_modules': target_modules or ["query", "value"],
        'bias': "none",
        'modules_to_save': None
    }


def make_cfg(lora_enabled=True):
    # Minimal config for TorsionBERT + LoRA
    return OmegaConf.create({
        'model_name_or_path': 'sayby/rna_torsionbert',
        'device': 'cpu',
        'angle_mode': 'sin_cos',
        'num_angles': 7,
        'max_length': 32,
        'lora': make_lora_cfg(enabled=lora_enabled),
        'init_from_scratch': True,  # Use dummy model for test
    })


def test_lora_wrapping_and_freezing():
    cfg = make_cfg(lora_enabled=True)
    predictor = StageBTorsionBertPredictor(cfg)
    # For the dummy model, LoRA cannot be applied (no matching modules)
    assert not predictor.lora_applied, (
        "LoRA should NOT be applied to the dummy model, as it has no target_modules. "
        "This is expected and correct for scientific robustness."
    )
    trainable_params = predictor.get_trainable_parameters()
    all_params = list(predictor.model.named_parameters())
    # All params should be trainable in this fallback case
    assert all(p.requires_grad for _, p in all_params), "All params should be trainable if LoRA not applied (dummy model)."
    assert len(trainable_params) == len(all_params), "Optimizer should return all params if LoRA not applied (dummy model)."


def test_no_lora_if_disabled():
    cfg = make_cfg(lora_enabled=False)
    predictor = StageBTorsionBertPredictor(cfg)
    assert not predictor.lora_applied, "LoRA should not be applied when disabled."
    trainable_params = predictor.get_trainable_parameters()
    all_params = list(predictor.model.parameters())
    # All params should be trainable if no LoRA
    assert all(p.requires_grad for p in all_params), "All params should be trainable if LoRA not applied."
    assert len(trainable_params) == len(all_params), "Optimizer should return all params if LoRA not applied."


def test_predictor_forward_dummy():
    cfg = make_cfg(lora_enabled=True)
    predictor = StageBTorsionBertPredictor(cfg)
    seq = "ACGUACGU"
    out = predictor.predict_angles_from_sequence(seq)
    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == len(seq)
    assert out.shape[1] == 14  # num_angles * 2 for sin_cos


def test_lora_real_model():
    """
    Integration test: LoRA should be applied to a real HuggingFace TorsionBERT model.
    Skips if PEFT/transformers not installed or model can't be loaded.
    """
    try:
        from peft import get_peft_model
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        pytest.skip("PEFT or transformers not installed")
    # Use a real model name; must be accessible
    model_name = "sayby/rna_torsionbert"
    cfg = make_cfg(lora_enabled=True)
    cfg.model_name_or_path = model_name
    cfg.init_from_scratch = False
    try:
        predictor = StageBTorsionBertPredictor(cfg)
    except Exception as e:
        pytest.skip(f"Could not load real HuggingFace model: {e}")
    assert predictor.lora_applied, "LoRA should be applied to real HuggingFace model with matching target_modules."
    trainable_params = predictor.get_trainable_parameters()
    all_params = list(predictor.model.named_parameters())
    # All non-LoRA params should be frozen
    non_lora_trainable = [n for n, p in all_params if p.requires_grad and ("lora" not in n and "adapter" not in n)]
    assert len(non_lora_trainable) == 0, f"Non-LoRA params should be frozen, but found: {non_lora_trainable}"
    # There should be some LoRA params trainable
    lora_trainable = [n for n, p in all_params if p.requires_grad and ("lora" in n or "adapter" in n)]
    assert len(lora_trainable) > 0, "No LoRA parameters are trainable."
    # The optimizer filter should only return LoRA params
    assert len(trainable_params) == len(lora_trainable), "Optimizer returned non-LoRA params."

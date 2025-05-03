from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from omegaconf import OmegaConf

cfg = OmegaConf.create({
    'model_name_or_path': 'sayby/rna_torsionbert',
    'device': 'cpu',
    'angle_mode': 'sin_cos',
    'num_angles': 7,
    'max_length': 32,
    'lora': {
        'enabled': True,
        'r': 4,
        'lora_alpha': 16,
        'lora_dropout': 0.1,
        'target_modules': ['query', 'value'],
        'bias': 'none',
        'modules_to_save': None
    },
    'init_from_scratch': False
})

predictor = StageBTorsionBertPredictor(cfg)
lora_params = sum(p.numel() for n, p in predictor.model.named_parameters() if p.requires_grad and ('lora' in n or 'adapter' in n))
total_params = sum(p.numel() for p in predictor.model.parameters())
print(f'LoRA trainable params: {lora_params}')
print(f'Total params: {total_params}')
print(f'LoRA %: {lora_params/total_params*100:.4f}%')

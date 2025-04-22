import lightning as L
from omegaconf import OmegaConf
from rna_predict.training.rna_lightning_module import RNALightningModule

# Use a minimal but real config for the full pipeline
cfg = OmegaConf.create({
    'device': 'cpu',
    'model': {
        'stageA': {
            'checkpoint_path': 'dummy_path',
            'device': 'cpu',
            'min_seq_length': 8,  # Added required key for StageARFoldPredictor
            'num_hidden': 64,     # Added required key for StageARFoldPredictor
            'dropout': 0.1,       # Added required key for StageARFoldPredictor
            'batch_size': 1,      # Added required key for StageARFoldPredictor
            'lr': 1e-3,           # Added required key for StageARFoldPredictor
            'model': {
                'conv_channels': [32, 64],
                'residual': True,
                'c_in': 32,
                'c_out': 32,
                'c_hid': 32,
                'seq2map': {
                    'input_dim': 4,
                    'num_hidden': 16,
                    'dropout': 0.1,
                    'query_key_dim': 8,
                    'expansion_factor': 2.0,
                    'heads': 2,
                    'attention_heads': 2,
                    'attention_dropout': 0.1,
                    'attention_query_key_dim': 8,
                    'attention_expansion_factor': 2.0,
                    'max_length': 3000,
                    'positional_encoding': True,
                    'use_positional_encoding': True,
                    'use_attention': True
                },
                'decoder': {
                    'up_conv_channels': [64, 32],
                    'skip_connections': True
                }
            }
        },
        'stageB': {
            'torsion_bert': {
                'model_name_or_path': 'bert-base-uncased',
                'device': 'cpu',
                'angle_mode': 'degrees',
                'num_angles': 16
            },
            'pairformer': {
                'model_name_or_path': 'dummy_path',
                'device': 'cpu',
                'stageB_pairformer': {
                    'c_z': 32,
                    'c_s': 0,
                    'dropout': 0.1,
                    'n_blocks': 1,
                    'n_heads': 2,
                    'enable': True,
                    'c_m': 32,
                    'c': 32,
                    'c_s_inputs': 0,
                    'blocks_per_ckpt': 1,
                    'input_feature_dims': {},
                    'strategy': 'default',
                    'train_cutoff': 0,
                    'test_cutoff': 0,
                    'train_lowerb': 0,
                    'test_lowerb': 0,
                    'use_checkpoint': False
                }
            }
        },
        'stageC': {},
        'stageD': {
            # Workaround: nest 'stageD' group inside model.stageD for Stage D pipeline
            'stageD': {
                'diffusion': {
                    'device': 'cpu',
                    'debug_logging': False,
                    'inference': {
                        'num_steps': 2,
                        'temperature': 1.0
                    },
                    'sigma_data': 16.0,
                    'c_atom': 128,
                    'c_atompair': 16,
                    'c_token': 768,
                    'c_s': 384,
                    'c_z': 128,
                    'c_s_inputs': 449,
                    'c_noise_embedding': 256,
                    'atom_encoder': {'n_blocks': 1, 'n_heads': 1},
                    'transformer': {'n_blocks': 1, 'n_heads': 1},
                    'atom_decoder': {'n_blocks': 1, 'n_heads': 1},
                    'model_architecture': {
                        'c_token': 2,
                        'c_s': 2,
                        'c_z': 2,
                        'c_s_inputs': 2,
                        'c_atom': 2,
                        'c_noise_embedding': 2,
                        'num_layers': 1,
                        'num_heads': 1,
                        'dropout': 0.0,
                        'coord_eps': 1e-6,
                        'coord_min': -10000.0,
                        'coord_max': 10000.0,
                        'coord_similarity_rtol': 0.001,
                        'test_residues_per_batch': 1
                    }
                }
            }
        }
    }
})

def test_trainer_fast_dev_run():
    """
    Integration test for RNALightningModule with real pipeline stages and fast_dev_run.
    Asserts unique error if pipeline construction fails.
    """
    # Ensure stageD group exists at the top level for pipeline construction
    if 'stageD' not in cfg.model:
        raise AssertionError('[UNIQUE-ERR-TEST-STAGED-GROUP] stageD group missing from config')
    if 'diffusion' not in cfg.model['stageD']['stageD']:
        raise AssertionError('[UNIQUE-ERR-TEST-STAGED-DIFFUSION] stageD.diffusion group missing from config')

    try:
        # Revert: Pass full cfg, not cfg.model
        model = RNALightningModule(cfg)
    except Exception as e:
        # Unique error for pipeline construction failure
        raise RuntimeError(f"[UNIQUE-ERR-PIPELINE-CONSTRUCT] Pipeline failed to construct: {e}")

    trainer = L.Trainer(fast_dev_run=True, enable_progress_bar=False, logger=False)
    try:
        trainer.fit(model)
    except Exception as e:
        raise RuntimeError(f"[UNIQUE-ERR-TRAINER] Trainer failed: {e}")

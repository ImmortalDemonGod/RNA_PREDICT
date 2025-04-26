import lightning as L
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf
from rna_predict.training.rna_lightning_module import RNALightningModule

# Patch the transformers classes before they are imported
patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock()).start()
patch('transformers.AutoModel.from_pretrained', return_value=MagicMock()).start()

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
                'model_name_or_path': 'dummy_path',  # Changed to dummy_path to avoid loading real model
                'device': 'cpu',
                'angle_mode': 'degrees',
                'num_angles': 7  # Changed to 7 to match expected output
            },
            'pairformer': {
                'model_name_or_path': 'dummy_path',
                'device': 'cpu',
                'stageB_pairformer': {
                    'c_z': 32,
                    'c_s': 64,  # Changed from 0 to 64 to match expected output
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
            # Direct diffusion config without extra nesting
            'diffusion': {
                'device': 'cpu',
                'debug_logging': False,
                'inference': {
                    'num_steps': 2,
                    'temperature': 1.0
                },
                'atom_encoder': {'n_blocks': 1, 'n_heads': 1, 'n_queries': 8, 'n_keys': 8},
                'transformer': {'n_blocks': 1, 'n_heads': 1},
                'atom_decoder': {'n_blocks': 1, 'n_heads': 1, 'n_queries': 8, 'n_keys': 8},
                'model_architecture': {
                    'c_token': 128,
                    'c_s': 384,
                    'c_z': 128,
                    'c_s_inputs': 449,
                    'c_atom': 128,
                    'c_atompair': 16,
                    'c_noise_embedding': 256,
                    'sigma_data': 16.0,
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
})

def test_trainer_fast_dev_run():
    """
    Integration test for RNALightningModule with real pipeline stages and fast_dev_run.
    Asserts unique error if pipeline construction fails.
    """
    # Ensure stageD group exists at the top level for pipeline construction
    if 'stageD' not in cfg.model:
        raise AssertionError('[UNIQUE-ERR-TEST-STAGED-GROUP] stageD group missing from config')
    if 'diffusion' not in cfg.model['stageD']:
        raise AssertionError('[UNIQUE-ERR-TEST-STAGED-DIFFUSION] stageD.diffusion group missing from config')

    # Create mock objects
    torsion_bert_mock = MagicMock()
    torsion_bert_mock.return_value = {"torsion_angles": torch.ones((4, 7))}
    torsion_bert_mock.to.return_value = torsion_bert_mock

    pairformer_mock = MagicMock()
    pairformer_mock.return_value = (torch.ones((4, 64)), torch.ones((4, 4, 32)))
    pairformer_mock.to.return_value = pairformer_mock
    pairformer_mock.predict.return_value = (torch.ones((4, 64)), torch.ones((4, 4, 32)))

    stageA_mock = MagicMock()
    stageA_mock.predict_adjacency.return_value = torch.eye(4)

    merger_mock = MagicMock()
    merger_mock.return_value = torch.ones((4, 128))
    merger_mock.to.return_value = merger_mock

    # Use patching to replace the real implementations with mocks
    with patch.object(StageBTorsionBertPredictor, '__call__', return_value={"torsion_angles": torch.ones((4, 7))}), \
         patch.object(PairformerWrapper, '__call__', return_value=(torch.ones((4, 64)), torch.ones((4, 4, 32)))), \
         patch.object(PairformerWrapper, 'predict', return_value=(torch.ones((4, 64)), torch.ones((4, 4, 32)))), \
         patch.object(StageARFoldPredictor, 'predict_adjacency', return_value=torch.eye(4)), \
         patch.object(SimpleLatentMerger, '__call__', return_value=torch.ones((4, 128))):

        try:
            # Revert: Pass full cfg, not cfg.model
            model = RNALightningModule(cfg)
        except Exception as e:
            # Unique error for pipeline construction failure
            raise RuntimeError(f"[UNIQUE-ERR-PIPELINE-CONSTRUCT] Pipeline failed to construct: {e}")

        # Create a dummy train_dataloader method that returns a simple dataloader
        def dummy_train_dataloader(self):
            class DummyDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return 1

                def __getitem__(self, _):
                    return {
                        "sequence": "ACGU",
                        "coords_true": torch.zeros((4, 3)),
                        "atom_mask": torch.ones(4, dtype=torch.bool),
                        "atom_to_token_idx": [0, 1, 2, 3],
                        "ref_element": ["C", "G", "A", "U"],
                        "ref_atom_name_chars": ["C", "G", "A", "U"],
                        "atom_names": ["C", "G", "A", "U"],
                        "residue_indices": [0, 1, 2, 3]
                    }

            return torch.utils.data.DataLoader(
                DummyDataset(),
                batch_size=1,
                shuffle=False,
                num_workers=0
            )

        # Replace the train_dataloader method with our dummy implementation
        model.train_dataloader = dummy_train_dataloader.__get__(model)

        trainer = L.Trainer(fast_dev_run=True, enable_progress_bar=False, logger=False)
        try:
            trainer.fit(model)
        except Exception as e:
            raise RuntimeError(f"[UNIQUE-ERR-TRAINER] Trainer failed: {e}")

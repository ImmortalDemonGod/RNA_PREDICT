import lightning as L
import torch
import torch.utils.data
from omegaconf import OmegaConf
from rna_predict.training.rna_lightning_module import RNALightningModule

# The test function will use context-managed patches to avoid global side-effects

# Use a minimal but real config for the full pipeline
cfg = OmegaConf.create({
    'device': 'cpu',
    'model': {
        'stageA': {
            'checkpoint_path': 'dummy_path',
            'device': 'cpu',
            'min_seq_length': 4,  # Reduced from 8
            'num_hidden': 8,      # Reduced from 64
            'dropout': 0.1,       # Added required key for StageARFoldPredictor
            'batch_size': 1,      # Added required key for StageARFoldPredictor
            'lr': 1e-3,           # Added required key for StageARFoldPredictor
            'model': {
                'conv_channels': [8, 16],  # Reduced from [32, 64]
                'residual': True,
                'c_in': 8,                 # Reduced from 32
                'c_out': 8,                # Reduced from 32
                'c_hid': 8,                # Reduced from 32
                'seq2map': {
                    'input_dim': 4,
                    'num_hidden': 8,       # Reduced from 16
                    'dropout': 0.1,
                    'query_key_dim': 4,    # Reduced from 8
                    'expansion_factor': 1.5, # Reduced from 2.0
                    'heads': 1,            # Reduced from 2
                    'attention_heads': 1,  # Reduced from 2
                    'attention_dropout': 0.1,
                    'attention_query_key_dim': 4, # Reduced from 8
                    'attention_expansion_factor': 1.5, # Reduced from 2.0
                    'max_length': 16,      # Reduced from 3000 (only need to handle test sequences)
                    'positional_encoding': True,
                    'use_positional_encoding': True,
                    'use_attention': True
                },
                'decoder': {
                    'up_conv_channels': [16, 8], # Reduced from [64, 32]
                    'skip_connections': True
                }
            }
        },
        'stageB': {
            'torsion_bert': {
                'init_from_scratch': True,  # Enable dummy mode
                'device': 'cpu',
                'angle_mode': 'degrees',
                'num_angles': 7,  # Changed to 7 to match expected output
                'max_length': 512,  # Add explicit max_length
                'debug_logging': True  # Enable debug logging
            },
            'pairformer': {
                'model_name_or_path': 'dummy_path',
                'device': 'cpu',
                'stageB_pairformer': {
                    'c_z': 8,              # Reduced from 32
                    'c_s': 16,             # Reduced from 64
                    'dropout': 0.1,
                    'n_blocks': 1,
                    'n_heads': 1,          # Reduced from 2
                    'enable': True,
                    'c_m': 8,              # Reduced from 32
                    'c': 8,                # Reduced from 32
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
        'stageC': {
            'enabled': True,
            'method': 'mp_nerf',
            'device': 'cpu',
            'do_ring_closure': True,
            'place_bases': True,
            'sugar_pucker': "C3'-endo",
            'angle_representation': 'degrees',
            'use_metadata': True,
            'use_memory_efficient_kernel': False,
            'use_deepspeed_evo_attention': False,
            'use_lma': False,
            'inplace_safe': True,
            'debug_logging': False,
            'mp_nerf': {
                'enabled': True
            }
        },
        'stageD': {
            'enabled': True,
            'device': 'cpu',
            'debug_logging': True,
            'diffusion': {
                'enabled': True,
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
                    'c_token': 16,         # Reduced from 128
                    'c_s': 32,             # Reduced from 384
                    'c_z': 16,             # Reduced from 128
                    'c_s_inputs': 32,      # Reduced from 449
                    'c_atom': 16,          # Reduced from 128
                    'c_atompair': 8,       # Reduced from 16
                    'c_noise_embedding': 32, # Reduced from 256
                    'sigma_data': 16.0,
                    'num_layers': 1,
                    'num_heads': 1,
                    'dropout': 0.0,
                    'coord_eps': 1e-6,
                    'coord_min': -10000.0,
                    'coord_max': 10000.0,
                    'coord_similarity_rtol': 0.001,
                    'test_residues_per_batch': 1
                },
                'noise_schedule': {  # Add noise schedule config
                    'p_mean': -1.2,
                    'p_std': 1.5,
                    's_min': 4e-4
                }
            }
        },
        'latent_merger': {
            'dim_angles': 7,
            'dim_s': 64,
            'dim_z': 32,
            'output_dim': 128,
            'device': 'cpu'  # Add explicit device parameter
        }
    },
})

def test_trainer_fast_dev_run():
    # Create a minimal test configuration
    cfg = OmegaConf.create({
        'device': 'cpu',
        'model': {
            'stageA': {
                'checkpoint_path': 'dummy_path',
                'num_hidden': 128,
                'dropout': 0.1,
                'batch_size': 1,
                'lr': 0.001,
                'device': 'cpu',
                'dummy_mode': True,
                'example_sequence_length': 10,
                'min_seq_length': 1,
                'max_seq_length': 100,
                'debug_logging': True,
                'model': {
                    'conv_channels': [8, 16],
                    'residual': True,
                    'c_in': 8,
                    'c_out': 8,
                    'c_hid': 8,
                    'seq2map': {
                        'input_dim': 4,
                        'num_hidden': 8,
                        'dropout': 0.1,
                        'query_key_dim': 4,
                        'expansion_factor': 1.5,
                        'heads': 1,
                        'attention_heads': 1,
                        'attention_dropout': 0.1,
                        'attention_query_key_dim': 4,
                        'attention_expansion_factor': 1.5,
                        'max_length': 16,
                        'positional_encoding': True,
                        'use_positional_encoding': True,
                        'use_attention': True
                    },
                    'decoder': {
                        'up_conv_channels': [16, 8],
                        'skip_connections': True
                    }
                }
            },
            'stageB': {
                'torsion_bert': {
                    'checkpoint_path': 'dummy_path',
                    'num_hidden': 128,
                    'dropout': 0.1,
                    'batch_size': 1,
                    'lr': 0.001,
                    'device': 'cpu',
                    'dummy_mode': True,
                    'init_from_scratch': True,
                    'angle_mode': 'degrees',
                    'num_angles': 7,
                    'max_length': 512,
                    'debug_logging': True
                },
                'pairformer': {
                    'model_name_or_path': 'dummy_path',
                    'device': 'cpu',
                    'stageB_pairformer': {
                        'c_z': 8,
                        'c_s': 16,
                        'dropout': 0.1,
                        'n_blocks': 1,
                        'n_heads': 1,
                        'enable': True,
                        'c_m': 8,
                        'c': 8,
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
            'stageC': {
                'enabled': True,
                'method': 'mp_nerf',
                'device': 'cpu',
                'do_ring_closure': True,
                'place_bases': True,
                'sugar_pucker': "C3'-endo",
                'angle_representation': 'degrees',
                'use_metadata': True,
                'use_memory_efficient_kernel': False,
                'use_deepspeed_evo_attention': False,
                'use_lma': False,
                'inplace_safe': True,
                'debug_logging': False,
                'mp_nerf': {
                    'enabled': True
                }
            },
            'stageD': {
                'enabled': True,
                'device': 'cpu',
                'debug_logging': True,
                'diffusion': {
                    'enabled': True,
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
                        'c_token': 16,
                        'c_s': 32,
                        'c_z': 16,
                        'c_s_inputs': 32,
                        'c_atom': 16,
                        'c_atompair': 8,
                        'c_noise_embedding': 32,
                        'sigma_data': 16.0,
                        'num_layers': 1,
                        'num_heads': 1,
                        'dropout': 0.0,
                        'coord_eps': 1e-6,
                        'coord_min': -10000.0,
                        'coord_max': 10000.0,
                        'coord_similarity_rtol': 0.001,
                        'test_residues_per_batch': 1
                    },
                    'noise_schedule': {
                        'p_mean': -1.2,
                        'p_std': 1.5,
                        's_min': 4e-4
                    }
                }
            },
            'latent_merger': {
                'dim_angles': 7,
                'dim_s': 64,
                'dim_z': 32,
                'output_dim': 128,
                'device': 'cpu'
            }
        }
    })

    # Create a minimal test batch
    batch = {
        'sequence': 'ACGU',
        'coords_true': torch.randn(1, 1, 4, 3),  # [batch, num_chains, num_residues, xyz]
        'atom_mask': torch.ones(1, 1, 4).bool(),  # [batch, num_chains, num_residues]
        'atom_to_token_idx': torch.arange(4).unsqueeze(0).unsqueeze(0),  # [batch, num_chains, num_residues]
        'ref_element': torch.randint(0, 10, (1, 1, 4)),  # [batch, num_chains, num_residues]
        'ref_atom_name_chars': torch.randint(0, 10, (1, 1, 4, 4)),  # [batch, num_chains, num_residues, 4]
        'atom_names': ['C1', 'C2', 'C3', 'C4'],
        'residue_indices': torch.arange(4).unsqueeze(0).unsqueeze(0),  # [batch, num_chains, num_residues]
        'adjacency': torch.randint(0, 2, (1, 1, 4, 4)).float()  # [batch, num_chains, num_residues, num_residues]
    }

    # Initialize the model and run a single training step
    model = RNALightningModule(cfg)
    trainer = L.Trainer(accelerator='cpu', fast_dev_run=True)
    trainer.fit(model, torch.utils.data.DataLoader([batch], batch_size=1))

    # Write detailed model summary to a file
    with open("model_summary.txt", "w") as f:
        f.write("Detailed Model Summary:\n")
        total_params = 0
        for name, module in model.named_children():
            f.write(f"\n{name}:\n")
            f.write(str(module))
            params = sum(p.numel() for p in module.parameters())
            total_params += params
            f.write(f"\nParameters: {params:,}\n")
        f.write(f"\nTotal Parameters: {total_params:,}\n")

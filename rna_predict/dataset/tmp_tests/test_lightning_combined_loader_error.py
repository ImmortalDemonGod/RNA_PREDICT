import tracemalloc
from omegaconf import OmegaConf
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.dataset.loader import RNADataset
from torch.utils.data import DataLoader
import lightning as L
from rna_predict.conf.config_schema import RNAConfig

# TODO: These modules need to be implemented or moved to the correct location
# from rna_predict.models.latent_merger import LatentMerger
# from rna_predict.models.stageB_pairformer import run_stageB_pairformer
# from rna_predict.models.stageB_torsion import run_stageB_torsion

def test_lightning_combined_loader_error():
    # Start memory profiling
    tracemalloc.start()
    
    dict_cfg = {
        'sequence': 'ACGUACGU',
        'seed': 42,
        'device': 'cpu',
        'atoms_per_residue': 44,
        'device_management': {
            'primary': 'cpu',
            'fallback': 'cpu',
            'auto_fallback': True,
            'force_components_to_cpu': []
        },
        'run_stageD': True,
        'enable_stageC': True,
        'merge_latent': True,
        'init_z_from_adjacency': False,
        'dimensions': {
            'c_s': 64,
            'c_z': 32,
            'c_s_inputs': 64,
            'c_token': 64,
            'c_atom': 32,
            'c_atom_coords': 3,
            'c_noise_embedding': 32,
            'restype_dim': 32,
            'profile_dim': 32,
            'c_pair': 32,
            'ref_element_size': 128,
            'ref_atom_name_chars_size': 256
        },
        'model': {
            'stageA': {
                'enabled': True,
                'num_hidden': 1,
                'dropout': 0.0,
                'debug_logging': True,
                'freeze_params': True,
                'min_seq_length': 1,
                'batch_size': 1,
                'lr': 0.001,
                'device': 'cpu',
                'checkpoint_path': 'dummy.ckpt',
                'checkpoint_url': 'https://www.dropbox.com/s/l04l9bf3v6z2tfd/checkpoints.zip?dl=1',
                'checkpoint_zip_path': 'RFold/checkpoints.zip',
                'threshold': 0.5,
                'run_example': True,
                'example_sequence': 'AAGUCUGGUGGACAUUGGCGUCCUGAGGUGUUAAAACCUCUUAUUGCUGACGCCAGAAAGAGAAGAACUUCGGUUCUACUAGUCGACUAUACUACAAGCUUUGGGUGUAUAGCGGCAAGACAACCUGGAUCGGGGGAGGCUAAGGGCGCAAGCCUAUGCUAACCCCGAGCCGAGCUACUGGAGGGCAACCCCCAGAUAGCCGGUGUAGAGCGCGGAAAGGUGUCGGUCAUCCUAUCUGAUAGGUGGCUUGAGGGACGUGCCGUCUCACCCGAAAGGGUGUUUCUAAGGAGGAGCUCCCAAAGGGCAAAUCUUAGAAAAGGGUGUAUACCCUAUAAUUUAACGGCCAGCAGCC',
                'visualization': {
                    'enabled': True,
                    'varna_jar_path': 'tools/varna-3-93.jar',
                    'resolution': 8.0,
                    'output_path': 'test_seq.png'
                },
                'model': {
                    'conv_channels': [64, 128, 256, 512],
                    'residual': True,
                    'c_in': 1,
                    'c_out': 1,
                    'c_hid': 32,
                    'seq2map': {
                        'input_dim': 4,
                        'max_length': 3000,
                        'attention_heads': 8,
                        'attention_dropout': 0.1,
                        'positional_encoding': True,
                        'query_key_dim': 128,
                        'expansion_factor': 2.0,
                        'heads': 1
                    },
                    'decoder': {
                        'up_conv_channels': [256, 128, 64],
                        'skip_connections': True
                    }
                }
            },
            'stageB_torsion': {
                'model_name_or_path': 'sayby/rna_torsionbert',
                'device': 'cpu',
                'angle_mode': 'sin_cos',
                'num_angles': 7,
                'max_length': 512,
                'checkpoint_path': None,
                'lora': {
                    'enabled': False,
                    'r': 8,
                    'alpha': 16,
                    'dropout': 0.1,
                    'target_modules': []
                },
                'debug_logging': True,
                'init_from_scratch': False
            },
            'stageB_pairformer': {
                'device': 'cpu',
                'n_blocks': 1,
                'n_heads': 1,
                'c_z': 32,
                'c_s': 64,
                'c_token': 64,
                'c_atom': 32,
                'c_pair': 32,
                'dropout': 0.1,
                'freeze_params': False,
                'protenix_integration': {
                    'device': 'cpu',
                    'c_token': 64,
                    'restype_dim': 32,
                    'profile_dim': 32,
                    'c_atom': 32,
                    'c_pair': 32,
                    'atoms_per_token': 4,
                    'num_heads': 2,
                    'num_layers': 2,
                    'r_max': 32,
                    's_max': 2,
                    'use_optimized': False
                },
                'c_hidden_mul': 2,
                'c_hidden_pair_att': 4,
                'no_heads_pair': 2,
                'init_z_from_adjacency': False,
                'use_checkpoint': True,
                'use_memory_efficient_kernel': False,
                'use_deepspeed_evo_attention': False,
                'use_lma': False,
                'inplace_safe': False,
                'chunk_size': None,
                'block': {
                    'n_heads': 2,
                    'c_z': 32,
                    'c_s': 64,
                    'c_hidden_mul': 4,
                    'c_hidden_pair_att': 4,
                    'no_heads_pair': 2,
                    'dropout': 0.25
                },
                'stack': {
                    'n_blocks': 1,
                    'n_heads': 2,
                    'c_z': 32,
                    'c_s': 64,
                    'dropout': 0.25,
                    'blocks_per_ckpt': None
                },
                'msa': {
                    'c_m': 64,
                    'c': 32,
                    'c_z': 32,
                    'dropout': 0.15,
                    'n_blocks': 1,
                    'enable': False,
                    'strategy': 'random',
                    'train_cutoff': 512,
                    'test_cutoff': 16384,
                    'train_lowerb': 1,
                    'test_lowerb': 1,
                    'n_heads': 2,
                    'pair_dropout': 0.25,
                    'input_feature_dims': {
                        'msa': 4,
                        'has_deletion': 1,
                        'deletion_value': 1
                    },
                    'c_s_inputs': 64,
                    'blocks_per_ckpt': 1
                },
                'template': {
                    'n_blocks': 1,
                    'c': 64,
                    'c_z': 32,
                    'dropout': 0.25,
                    'blocks_per_ckpt': None,
                    'input_feature_dims': {
                        'feature1': {
                            'template_distogram': 39,
                            'b_template_backbone_frame_mask': 1,
                            'template_unit_vector': 3,
                            'b_template_pseudo_beta_mask': 1
                        },
                        'feature2': {
                            'template_restype_i': 32,
                            'template_restype_j': 32
                        }
                    },
                    'distogram': {
                        'max_bin': 50.75,
                        'min_bin': 3.25,
                        'no_bins': 39.0
                    }
                },
                'lora': {
                    'enabled': False,
                    'r': 8,
                    'alpha': 16,
                    'dropout': 0.1,
                    'target_modules': []
                },
                'debug_logging': True
            },
            'stageB': {
                'torsion_bert': {
                    'model_name_or_path': 'sayby/rna_torsionbert',
                    'device': 'cpu',
                    'angle_mode': 'sin_cos',
                    'num_angles': 7,
                    'max_length': 512,
                    'checkpoint_path': None,
                    'lora': {
                        'enabled': False,
                        'r': 8,
                        'alpha': 16,
                        'dropout': 0.1,
                        'target_modules': []
                    },
                    'debug_logging': True,
                    'init_from_scratch': False
                },
                'pairformer': {
                    'device': 'cpu',
                    'n_blocks': 1,
                    'n_heads': 1,
                    'c_z': 32,
                    'c_s': 64,
                    'c_token': 64,
                    'c_atom': 32,
                    'c_pair': 32,
                    'dropout': 0.1,
                    'freeze_params': False,
                    'protenix_integration': {
                        'device': 'cpu',
                        'c_token': 64,
                        'restype_dim': 32,
                        'profile_dim': 32,
                        'c_atom': 32,
                        'c_pair': 32,
                        'atoms_per_token': 4,
                        'num_heads': 2,
                        'num_layers': 2,
                        'r_max': 32,
                        's_max': 2,
                        'use_optimized': False
                    },
                    'c_hidden_mul': 2,
                    'c_hidden_pair_att': 4,
                    'no_heads_pair': 2,
                    'init_z_from_adjacency': False,
                    'use_checkpoint': True,
                    'use_memory_efficient_kernel': False,
                    'use_deepspeed_evo_attention': False,
                    'use_lma': False,
                    'inplace_safe': False,
                    'chunk_size': None,
                    'block': {
                        'n_heads': 2,
                        'c_z': 32,
                        'c_s': 64,
                        'c_hidden_mul': 4,
                        'c_hidden_pair_att': 4,
                        'no_heads_pair': 2,
                        'dropout': 0.25
                    },
                    'stack': {
                        'n_blocks': 1,
                        'n_heads': 2,
                        'c_z': 32,
                        'c_s': 64,
                        'dropout': 0.25,
                        'blocks_per_ckpt': None
                    },
                    'msa': {
                        'c_m': 64,
                        'c': 32,
                        'c_z': 32,
                        'dropout': 0.15,
                        'n_blocks': 1,
                        'enable': False,
                        'strategy': 'random',
                        'train_cutoff': 512,
                        'test_cutoff': 16384,
                        'train_lowerb': 1,
                        'test_lowerb': 1,
                        'n_heads': 2,
                        'pair_dropout': 0.25,
                        'input_feature_dims': {
                            'msa': 4,
                            'has_deletion': 1,
                            'deletion_value': 1
                        },
                        'c_s_inputs': 64,
                        'blocks_per_ckpt': 1
                    },
                    'template': {
                        'n_blocks': 1,
                        'c': 64,
                        'c_z': 32,
                        'dropout': 0.25,
                        'blocks_per_ckpt': None,
                        'input_feature_dims': {
                            'feature1': {
                                'template_distogram': 39,
                                'b_template_backbone_frame_mask': 1,
                                'template_unit_vector': 3,
                                'b_template_pseudo_beta_mask': 1
                            },
                            'feature2': {
                                'template_restype_i': 32,
                                'template_restype_j': 32
                            }
                        },
                        'distogram': {
                            'max_bin': 50.75,
                            'min_bin': 3.25,
                            'no_bins': 39.0
                        }
                    },
                    'lora': {
                        'enabled': False,
                        'r': 8,
                        'alpha': 16,
                        'dropout': 0.1,
                        'target_modules': []
                    },
                    'debug_logging': True
                }
            },
            'stageC': {
                'enabled': True,
                'method': 'mp_nerf',
                'do_ring_closure': False,
                'place_bases': True,
                'sugar_pucker': "C3'-endo",
                'device': 'cpu',
                'angle_representation': 'cartesian',
                'use_metadata': False,
                'use_memory_efficient_kernel': False,
                'use_deepspeed_evo_attention': False,
                'use_lma': False,
                'inplace_safe': False,
                'chunk_size': None,
                'debug_logging': True
            },
            'stageD': {
                'enabled': True,
                'mode': 'inference',
                'device': 'cpu',
                'debug_logging': True,
                'ref_element_size': 128,
                'ref_atom_name_chars_size': 256,
                'profile_size': 32,
                'model_architecture': {
                    'c_token': 64,
                    'c_s': 64,
                    'c_z': 32,
                    'c_s_inputs': 64,
                    'c_atom': 32,
                    'c_noise_embedding': 32,
                    'num_layers': 2,
                    'num_heads': 2,
                    'dropout': 0.1,
                    'coord_eps': 1e-06,
                    'coord_min': -10000.0,
                    'coord_max': 10000.0,
                    'coord_similarity_rtol': 0.001,
                    'test_residues_per_batch': 2,
                    'c_atompair': 32,
                    'sigma_data': 1.0
                },
                'transformer': {
                    'n_blocks': 2,
                    'n_heads': 2,
                    'blocks_per_ckpt': None
                },
                'atom_encoder': {
                    'c_in': 32,
                    'c_hidden': [64],
                    'c_out': 32,
                    'dropout': 0.1,
                    'n_blocks': 1,
                    'n_heads': 2,
                    'n_queries': 2,
                    'n_keys': 2
                },
                'atom_decoder': {
                    'c_in': 32,
                    'c_hidden': [64, 32, 16],
                    'c_out': 32,
                    'dropout': 0.1,
                    'n_blocks': 1,
                    'n_heads': 2,
                    'n_queries': 2,
                    'n_keys': 2
                },
                'diffusion': {
                    'init_from_scratch': True,
                    'enabled': True,
                    'mode': 'inference',
                    'device': 'cpu',
                    'debug_logging': True,
                    'ref_element_size': 128,
                    'ref_atom_name_chars_size': 256,
                    'profile_size': 32,
                    'feature_dimensions': {
                        'c_s': 64,
                        'c_s_inputs': 64,
                        'c_sing': 64,
                        's_trunk': 64,
                        's_inputs': 64
                    },
                    'test_residues_per_batch': 2,
                    'model_architecture': {
                        'c_token': 64,
                        'c_s': 64,
                        'c_z': 32,
                        'c_s_inputs': 64,
                        'c_atom': 32,
                        'c_atompair': 32,
                        'c_noise_embedding': 32,
                        'sigma_data': 1.0
                    },
                    'transformer': {
                        'n_blocks': 2,
                        'n_heads': 2,
                        'blocks_per_ckpt': None
                    },
                    'atom_encoder': {
                        'c_in': 32,
                        'c_hidden': [64],
                        'c_out': 32,
                        'dropout': 0.1,
                        'n_blocks': 1,
                        'n_heads': 2,
                        'n_queries': 2,
                        'n_keys': 2
                    },
                    'atom_decoder': {
                        'c_in': 32,
                        'c_hidden': [64],
                        'c_out': 32,
                        'dropout': 0.1,
                        'n_blocks': 1,
                        'n_heads': 2,
                        'n_queries': 2,
                        'n_keys': 2
                    },
                    'noise_schedule': {
                        'schedule_type': 'linear',
                        's_max': 1.0,
                        's_min': 0.01,
                        'p': 0.5,
                        'p_mean': 0.0,
                        'p_std': 1.0
                    },
                    'inference': {
                        'num_steps': 2,
                        'temperature': 1.0,
                        'use_ddim': True,
                        'sampling': {
                            'num_samples': 1,
                            'gamma0': 0.8,
                            'gamma_min': 1.0,
                            'noise_scale_lambda': 1.003,
                            'step_scale_eta': 1.5
                        }
                    },
                    'use_memory_efficient_kernel': False,
                    'use_deepspeed_evo_attention': False,
                    'use_lma': False,
                    'inplace_safe': False,
                    'chunk_size': None
                }
            }
        },
        'pipeline': {
            'verbose': True,
            'save_intermediates': True,
            'output_dir': 'outputs',
            'ignore_nan_values': False,
            'nan_replacement_value': 0.0,
            'lance_db': {
                'enabled': False
            }
        },
        'latent_merger': {
            'merge_method': 'concat',
            'attention_heads': 8,
            'dropout': 0.1,
            'output_dim': 384,
            'use_residual': True
        },
        'memory_optimization': {
            'use_checkpointing': True,
            'checkpoint_every_n_layers': 2,
            'optimize_memory_layout': True,
            'mixed_precision': True
        },
        'energy_minimization': {
            'enabled': True,
            'method': 'steepest_descent',
            'max_iterations': 1000,
            'tolerance': 1e-06,
            'learning_rate': 0.01
        },
        'test_data': {
            'sequence': 'ACGUACGU',
            'sequence_length': 8,
            'atoms_per_residue': 44,
            'adjacency_fill_value': 1.0,
            'target_dim': 3,
            'torsion_angle_dim': 7,
            'embedding_dims': {
                's_trunk': 384,
                'z_trunk': 128,
                's_inputs': 32
            },
            'sequence_path': None,
            'data_index': None,
            'target_id': None,
            'model': None
        },
        'data': {
            'index_csv': 'rna_predict/dataset/examples/kaggle_minimal_index.csv',
            'root_dir': './data/',
            'max_residues': 128,
            'max_atoms': 1024,
            'C_element': 128,
            'C_char': 256,
            'ref_element_size': 128,
            'ref_atom_name_chars_size': 256,
            'batch_size': 1,
            'num_workers': 0,
            'load_adj': True,
            'load_ang': True,
            'coord_fill_value': 'nan',
            'coord_dtype': 'float32'
        },
        'training': {
            'checkpoint_dir': 'outputs/checkpoints',
            'accelerator': 'cpu',
            'devices': 1
        }
    }
    
    try:
        cfg = OmegaConf.merge(OmegaConf.structured(RNAConfig), dict_cfg)
        print("Memory after config creation:", tracemalloc.get_traced_memory())
        
        dataset = RNADataset(cfg.data.index_csv, cfg, load_adj=True, load_ang=True)
        print("Memory after dataset creation:", tracemalloc.get_traced_memory())
        
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
        )
        print("Memory after dataloader creation:", tracemalloc.get_traced_memory())
        
        lightning_module = RNALightningModule(cfg)
        print("Memory after lightning module creation:", tracemalloc.get_traced_memory())
        
        trainer = L.Trainer(fast_dev_run=True, enable_model_summary=False, logger=False)
        print("Memory after trainer creation:", tracemalloc.get_traced_memory())

        trainer.fit(lightning_module, train_dataloaders=dataloader)
        
    finally:
        # Print memory snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        print("\nTop 10 memory users:")
        for stat in top_stats[:10]:
            print(stat)
        
        tracemalloc.stop()

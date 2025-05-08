import pytest
from omegaconf import OmegaConf
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.dataset.loader import RNADataset
from torch.utils.data import DataLoader
import lightning as L
from rna_predict.conf.config_schema import RNAConfig

def test_lightning_combined_loader_error():
    dict_cfg = {
        "device": "cpu",
        "data": {
            "index_csv": "rna_predict/dataset/examples/kaggle_minimal_index.csv",
            "batch_size": 2,
            "num_workers": 8,  # Set to 8 to match main pipeline
            "max_residues": 512,
            "max_atoms": 4096,
            "coord_dtype": "float32",
            "coord_fill_value": "nan",
            "ref_element_size": 5,
            "ref_atom_name_chars_size": 10,
        },
        "model": {
            "stageA": {
                "checkpoint_path": "dummy.ckpt",
                "min_seq_length": 1,
                "num_hidden": 1,
                "dropout": 0.0,
                "batch_size": 1,
                "lr": 0.001,
                "model": {},
            },
            "stageB": {
                "torsion_bert": {
                    "device": "cpu"
                },
                "pairformer": {
                    "device": "cpu"
                },
            },
            "stageC": {},
            "stageD": {
                "diffusion": {
                    "init_from_scratch": True,
                    "enabled": True,
                    "mode": "inference",
                    "device": "cpu",
                    "debug_logging": True,
                    "ref_element_size": 5,
                    "ref_atom_name_chars_size": 10,
                    "profile_size": 1,
                    "feature_dimensions": {
                        "c_s": 1,
                        "c_s_inputs": 1,
                        "c_sing": 1,
                        "s_trunk": 1,
                        "s_inputs": 1
                    },
                    "test_residues_per_batch": 1,
                    "model_architecture": {
                        "c_token": 1,
                        "c_s": 1,
                        "c_z": 1,
                        "c_s_inputs": 1,
                        "c_atom": 1,
                        "c_atompair": 1,
                        "c_noise_embedding": 1,
                        "sigma_data": 1.0
                    },
                    "transformer": {
                        "n_blocks": 1,
                        "n_heads": 1,
                        "blocks_per_ckpt": None
                    },
                    "atom_encoder": {
                        "c_in": 1,
                        "c_hidden": [1],
                        "c_out": 1,
                        "dropout": 0.1,
                        "n_blocks": 1,
                        "n_heads": 1,
                        "n_queries": 1,
                        "n_keys": 1
                    },
                    "atom_decoder": {
                        "c_in": 1,
                        "c_hidden": [1],
                        "c_out": 1,
                        "dropout": 0.1,
                        "n_blocks": 1,
                        "n_heads": 1,
                        "n_queries": 1,
                        "n_keys": 1
                    },
                    "noise_schedule": {
                        "schedule_type": "linear",
                        "s_max": 1.0,
                        "s_min": 0.01,
                        "p": 0.5,
                        "p_mean": 0.0,
                        "p_std": 1.0
                    },
                    "inference": {
                        "num_steps": 2,
                        "temperature": 1.0,
                        "use_ddim": True,
                        "sampling": {
                            "num_samples": 1,
                            "gamma0": 0.8,
                            "gamma_min": 1.0,
                            "noise_scale_lambda": 1.003,
                            "step_scale_eta": 1.5
                        }
                    },
                    "use_memory_efficient_kernel": False,
                    "use_deepspeed_evo_attention": False,
                    "use_lma": False,
                    "inplace_safe": False,
                    "chunk_size": None
                }
            },
        }
    }
    cfg = OmegaConf.merge(OmegaConf.structured(RNAConfig), dict_cfg)
    dataset = RNADataset(cfg.data.index_csv, cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    lightning_module = RNALightningModule(cfg)
    trainer = L.Trainer(fast_dev_run=True, enable_model_summary=False, logger=False)

    with pytest.raises(RuntimeError, match=r"Please call `iter\\(combined_loader\\)` first."):
        trainer.fit(lightning_module, train_dataloaders=dataloader)

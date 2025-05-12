import pytest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from rna_predict.training.rna_lightning_module import RNALightningModule

@pytest.fixture
def hydra_cfg():
    # Hypothesis-driven fix: use initialize_config_dir for absolute paths
    abs_conf = "/Users/tomriddle1/RNA_PREDICT/rna_predict/conf"
    with initialize_config_dir(config_dir=abs_conf, version_base=None):
        cfg = compose(config_name="default")
        cfg.device = "cpu"
        if not hasattr(cfg.model.stageD, 'diffusion'):
            cfg.model.stageD.diffusion = OmegaConf.create({})
        cfg.model.stageD.diffusion.noise_schedule = OmegaConf.create({
            'p_mean': -1.2,
            'p_std': 1.5,
            's_min': 4e-4,
            's_max': 160.0,
        })
        cfg.model.stageD.diffusion.model_architecture = OmegaConf.create({
            'c_token': 8,
            'c_s': 8,
            'c_z': 4,
            'c_s_inputs': 8,
            'c_atom': 4,
            'c_noise_embedding': 4,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'coord_eps': 1e-6,
            'coord_min': -1e4,
            'coord_max': 1e4,
            'coord_similarity_rtol': 1e-3,
            'test_residues_per_batch': 2,
            'c_atompair': 4,
            'sigma_data': 1.0,
        })
        return cfg

def test_sample_noise_level_shape_and_device(hydra_cfg):
    model = RNALightningModule(cfg=hydra_cfg)
    batch_size = 7
    sigma_t = model._sample_noise_level(batch_size)
    assert sigma_t.shape == (batch_size,)
    assert sigma_t.device.type == 'cpu'
    assert (sigma_t > 0).all()
    assert sigma_t.min() >= torch.tensor(hydra_cfg.model.stageD.diffusion.noise_schedule.s_min)
    assert sigma_t.max() <= torch.tensor(hydra_cfg.model.stageD.diffusion.noise_schedule.s_max)

def test_add_noise_shapes_and_broadcasting(hydra_cfg):
    model = RNALightningModule(cfg=hydra_cfg)
    batch_size = 4
    N_atom = 10
    coords_true = torch.randn(batch_size, N_atom, 3)
    sigma_t = model._sample_noise_level(batch_size)
    coords_noisy, epsilon = model._add_noise(coords_true, sigma_t)
    assert coords_noisy.shape == coords_true.shape
    assert epsilon.shape == coords_true.shape
    assert not torch.equal(coords_noisy, coords_true)
    assert coords_noisy.device == coords_true.device
    assert epsilon.device == coords_true.device

def test_add_noise_empty_tensor(hydra_cfg):
    model = RNALightningModule(cfg=hydra_cfg)
    coords_true = torch.empty(0, 5, 3)
    sigma_t = torch.empty(0)
    coords_noisy, epsilon = model._add_noise(coords_true, sigma_t)
    assert coords_noisy.shape == coords_true.shape
    assert epsilon.shape == coords_true.shape
    assert torch.equal(coords_noisy, coords_true)
    assert torch.equal(epsilon, torch.zeros_like(coords_true))

def test_sample_noise_level_extreme_sigma(hydra_cfg):
    hydra_cfg.model.stageD.diffusion.noise_schedule.s_min = 1e-8
    hydra_cfg.model.stageD.diffusion.noise_schedule.s_max = 1e2
    model = RNALightningModule(cfg=hydra_cfg)
    sigma_t = model._sample_noise_level(3)
    assert (sigma_t >= 1e-8).all()
    assert (sigma_t <= 1e2).all()

# tests/stageD/unit/diffusion/test_generator.py
from typing import Any, Dict, Optional

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Assuming these imports match your project structure.
# If necessary, adjust the import path accordingly (e.g., from rna_predict.pipeline.stageD.diffusion.generator import ...)
from rna_predict.pipeline.stageD.diffusion.generator import (
    InferenceNoiseScheduler,
    TrainingNoiseSampler,
    sample_diffusion,
    sample_diffusion_training,
)


@pytest.fixture
def mock_denoise_net():
    """
    Returns a mock function simulating the denoising network.
    This will help us ensure we can track the calls and return shapes we expect.
    """

    def _mock_denoise_net(
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: Dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        chunk_size: Optional[int] = None,
        inplace_safe: bool = False,
        debug_logging: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:  # Return a tuple (coords, loss)
        # Return something shaped like x_noisy but offset to test differences
        mock_coords = x_noisy - 0.1  # simple offset
        mock_loss = torch.tensor(0.0, device=x_noisy.device)  # Dummy scalar loss
        return mock_coords, mock_loss

    return _mock_denoise_net


class TestTrainingNoiseSampler:
    """
    Test suite for the TrainingNoiseSampler class to ensure it correctly
    samples noise levels based on specified parameters.
    """

    def test_default_init_and_call(self) -> None:
        """
        Test the default constructor and verify sampling shape and range.
        """
        sampler = TrainingNoiseSampler()
        size = torch.Size([2, 3])
        noise_levels = sampler(size=size, device=torch.device("cpu"))
        assert (
            noise_levels.shape == size
        ), "Noise level shape should match requested size"
        assert (noise_levels >= 0).all(), "Noise level should be non-negative"

    @pytest.mark.parametrize(
        "p_mean, p_std, sigma_data",
        [
            (-2.0, 0.5, 8.0),
            (0.0, 1.0, 1.0),
            (1.0, 2.0, 16.0),
            (2.0, 3.0, 32.0),
        ],
    )
    def test_param_init_and_call(
        self, p_mean: float, p_std: float, sigma_data: float
    ) -> None:
        """
        Test TrainingNoiseSampler with parameterized values to ensure coverage.
        """
        sampler = TrainingNoiseSampler(
            p_mean=p_mean, p_std=p_std, sigma_data=sigma_data
        )
        size = torch.Size([3, 2])
        noise_levels = sampler(size=size, device=torch.device("cpu"))
        assert noise_levels.shape == size
        # We cannot know the exact distribution, but we verify it's not all zeros
        assert torch.any(
            noise_levels != 0
        ), "Noise levels should not be identically zero"

    @given(
        p_mean=st.floats(min_value=-5, max_value=5),
        p_std=st.floats(min_value=0.01, max_value=5),
        sigma_data=st.floats(min_value=0.01, max_value=100),
        width=st.integers(min_value=1, max_value=5),
        height=st.integers(min_value=1, max_value=5),
    )
    def test_fuzz_init_and_call(
        self, p_mean: float, p_std: float, sigma_data: float, width: int, height: int
    ) -> None:
        """
        Hypothesis-based fuzz test for TrainingNoiseSampler to explore
        a variety of parameter values and verify shape correctness.
        """
        sampler = TrainingNoiseSampler(
            p_mean=p_mean, p_std=p_std, sigma_data=sigma_data
        )
        size = torch.Size([width, height])
        noise_levels = sampler(size=size, device=torch.device("cpu"))
        assert noise_levels.shape == size, "Noise level shape mismatch"
        # We simply check that the resulting tensor is finite
        assert torch.isfinite(noise_levels).all(), "Noise levels must be finite"


class TestInferenceNoiseScheduler:
    """
    Test suite for the InferenceNoiseScheduler class to ensure correct scheduling
    of noise levels and shapes.
    """

    def test_default_init_and_call(self) -> None:
        """
        Test default constructor usage and verify final schedule shape and last step = 0.
        """
        scheduler = InferenceNoiseScheduler()
        schedule = scheduler(device=torch.device("cpu"))
        # Default N_step=200 => schedule length = 201
        assert schedule.shape[0] == 201, "Schedule should have N_step+1 elements"
        assert schedule[-1].item() == 0, "Last element of schedule must be 0"

    @pytest.mark.parametrize(
        "s_max, s_min, p, sigma_data, N_step",
        [
            (160.0, 4e-4, 7.0, 16.0, 200),
            (100.0, 1e-4, 5.0, 8.0, 100),
        ],
    )
    def test_inference_noise_scheduler_call(
        self, s_max: float, s_min: float, p: float, sigma_data: float, N_step: int
    ) -> None:
        """Test InferenceNoiseScheduler call with various parameters."""
        scheduler = InferenceNoiseScheduler(
            s_max=s_max, s_min=s_min, p=p, sigma_data=sigma_data
        )
        schedule = scheduler(N_step=N_step, device=torch.device("cpu"))
        assert schedule.shape[0] == N_step + 1, "Schedule length mismatch"
        assert schedule[-1].item() == 0, "Last element must be zero"
        # Check monotonic decreasing property except for last forced 0:
        # For coverage, let's at least ensure all steps are finite
        assert torch.isfinite(schedule).all(), "All schedule entries must be finite"

    @given(
        s_max=st.floats(min_value=1, max_value=200),
        s_min=st.floats(min_value=1e-6, max_value=1e-2),
        p=st.floats(min_value=1, max_value=10),
        sigma_data=st.floats(min_value=1, max_value=32),
        n_step=st.integers(min_value=10, max_value=1000),
    )
    def test_inference_noise_scheduler_property_based(
        self, s_max: float, s_min: float, p: float, sigma_data: float, n_step: int
    ) -> None:
        """Property-based test for InferenceNoiseScheduler."""
        scheduler = InferenceNoiseScheduler(
            s_max=s_max, s_min=s_min, p=p, sigma_data=sigma_data
        )
        schedule = scheduler(N_step=n_step, device=torch.device("cpu"))
        assert schedule.shape[0] == n_step + 1, "Fuzz: schedule shape mismatch"
        assert schedule[-1].item() == 0, "Fuzz: last schedule element must be zero"
        assert torch.isfinite(schedule).all(), "Fuzz: schedule must be finite"


class TestSampleDiffusion:
    """
    Test suite for the sample_diffusion function, ensuring it handles chunking
    and iterative denoising steps correctly.
    """

    @pytest.fixture
    def basic_input_feature_dict(self) -> Dict[str, Any]:
        """
        Provides a simple input_feature_dict with all required features
        used by sample_diffusion.
        """
        return {
            "atom_to_token_idx": torch.zeros((1, 5), dtype=torch.long),
            "ref_pos": torch.randn(1, 5, 3),
            "ref_space_uid": torch.arange(5).unsqueeze(0),
            "ref_charge": torch.zeros(1, 5, 1),
            "ref_mask": torch.ones(1, 5, 1),
            "ref_element": torch.zeros(1, 5, 128),
            "ref_atom_name_chars": torch.zeros(1, 5, 256),
            "restype": torch.zeros(1, 5, 32),
            "profile": torch.zeros(1, 5, 32),
            "deletion_mean": torch.zeros(1, 5, 1),
            "sing": torch.randn(1, 5, 449),  # Required for s_inputs fallback
        }

    def test_basic_sample_diffusion_no_chunk(
        self, mock_denoise_net, basic_input_feature_dict
    ) -> None:
        """
        Test sample_diffusion with minimal arguments and no chunking.
        """
        noise_schedule = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float32)
        s_inputs = torch.randn(1, 5, 449)  # Updated shape to match config
        s_trunk = torch.randn(1, 5, 384)  # Updated shape to match config
        z_trunk = torch.randn(1, 5, 5, 32)  # Updated shape to match config

        x_l = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=basic_input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=1
        )
        assert isinstance(x_l, torch.Tensor)
        # Expect [batch, N_sample, n_atoms, 3] from sample_diffusion
        assert x_l.ndim == 4, f"Expected 4 dimensions, got {x_l.ndim}"
        assert x_l.shape[0] == 1, f"Expected batch size 1, got {x_l.shape[0]}"
        assert (
            x_l.shape[1] == 1
        ), f"Expected N_sample 1, got {x_l.shape[1]}"  # N_sample=1 was passed
        assert (
            x_l.shape[2] == 5
        ), f"Expected 5 atoms, got {x_l.shape[2]}"  # Check number of atoms matches
        assert (
            x_l.shape[3] == 3
        ), f"Expected 3 coordinates, got {x_l.shape[3]}"  # Check coordinate dimension

    def test_sample_diffusion_with_chunk(
        self, mock_denoise_net, basic_input_feature_dict
    ) -> None:
        """
        Test sample_diffusion with chunking enabled.
        """
        noise_schedule = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float32)
        s_inputs = torch.randn(1, 5, 449)  # Updated shape to match config
        s_trunk = torch.randn(1, 5, 384)  # Updated shape to match config
        z_trunk = torch.randn(1, 5, 5, 32)  # Updated shape to match config

        x_l = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=basic_input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=1,
            diffusion_chunk_size=2,  # Chunk size smaller than total atoms
        )

        assert isinstance(x_l, torch.Tensor)
        # Expect [batch, N_sample, n_atoms, 3] from sample_diffusion
        assert x_l.ndim == 4, f"Expected 4 dimensions, got {x_l.ndim}"
        assert x_l.shape[0] == 1, f"Expected batch size 1, got {x_l.shape[0]}"
        assert (
            x_l.shape[1] == 1
        ), f"Expected N_sample 1, got {x_l.shape[1]}"  # N_sample=1 was passed
        assert (
            x_l.shape[2] == 5
        ), f"Expected 5 atoms, got {x_l.shape[2]}"  # Check number of atoms matches
        assert (
            x_l.shape[3] == 3
        ), f"Expected 3 coordinates, got {x_l.shape[3]}"  # Check coordinate dimension

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        n_sample=st.integers(min_value=1, max_value=5),
        chunk=st.one_of(st.none(), st.integers(min_value=1, max_value=5)),
    )
    def test_fuzz_sample_diffusion(
        self, mock_denoise_net, n_sample: int, chunk: Optional[int]
    ) -> None:
        """
        Hypothesis test for sample_diffusion with random chunk sizes and sample counts.
        This helps ensure coverage of edge parameters.
        """
        # Create a basic input feature dictionary
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros((1, 5), dtype=torch.long),
            "ref_pos": torch.randn(1, 5, 3),
            "ref_space_uid": torch.arange(5).unsqueeze(0),
            "ref_charge": torch.zeros(1, 5, 1),
            "ref_mask": torch.ones(1, 5, 1),
            "ref_element": torch.zeros(1, 5, 128),
            "ref_atom_name_chars": torch.zeros(1, 5, 256),
            "restype": torch.zeros(1, 5, 32),
            "profile": torch.zeros(1, 5, 32),
            "deletion_mean": torch.zeros(1, 5, 1),
            "sing": torch.randn(1, 5, 449),  # Required for s_inputs fallback
        }

        # Create a simple noise schedule
        noise_schedule = torch.tensor([10.0, 5.0, 0.0], dtype=torch.float32)

        # Create input tensors
        s_inputs = torch.randn(1, 5, 449)
        s_trunk = torch.randn(1, 5, 384)
        z_trunk = torch.randn(1, 5, 5, 32)

        # Call sample_diffusion with the test parameters
        x_l = sample_diffusion(
            denoise_net=mock_denoise_net,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            noise_schedule=noise_schedule,
            N_sample=n_sample,
            diffusion_chunk_size=chunk,
        )

        # Verify the output shape
        assert isinstance(x_l, torch.Tensor), "Output should be a tensor"
        assert x_l.ndim == 4, f"Expected 4 dimensions, got {x_l.ndim}"
        assert x_l.shape[0] == 1, f"Expected batch size 1, got {x_l.shape[0]}"
        assert x_l.shape[1] == n_sample, f"Expected N_sample {n_sample}, got {x_l.shape[1]}"
        assert x_l.shape[2] == 5, f"Expected 5 atoms, got {x_l.shape[2]}"
        assert x_l.shape[3] == 3, f"Expected 3 coordinates, got {x_l.shape[3]}"

        # Verify the output values are finite
        assert torch.isfinite(x_l).all(), "Output values must be finite"


class TestSampleDiffusionTraining:
    """
    Test suite for the sample_diffusion_training function.
    """

    @pytest.fixture
    def label_dict_fixture(self) -> Dict[str, Any]:
        """
        Provides a sample label_dict with 'coordinate' and 'coordinate_mask'.
        """
        coords = torch.zeros((1, 4, 3), dtype=torch.float32)  # batch=1, 4 atoms
        mask = torch.ones((1, 4), dtype=torch.bool)
        return {"coordinate": coords, "coordinate_mask": mask}

    def test_sample_diffusion_training_no_chunk(
        self, mock_denoise_net, label_dict_fixture
    ) -> None:
        """
        Test sample_diffusion_training with no chunking to verify shapes.
        """
        noise_sampler = TrainingNoiseSampler()
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros((1, 4), dtype=torch.long)
        }
        s_inputs = torch.randn(1, 4, 8)
        s_trunk = torch.randn(1, 4, 16)
        z_trunk = torch.randn(1, 4, 4, 16)

        x_gt_aug, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=mock_denoise_net,
            label_dict=label_dict_fixture,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            N_sample=3,
            device=torch.device("cpu")
        )
        # x_gt_aug => [1, 3, 4, 3]
        # x_denoised => [1, 3, 4, 3]
        # sigma => [1, 3]
        # Expected shapes:
        # x_gt_aug: [batch, N_sample, n_atoms, 3] -> [1, 3, 4, 3]
        # x_denoised: [batch, N_sample, n_atoms, 3] -> [1, 3, 4, 3]
        # sigma: [batch, N_sample] -> [1, 3]
        assert x_gt_aug.shape == (1, 3, 4, 3), f"Expected x_gt_aug shape (1, 3, 4, 3), got {x_gt_aug.shape}"
        assert x_denoised.shape == (1, 3, 4, 3), f"Expected x_denoised shape (1, 3, 4, 3), got {x_denoised.shape}"
        assert sigma.shape == (1, 3), f"Expected sigma shape (1, 3), got {sigma.shape}"

    def test_sample_diffusion_training_with_chunk(
        self, mock_denoise_net, label_dict_fixture
    ) -> None:
        """
        Test sample_diffusion_training with chunking for coverage.
        """
        noise_sampler = TrainingNoiseSampler()
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros((1, 4), dtype=torch.long)
        }
        s_inputs = torch.randn(1, 4, 8)
        s_trunk = torch.randn(1, 4, 16)
        z_trunk = torch.randn(1, 4, 4, 16)

        x_gt_aug, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=mock_denoise_net,
            label_dict=label_dict_fixture,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            N_sample=5,
            diffusion_chunk_size=2,
            device=torch.device("cpu")
        )
        # Expected shapes:
        # x_gt_aug: [batch, N_sample, n_atoms, 3] -> [1, 5, 4, 3]
        # x_denoised: [batch, N_sample, n_atoms, 3] -> [1, 5, 4, 3]
        # sigma: [batch, N_sample] -> [1, 5]
        assert x_gt_aug.shape == (1, 5, 4, 3), f"Expected x_gt_aug shape (1, 5, 4, 3), got {x_gt_aug.shape}"
        assert x_denoised.shape == (1, 5, 4, 3), f"Expected x_denoised shape (1, 5, 4, 3), got {x_denoised.shape}"
        assert sigma.shape == (1, 5), f"Expected sigma shape (1, 5), got {sigma.shape}"

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        n_sample=st.integers(min_value=1, max_value=5),
        chunk=st.one_of(st.none(), st.integers(min_value=1, max_value=5)),
    )
    def test_fuzz_sample_diffusion_training(
        self, mock_denoise_net, n_sample: int, chunk: Optional[int]
    ) -> None:
        """
        Hypothesis-based fuzz test of sample_diffusion_training with random chunk sizes and sample counts.
        """
        noise_sampler = TrainingNoiseSampler()
        label_dict = {
            "coordinate": torch.zeros((1, 3, 3), dtype=torch.float32),
            "coordinate_mask": torch.ones((1, 3), dtype=torch.bool),
        }
        input_feature_dict = {
            "atom_to_token_idx": torch.zeros((1, 3), dtype=torch.long)
        }
        s_inputs = torch.randn(1, 3, 4)
        s_trunk = torch.randn(1, 3, 8)
        z_trunk = torch.randn(1, 3, 3, 8)

        x_gt_aug, x_denoised, sigma = sample_diffusion_training(
            noise_sampler=noise_sampler,
            denoise_net=mock_denoise_net,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            N_sample=n_sample,
            diffusion_chunk_size=chunk,
            device=torch.device("cpu")
        )
        assert x_gt_aug.shape == (
            1,
            n_sample,
            3,
            3,
        ), "Fuzz: ground-truth augmented shape mismatch"
        assert x_denoised.shape == (1, n_sample, 3, 3), "Fuzz: denoised shape mismatch"
        assert sigma.shape == (1, n_sample), "Fuzz: sigma shape mismatch"
        # Validate finite
        assert torch.isfinite(x_gt_aug).all(), "Fuzz: x_gt_aug must be finite"
        assert torch.isfinite(x_denoised).all(), "Fuzz: x_denoised must be finite"
        assert torch.isfinite(sigma).all(), "Fuzz: sigma must be finite"
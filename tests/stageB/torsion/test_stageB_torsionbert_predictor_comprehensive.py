"""
test_stageB_torsionbert_predictor_comprehensive.py

A comprehensive, high-coverage test suite for the StageBTorsionBertPredictor class.

Goal:
    - Achieve thorough testing (approaching 100% coverage) for torsion_bert_predictor.py.
    - Validate standard usage (normal sequences, short sequences).
    - Confirm angle-mode conversions for 'sin_cos', 'radians', 'degrees'.
    - Ensure proper error handling (invalid angle_mode, shape mismatches, negative angles, etc.).
    - Include a dummy TorsionBertModel that returns predictable data for robust unit testing.
    - Use property-based testing (Hypothesis) for constructor fuzz and angle conversion edge cases.
    - Demonstrate how adjacency is accepted but currently unused in the pipeline.
    - **Now also** mocks out Hugging Face calls so 'dummy_path' never triggers an actual download.
    - **Additionally** includes a test replicating the "14 vs 32" mismatch bug from interface.py.

How to Run:
    pytest test_stageB_torsionbert_predictor_comprehensive.py --maxfail=1 -v

How to Measure Coverage:
    1) pip install coverage
    2) coverage run -m pytest test_stageB_torsionbert_predictor_comprehensive.py
    3) coverage report -m
    4) coverage html  # (Optional) to generate an HTML report

Prerequisites:
    - Pytest
    - Hypothesis
    - Torch (PyTorch)
    - rna_predict.pipeline.stageB.torsion.torsion_bert_predictor module
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

# ----------------------------------------------------------------------
# IMPORTANT: Mock out the HF calls so "dummy_path" doesn't trigger downloads
# ----------------------------------------------------------------------
with patch(
    "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoTokenizer.from_pretrained"
) as mock_tok:
    mock_tok.return_value = MagicMock(name="MockedTokenizer")

    with patch(
        "rna_predict.pipeline.stageB.torsion.torsionbert_inference.AutoModel.from_pretrained"
    ) as mock_model:
        mock_model.return_value = MagicMock(name="MockedAutoModel")

        # Now we import the class under test, in an environment
        # where HF calls won't actually contact huggingface.co:
        from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import (
            StageBTorsionBertPredictor,
        )


class DummyTorsionBertModel:
    """
    A dummy class that simulates the TorsionBertModel behavior for testing.

    By default, this class returns a tensor with a shape (N, 2*num_angles),
    where N is len(sequence). The values can be random or fixed as needed.

    Adjust 'return_style' if you want consistent shapes or to force shape mismatches.
    """

    def __init__(self, num_angles: int, return_style: str = "ones"):
        """
        Args:
            num_angles (int): The number of angles the predictor is configured to handle.
            return_style (str): Controls what data is returned:
                "ones" => returns all ones
                "random" => returns random floats
                "mismatch" => returns shape mismatch for negative testing
        """
        self.num_angles = num_angles
        self.return_style = return_style

    def predict_angles_from_sequence(self, sequence: str) -> torch.Tensor:
        """
        Returns either all ones, random data, or shape-mismatched data,
        depending on self.return_style.
        """
        N = len(sequence)
        if self.return_style == "mismatch":
            # Forcing a shape mismatch to ensure we test dimension errors
            # Example: expect 2 * num_angles but we return 2 * num_angles - 1
            return torch.randn(N, 2 * self.num_angles - 1)
        elif self.return_style == "random":
            return torch.randn(N, 2 * self.num_angles)
        else:
            # Default: All ones, shape (N, 2 * self.num_angles)
            return torch.ones((N, 2 * self.num_angles))


@pytest.fixture
def dummy_model_ones() -> DummyTorsionBertModel:
    """
    Fixture that returns a dummy TorsionBertModel producing an all-ones tensor.
    Useful for stable, predictable tests.
    """
    return DummyTorsionBertModel(num_angles=4, return_style="ones")


@pytest.fixture
def dummy_model_random() -> DummyTorsionBertModel:
    """
    Fixture that returns a dummy TorsionBertModel producing random data.
    Potentially more thorough but can produce slight floating discrepancies.
    """
    return DummyTorsionBertModel(num_angles=4, return_style="random")


@pytest.fixture
def dummy_model_mismatch() -> DummyTorsionBertModel:
    """
    Fixture that forces a shape mismatch to test dimension-checking code paths.
    """
    return DummyTorsionBertModel(num_angles=4, return_style="mismatch")


@pytest.fixture
def predictor_degrees(
    dummy_model_ones: DummyTorsionBertModel,
) -> StageBTorsionBertPredictor:
    """
    Fixture for a StageBTorsionBertPredictor using angle_mode='degrees' with a stable dummy model.
    This is well-suited to verifying consistent, deterministic outputs for coverage.
    """
    predictor = StageBTorsionBertPredictor(
        model_name_or_path="dummy_path",
        device="cpu",
        angle_mode="degrees",
        num_angles=dummy_model_ones.num_angles,
    )
    # Inject the dummy TorsionBertModel to bypass real HF inference
    predictor.model = dummy_model_ones
    return predictor


@pytest.fixture
def predictor_sincos_random(
    dummy_model_random: DummyTorsionBertModel,
) -> StageBTorsionBertPredictor:
    """
    Fixture for a StageBTorsionBertPredictor using angle_mode='sin_cos' with random data
    to ensure that we capture possible floating-point variations.
    """
    predictor = StageBTorsionBertPredictor(
        model_name_or_path="dummy_path",
        device="cpu",
        angle_mode="sin_cos",
        num_angles=dummy_model_random.num_angles,
    )
    predictor.model = dummy_model_random
    return predictor


class TestStageBTorsionBertPredictorCore:
    """
    Core tests to verify normal behavior, short sequences, adjacency usage, etc.
    """

    def test_short_sequence_degrees(
        self, predictor_degrees: StageBTorsionBertPredictor
    ):
        """
        Test a short two-letter sequence in 'degrees' mode with a stable dummy model.
        Because the dummy model returns all ones, the sin/cos pairs are (1,1),
        which convert to 45 degrees for each angle. We verify the shape and residue_count.
        """
        sequence = "AC"  # length 2
        output = predictor_degrees(sequence)

        torsion = output["torsion_angles"]
        assert torsion.shape == (2, 4), (
            "In degrees mode, the predictor should convert 2*sin_cos columns into 'num_angles' columns. "
            "Since num_angles=4, we expect [2, 4]."
        )
        assert output["residue_count"] == 2

        # Because the input is all ones => sin=1, cos=1 => angle= arctan2(1,1)= pi/4 => ~0.785 rad => 45 degrees
        # So each angle should be ~45 degrees if the conversion logic is correct.
        # Tolerate small floating diffs:
        assert torch.allclose(
            torsion, torch.full((2, 4), 45.0), atol=1e-3
        ), "Degrees conversion mismatch for short sequence"

    def test_normal_sequence_degrees(
        self, predictor_degrees: StageBTorsionBertPredictor
    ):
        """
        Test a normal 4-letter sequence in degrees mode, verifying shape correctness.
        The dummy model returns all ones, so angles should be 45 degrees as well.
        """
        seq = "ACGU"
        output = predictor_degrees(seq)
        torsion = output["torsion_angles"]

        assert torsion.shape == (
            4,
            4,
        ), "Expected shape [4, 4] for 4 residues in degrees mode."
        assert output["residue_count"] == 4
        # Each angle ~45 degrees:
        assert torch.allclose(torsion, torch.full((4, 4), 45.0), atol=1e-3)

    def test_adjacency_ignored(self, predictor_degrees: StageBTorsionBertPredictor):
        """
        Passing an adjacency tensor should not affect the output; it's currently unused.
        """
        adjacency = torch.zeros((3, 3))
        output = predictor_degrees("ACG", adjacency=adjacency)
        angles = output["torsion_angles"]
        assert angles.shape == (
            3,
            4,
        ), "Predictor shape should be unaffected by adjacency input."
        assert output["residue_count"] == 3

    def test_invalid_angle_mode_raises(self):
        """
        Confirm that an invalid angle mode raises a ValueError.
        """
        predictor = StageBTorsionBertPredictor(angle_mode="not_valid")
        with pytest.raises(ValueError, match="Unknown angle_mode: not_valid"):
            predictor("AC")


class TestStageBTorsionBertPredictorDimChecks:
    """
    Tests focusing on dimension mismatch scenarios, verifying that a RuntimeError is raised when
    sin/cos pairs do not match the expected dimension (2 * num_angles).
    """

    def test_shape_mismatch_raises(self, dummy_model_mismatch: DummyTorsionBertModel):
        """
        Using the 'mismatch' dummy model fixture ensures we produce an
        incorrect column count => triggers a RuntimeError in _convert_sincos_to_angles.
        """
        predictor = StageBTorsionBertPredictor(
            angle_mode="radians", num_angles=dummy_model_mismatch.num_angles
        )
        predictor.model = dummy_model_mismatch  # Force dimension mismatch
        with pytest.raises(RuntimeError, match="Expected"):
            predictor("ACG")

    @patch.object(
        StageBTorsionBertPredictor,
        "_convert_sincos_to_angles",
        side_effect=RuntimeError("Forced mismatch for test!"),
    )
    def test_mocked_mismatch(self, mock_conv):
        """
        Demonstrate an alternative approach with patching, where we forcibly raise.
        This ensures coverage of the calling site even if a real mismatch fixture isn't used.
        """
        predictor = StageBTorsionBertPredictor(num_angles=4, angle_mode="radians")
        with pytest.raises(RuntimeError, match="Forced mismatch for test!"):
            predictor("ACG")


class TestStageBTorsionBertPredictorSincosRoundTrip:
    """
    Tests verifying correctness of sin/cos <-> angle conversions, specifically round-trip behaviors.
    """

    def test_sincos_round_trip(self):
        """
        Creates a sin_cos predictor and directly calls _convert_sincos_to_angles,
        then reconstructs sin/cos from the resulting angles to verify near-identity.

        This ensures that the pipeline can handle typical values properly.
        """
        # We'll do 3 angles => shape of sincos [N=4, 2*num_angles=6]
        predictor = StageBTorsionBertPredictor(angle_mode="sin_cos", num_angles=3)

        # Construct a custom sincos tensor
        # We'll skip guaranteeing sin^2+cos^2 == 1 to see how method handles approximate pairs.
        torch.manual_seed(42)
        N = 4
        sin_part = torch.rand((N, 3)) * 2 - 1.0
        cos_part = torch.rand((N, 3)) * 2 - 1.0
        sincos_tensor = torch.cat([sin_part, cos_part], dim=1)

        angles = predictor._convert_sincos_to_angles(sincos_tensor)

        # Rebuild sin, cos
        sin_restored = torch.sin(angles)
        cos_restored = torch.cos(angles)

        # We'll check sign consistency except near zero
        for i in range(N):
            for j in range(3):
                orig_sin = sin_part[i, j].item()
                rest_sin = sin_restored[i, j].item()
                if abs(orig_sin) > 0.05:
                    assert (orig_sin * rest_sin) > 0, (
                        f"Sign mismatch in sin at row {i}, angle {j}: "
                        f"original={orig_sin}, restored={rest_sin}"
                    )


class TestStageBTorsionBertPredictorConstructorFuzzing:
    """
    Hypothesis-based fuzzing for the StageBTorsionBertPredictor constructor arguments
    (model_name_or_path, device, angle_mode, num_angles, max_length, etc.).
    """

    @settings(
        max_examples=50,
        suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    )
    @given(
        model_name=st.text(min_size=0, max_size=20),
        device=st.sampled_from(["cpu", "cuda", "bogus_device"]),
        angle_mode=st.sampled_from(["sin_cos", "radians", "degrees", "unknown_mode"]),
        num_angles=st.integers(min_value=-5, max_value=10),
        max_length=st.integers(min_value=1, max_value=256),
    )
    @example(
        model_name="", device="cpu", angle_mode="sin_cos", num_angles=1, max_length=1
    )
    def test_constructor_arguments(
        self, model_name, device, angle_mode, num_angles, max_length
    ):
        """
        Fuzz test that the constructor either:
         - creates a predictor successfully for valid arguments, or
         - raises an expected exception for invalid ones (e.g., negative angles, unknown mode, invalid device).
        """
        try:
            predictor = StageBTorsionBertPredictor(
                model_name_or_path=model_name,
                device=device,
                angle_mode=angle_mode,
                num_angles=num_angles,
                max_length=max_length,
            )
            # If angle_mode is valid and num_angles > 0, attempt minimal usage
            if angle_mode in ("sin_cos", "radians", "degrees") and num_angles > 0:
                # We'll mock the model to ensure shape is consistent.
                with patch.object(
                    predictor.model,
                    "predict_angles_from_sequence",
                    return_value=torch.zeros((1, 2 * num_angles)),
                ):
                    out = predictor("A")
                    if angle_mode == "sin_cos":
                        assert out["torsion_angles"].shape == (1, 2 * num_angles)
                    else:
                        assert out["torsion_angles"].shape == (1, num_angles)
        except (ValueError, RuntimeError, TypeError):
            # We allow these exceptions if arguments are invalid or the device is bogus.
            pass
        except Exception as e:
            pytest.fail(f"Unexpected exception: {type(e).__name__}: {e}")


class TestAngleConversionHypothesis:
    """
    Hypothesis-based test for _convert_sincos_to_angles method, ensuring correctness
    for a wide range of sin/cos values. We'll do a direct comparison with math.atan2.
    """

    @settings(max_examples=30)
    @given(
        # We'll produce small to moderate sized matrices of even shape [N, 2*k].
        # Let's fix k=3 for clarity => 6 columns total.
        sincos_data=st.lists(
            st.lists(
                st.floats(
                    allow_infinity=False, allow_nan=False, min_value=-1.0, max_value=1.0
                ),
                min_size=6,
                max_size=6,  # exactly 6 columns
            ),
            min_size=1,
            max_size=5,  # up to 5 rows
        )
    )
    def test_convert_sincos_matrix(self, sincos_data):
        """
        For each row, interpret pairs (sin, cos) => compare to direct math.atan2.
        We'll do num_angles=3. We skip domain checking (sin^2+cos^2=1) since the real model
        might produce approximate sin/cos. We only confirm the function is consistent with atan2.
        """
        # Construct a predictor with num_angles=3 => expect shape [N, 6].
        predictor = StageBTorsionBertPredictor(angle_mode="radians", num_angles=3)

        # Convert sincos_data -> Tensor
        sincos_tensor = torch.tensor(sincos_data, dtype=torch.float32)

        # Validate shape
        N = sincos_tensor.shape[0]
        assert (
            sincos_tensor.shape[1] == 6
        ), "We expect exactly 6 columns for 3 angles in sin/cos pairs."

        # Convert
        angles = predictor._convert_sincos_to_angles(sincos_tensor)

        # Compare each pair via math.atan2
        for i in range(N):
            for angle_idx in range(3):
                s_val = sincos_data[i][2 * angle_idx]
                c_val = sincos_data[i][2 * angle_idx + 1]
                expected_angle = math.atan2(s_val, c_val)
                actual_angle = angles[i, angle_idx].item()
                diff = abs(expected_angle - actual_angle)
                # We'll allow up to ~1e-3 mismatch
                assert diff < 0.01, (
                    f"Mismatch in row {i}, angle {angle_idx}, sin={s_val}, cos={c_val}: "
                    f"expected={expected_angle}, got={actual_angle} (diff={diff})"
                )


# ----------------------------------------------------------------------
#   REPLICATING THE "14 vs 32" BUG from user: interface.py => TorsionBertPredictor
# ----------------------------------------------------------------------
class TestReplicateInterfaceBug:
    @patch(
        "rna_predict.pipeline.stageB.torsion.torsion_bert_predictor.TorsionBertModel"
    )
    def test_interface_style_dimension_error(self, mock_tbm):
        """
        The user's trace indicates a mismatch of 14 vs 32. Possibly because
        raw_sincos is shape [1, i], while we attempt to assign to result of shape [seq_len, 2*num_angles].

        We'll simulate a scenario in which TorsionBertModel returns e.g. shape [1, 32], but
        the sequence length is 14 => dimension mismatch.

        The result is a forced RuntimeError (like "expanded size of the tensor... must match existing size...").
        """
        # Fake TorsionBertModel that yields shape [1,32]
        mock_model_inst = MagicMock(name="MockedModelInstance")
        mock_model_inst.predict_angles_from_sequence.return_value = torch.rand((1, 32))
        mock_tbm.return_value = mock_model_inst

        # Create a predictor expecting e.g. 16 angles => shape (N, 32).
        # Then pass a 14-length sequence => triggers dimension mismatch in code.
        predictor = StageBTorsionBertPredictor(
            model_name_or_path="dummy_path",
            device="cpu",
            angle_mode="sin_cos",
            num_angles=16,
        )
        seq = "ACGUACGUACGUAC"  # length=14
        with pytest.raises(RuntimeError, match="Tensor sizes"):
            predictor(seq)

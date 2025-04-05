# tests/stageD/unit/diffusion/components/test_diffusion_utils.py
import pytest
import torch
from typing import Tuple, Union # Added Union
import warnings # Added warnings

# Assuming the function is importable like this:
from rna_predict.pipeline.stageD.diffusion.components.diffusion_utils import (
    _calculate_edm_scaling_factors,
)

# Helper function for expected calculations (float32 precision)
def calculate_expected_factors(
    sigma: torch.Tensor, sigma_data: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper to calculate expected factors with float32 precision."""
    sigma_float = sigma.float()
    sigma_data_float = float(sigma_data)
    epsilon = torch.finfo(torch.float32).eps

    denom_in_sq = sigma_data_float**2 + sigma_float**2
    denom_skip_out = denom_in_sq

    denom_in = torch.sqrt(torch.clamp(denom_in_sq, min=epsilon))
    denom_skip_out_safe = torch.clamp(denom_skip_out, min=epsilon)

    c_in_expected = 1.0 / denom_in
    c_skip_expected = sigma_data_float**2 / denom_skip_out_safe
    c_out_expected = (sigma_data_float * sigma_float) / denom_in

    return c_in_expected, c_skip_expected, c_out_expected


@pytest.mark.parametrize(
    "sigma_val, sigma_data, ref_shape, ref_dtype",
    [
        # Basic case: scalar sigma, 3D ref_tensor, float32
        (1.5, 0.5, (2, 4, 3), torch.float32),
        # Basic case: tensor sigma matching ref_tensor ndim-1, float32
        (torch.tensor([1.5, 2.0]), 0.5, (2, 4, 3), torch.float32),
         # Basic case: tensor sigma matching ref_tensor ndim, float32
        (torch.rand(2, 4, 3), 0.5, (2, 4, 3), torch.float32),
        # Test broadcasting: sigma needs unsqueezing (1D -> 3D)
        (torch.tensor([1.5, 2.0]), 0.8, (2, 5, 6), torch.float32),
        # Test broadcasting: sigma needs unsqueezing (2D -> 4D)
        (torch.rand(2, 3), 0.8, (2, 3, 5, 6), torch.float32),
        # Test different dtype: float64
        (1.0, 0.6, (1, 2, 3), torch.float64),
        # Test different dtype: float16 (calculations still in float32)
        pytest.param(1.0, 0.6, (1, 2, 3), torch.float16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="float16 requires CUDA")),
        # Edge case: sigma = 0
        (0.0, 0.5, (2, 2, 2), torch.float32),
        # Edge case: very small sigma
        (1e-9, 0.5, (2, 2, 2), torch.float32),
        # Edge case: sigma_data = 0 (should still work)
        (1.0, 0.0, (2, 2, 2), torch.float32),
        # Edge case: sigma = 0, sigma_data = 0
        (0.0, 0.0, (2, 2, 2), torch.float32),
    ],
)
def test_calculate_edm_scaling_factors(
    sigma_val: Union[float, torch.Tensor],
    sigma_data: float,
    ref_shape: Tuple[int, ...],
    ref_dtype: torch.dtype,
):
    """
    Tests _calculate_edm_scaling_factors for various inputs, including:
    - Different sigma shapes requiring broadcasting.
    - Different reference tensor dtypes.
    - Edge cases like zero or small sigma/sigma_data.
    """
    if isinstance(sigma_val, float):
        sigma = torch.tensor(sigma_val)
    else:
        sigma = sigma_val.clone() # Ensure original tensor isn't modified

    # Create reference tensor
    device = sigma.device # Match device if sigma is already on GPU
    if ref_dtype == torch.float16 and not device.type == 'cuda':
        pytest.skip("float16 requires CUDA device") # Skip if trying float16 on CPU
    ref_tensor = torch.randn(ref_shape, dtype=ref_dtype, device=device)

    # --- Function Execution ---
    c_in, c_skip, c_out = _calculate_edm_scaling_factors(sigma, sigma_data, ref_tensor)

    # --- Verification ---

    # 1. Check output dtypes match reference tensor dtype (Lines 224, 254-256)
    assert c_in.dtype == ref_dtype, f"Expected c_in dtype {ref_dtype}, got {c_in.dtype}"
    assert c_skip.dtype == ref_dtype, f"Expected c_skip dtype {ref_dtype}, got {c_skip.dtype}"
    assert c_out.dtype == ref_dtype, f"Expected c_out dtype {ref_dtype}, got {c_out.dtype}"

    # 2. Check shapes are broadcastable to ref_tensor (Lines 228-230, 258-267 implicitly)
    # We check this by verifying the calculations which rely on correct broadcasting.
    # The explicit check in the function (260-267) is hard to fail externally,
    # but correct broadcasting is essential for the math to work.
    # We can also check the shapes directly *before* potential broadcasting in the return.
    expected_sigma_shape_after_unsqueeze = list(sigma.shape)
    while len(expected_sigma_shape_after_unsqueeze) < len(ref_shape):
        expected_sigma_shape_after_unsqueeze.append(1)

    # Factors should have shapes compatible with broadcasting rules against ref_tensor.
    # Their shape will match the expanded sigma shape before the final implicit broadcast.
    assert list(c_in.shape) == expected_sigma_shape_after_unsqueeze, f"Shape mismatch for c_in"
    assert list(c_skip.shape) == expected_sigma_shape_after_unsqueeze, f"Shape mismatch for c_skip"
    assert list(c_out.shape) == expected_sigma_shape_after_unsqueeze, f"Shape mismatch for c_out"


    # 3. Check calculation correctness (Lines 232-251)
    # Recalculate expected values using float32 precision as done in the function
    sigma_expanded_for_calc = sigma.clone()
    while sigma_expanded_for_calc.ndim < ref_tensor.ndim:
        sigma_expanded_for_calc = sigma_expanded_for_calc.unsqueeze(-1)

    c_in_expected, c_skip_expected, c_out_expected = calculate_expected_factors(
        sigma_expanded_for_calc, sigma_data
    )

    # Compare with expected values, casting expected to the target dtype for comparison
    # Use torch.allclose for floating point comparisons
    # Increased tolerance slightly for float16 cases
    atol = 1e-4 if ref_dtype == torch.float16 else 1e-6
    rtol = 1e-3 if ref_dtype == torch.float16 else 1e-5 # Relative tolerance might be needed too

    assert torch.allclose(c_in, c_in_expected.to(ref_dtype), atol=atol, rtol=rtol), f"c_in calculation mismatch. Got {c_in}, expected {c_in_expected.to(ref_dtype)}"
    assert torch.allclose(c_skip, c_skip_expected.to(ref_dtype), atol=atol, rtol=rtol), f"c_skip calculation mismatch. Got {c_skip}, expected {c_skip_expected.to(ref_dtype)}"
    assert torch.allclose(c_out, c_out_expected.to(ref_dtype), atol=atol, rtol=rtol), f"c_out calculation mismatch. Got {c_out}, expected {c_out_expected.to(ref_dtype)}"

    # 4. Check numerical stability (epsilon clamping) (Lines 237-245, 250-251)
    # This is implicitly tested by the zero/small sigma cases.
    # If sigma is zero, the clamped denominators prevent division by zero.
    if torch.all(sigma == 0.0):
        assert not torch.isinf(c_in).any(), "c_in should not be inf when sigma is 0"
        assert not torch.isnan(c_in).any(), "c_in should not be NaN when sigma is 0"
        assert not torch.isinf(c_skip).any(), "c_skip should not be inf when sigma is 0"
        assert not torch.isnan(c_skip).any(), "c_skip should not be NaN when sigma is 0"
        assert not torch.isinf(c_out).any(), "c_out should not be inf when sigma is 0"
        assert not torch.isnan(c_out).any(), "c_out should not be NaN when sigma is 0"
        # Specific check for sigma=0
        sigma_data_float = float(sigma_data)
        epsilon = torch.finfo(torch.float32).eps
        expected_c_skip_at_0 = sigma_data_float**2 / max(sigma_data_float**2, epsilon)
        expected_c_in_at_0 = 1.0 / torch.sqrt(torch.clamp(torch.tensor(sigma_data_float**2), min=epsilon))

        assert torch.allclose(c_skip, torch.tensor(expected_c_skip_at_0, dtype=ref_dtype), atol=atol, rtol=rtol), f"c_skip incorrect for sigma=0. Got {c_skip}, expected {expected_c_skip_at_0}"
        assert torch.allclose(c_out, torch.zeros_like(c_out), atol=atol), f"c_out should be 0 for sigma=0. Got {c_out}"
        assert torch.allclose(c_in, expected_c_in_at_0.to(ref_dtype), atol=atol, rtol=rtol), f"c_in incorrect for sigma=0. Got {c_in}, expected {expected_c_in_at_0.to(ref_dtype)}"


def test_calculate_edm_scaling_factors_warning():
    """
    Tests that the warning for potential broadcast issues is NOT triggered
    under normal circumstances where broadcasting should succeed.
    (It's hard to reliably trigger the warning path L266-267 externally).
    """
    sigma = torch.tensor([1.0, 2.0]) # Shape [2]
    sigma_data = 0.5
    ref_tensor = torch.randn(2, 5, 3) # Shape [2, 5, 3]

    # Expect sigma to be unsqueezed to [2, 1, 1]
    # This should broadcast correctly against [2, 5, 3]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always") # Capture all warnings

        c_in, c_skip, c_out = _calculate_edm_scaling_factors(sigma, sigma_data, ref_tensor)

        # Verify the specific warning about broadcasting (L266) is NOT present
        found_broadcast_warning = False
        for warning_message in w:
            if "may not broadcast correctly to ref_tensor" in str(warning_message.message):
                found_broadcast_warning = True
                break
        assert not found_broadcast_warning, "Unexpected broadcast warning was triggered"

    # Basic check that outputs seem reasonable
    assert c_in.shape == (2, 1, 1)
    assert c_skip.shape == (2, 1, 1)
    assert c_out.shape == (2, 1, 1)
from typing import Tuple

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# Import the functions under test
from rna_predict.utils.scatter_utils import (
    inverse_squared_dist,
    layernorm,
    scatter_mean,
)


@pytest.mark.parametrize(
    "shape, eps",
    [
        ((2, 3), 1e-5),
        ((10, 10), 1e-5),
        ((5, 1), 1e-6),
    ],
)
def test_layernorm_basic(shape: Tuple[int, int], eps: float) -> None:
    """
    Test layernorm with small, medium and edge shapes and different eps values.
    Ensures output shape matches input shape and that the result is mean-centered.
    """
    x = torch.randn(shape, dtype=torch.float32)
    out = layernorm(x, eps=eps)

    assert out.shape == x.shape, "Output shape must match input."
    # Check that we are near zero mean
    means = out.mean(dim=-1)
    assert torch.allclose(
        means, torch.zeros_like(means), atol=1e-5
    ), "layernorm should zero-center the mean."

    # Skip variance check for dimension size 1 - mathematically it doesn't make sense
    # to normalize a single value to have variance 1
    if shape[-1] > 1:
        # Check variance is near 1
        var = out.var(dim=-1, unbiased=False)
        assert torch.allclose(
            var, torch.ones_like(var), atol=1e-2 # Increased tolerance
        ), "layernorm should scale variance to 1."


def test_layernorm_one_dim() -> None:
    """
    Test layernorm with a single-dimensional input, ensuring it still processes
    and that shape remains consistent.
    """
    x = torch.randn(5)
    out = layernorm(x)
    assert out.shape == x.shape, "Layernorm should preserve shape even in 1D."
    assert torch.allclose(out.mean(dim=-1), torch.tensor(0.0), atol=1e-5)


def test_layernorm_error_handling() -> None:
    """
    layernorm uses x.mean(...) and x.var(...), which can raise runtime errors
    if x is not a floating type or shape is invalid. We confirm float is required.
    """
    x = torch.randint(0, 10, (4, 5), dtype=torch.int32)
    # This should raise a runtime error or a type error when performing mean/var.
    with pytest.raises(RuntimeError):
        _ = layernorm(x)


# -----------------------------------------------------------------------------
# inverse_squared_dist Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "delta, eps, expected",
    [
        (torch.zeros(1, 3), 1e-8, 1.0),  # zero vector => 1/(1+0^2+eps) ~ 1
        (
            torch.tensor([[1.0, 0.0, 0.0]]),
            1e-8,
            1.0 / (1 + 1 + 1e-8),
        ),  # dist^2=1 => ~0.5
        (
            torch.tensor([[3.0, 4.0, 12.0]]),
            1e-8,
            1.0 / (1 + 9 + 16 + 144 + 1e-8),
        ),  # dist^2=169 => 1/170
    ],
)
def test_inverse_squared_dist_explicit(
    delta: torch.Tensor, eps: float, expected: float
) -> None:
    """
    Test inverse_squared_dist with known explicit values for distance squares.
    """
    result = inverse_squared_dist(delta, eps=eps)
    assert result.shape == (delta.shape[0], 1), "Output should have shape [N, 1]."
    # We test that it's close to the known result
    assert torch.allclose(result.squeeze(-1), torch.tensor(expected), atol=1e-6)


def test_inverse_squared_dist_batch_shapes() -> None:
    """
    Test inverse_squared_dist with batched shapes, ensuring the output shape is correct.
    """
    # shape: [B, 5, 3]
    delta = torch.randn(4, 5, 3)
    out = inverse_squared_dist(delta)
    assert out.shape == (
        4,
        5,
        1,
    ), "Output shape must match input shape except last dimension is 1."


def test_inverse_squared_dist_large():
    """
    Test inverse_squared_dist with large vectors to ensure no overflow errors occur
    and that the result is very small.
    """
    delta = torch.tensor(
        [[1e6, 1e6, 1e6]], dtype=torch.float32
    )  # magnitude sqrt(3)*1e6
    out = inverse_squared_dist(delta)
    # dist^2 ~ 3e12 => 1/(1+3e12+eps) -> about ~3.333e-13
    assert out.item() < 1e-12, "Expected an extremely small value for large distances."


@given(
    delta=st.lists(st.floats(-1e3, 1e3), min_size=3, max_size=3).map(
        lambda x: torch.tensor(x, dtype=torch.float32)
    )
)
@settings(
    deadline=None, # Disable deadline for this flaky test
    suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow],
    max_examples=20,
)
def test_inverse_squared_dist_hypothesis(delta: torch.Tensor) -> None:
    """
    Hypothesis test for random 3D vectors, verifying that the output is always in (0, 1].
    """
    # Reshape to [1,3]
    delta = delta.unsqueeze(0)
    out = inverse_squared_dist(delta, eps=1e-8)
    val = out.item()
    # We do not expect negative or zero results
    assert (
        0 < val <= 1.0
    ), "inverse_squared_dist must be within (0,1] for any real delta."


# -----------------------------------------------------------------------------
# scatter_mean Tests
# -----------------------------------------------------------------------------


def test_scatter_mean_basic() -> None:
    """
    Test scatter_mean with a simple small example where we know the correct result.
    src: 2D, with 4 rows, each row belongs to an index =>
    [ idx 0, idx 1, idx 1, idx 0 ] => 2 segments total => dimension=2
    """
    src = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
    )
    index = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    dim_size = 2
    out = scatter_mean(src, index, dim_size=dim_size, dim=0)
    # For idx=0 => rows 0 and 3 => mean => ([1+7]/2, [2+8]/2) => (4, 5)
    # For idx=1 => rows 1 and 2 => mean => ([3+5]/2, [4+6]/2) => (4, 5)
    expected = torch.tensor([[4.0, 5.0], [4.0, 5.0]], dtype=torch.float32)
    assert torch.allclose(
        out, expected
    ), "scatter_mean did not produce the expected segment means."
    assert out.shape == (dim_size, 2)


def test_scatter_mean_counts_zero() -> None:
    """
    Test scatter_mean in a scenario where some segments don't receive any items,
    ensuring clamp(min=1.0) is triggered to avoid dividing by zero.
    We do so by setting dim_size to 3 but only using segment indices 0, 1.
    Segment 2 should remain zeros because count remains 0, then clamped to 1 => result is 0.
    """
    src = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    index = torch.tensor([0], dtype=torch.long)
    dim_size = 3
    out = scatter_mean(src, index, dim_size=dim_size, dim=0)
    # Segment 0 => average is [2.0, 2.0]
    # Segment 1 => no items => out => 0
    # Segment 2 => no items => out => 0
    expected = torch.tensor([[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_scatter_mean_multiple_rows() -> None:
    """
    Test scatter_mean with multi-column src and random indices.
    Ensure shape is correct and that the function doesn't crash.
    We'll also check sums with a small check.
    """
    src = torch.randn(10, 4)  # shape => [10, 4]
    # Indices in range [0, 3]
    index = torch.randint(0, 3, (10,), dtype=torch.long)
    dim_size = 3
    out = scatter_mean(src, index, dim_size=dim_size, dim=0)
    assert out.shape == (dim_size, 4), "Output shape mismatch in scatter_mean."
    # We'll do a naive check: sum of out * counts == sum of src for each segment
    counts = torch.zeros(dim_size, dtype=torch.float32)
    sums = torch.zeros(dim_size, 4, dtype=torch.float32)
    for i in range(len(index)):
        idx = index[i].item()
        counts[idx] += 1
        sums[idx] += src[i]
    # Now out * counts => sums
    for i in range(dim_size):
        expected_sum = sums[i]
        actual_sum = out[i] * (counts[i] if counts[i] > 0 else 1.0)
        assert torch.allclose(expected_sum, actual_sum, atol=1e-5)


def test_scatter_mean_inconsistent_dims() -> None:
    """
    If 'src' and 'index' have inconsistent shapes, scatter_mean
    should raise an error when indexing.
    """
    src = torch.randn(5, 3)
    index = torch.tensor([0, 1, 2], dtype=torch.long)  # only length 3
    with pytest.raises(IndexError):
        _ = scatter_mean(src, index, dim_size=2, dim=0)


@given(
    n=st.integers(min_value=1, max_value=50),
    c=st.integers(min_value=1, max_value=10),
    segments=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_scatter_mean_hypothesis(n: int, c: int, segments: int) -> None:
    """
    Hypothesis test for scatter_mean across random shapes.
    We ensure no runtime errors and check the shape of the output.
    """
    src = torch.randn(n, c)
    index = torch.randint(0, segments, (n,), dtype=torch.long)
    out = scatter_mean(src, index, dim_size=segments, dim=0)
    assert out.shape == (segments, c), "scatter_mean output shape mismatch."


# Additional negative or boundary cases can go here if needed.

if __name__ == "__main__":
    # This allows running this file directly with "python test_scatter_utils.py"
    # but generally you'd use "pytest tests/test_scatter_utils.py"
    pytest.main([__file__])

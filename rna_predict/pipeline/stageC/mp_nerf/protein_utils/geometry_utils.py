"""
Basic geometry utility functions.
"""

from typing import TypeVar, Union, overload

import numpy as np
import torch

# Define a type variable for numeric types
NumericType = TypeVar("NumericType", float, np.float64, np.float32)


@overload
def to_zero_two_pi(x: float) -> float: ...


@overload
def to_zero_two_pi(x: np.ndarray) -> np.ndarray: ...


@overload
def to_zero_two_pi(x: torch.Tensor) -> torch.Tensor: ...


def to_zero_two_pi(
    x: Union[float, np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    """Convert angle to [0, 2π] range.

    Args:
        x: Input angle(s) in radians

    Returns:
        Angle(s) in range [0, 2π]
    """
    # Correct logic: Use modulo 2*pi to wrap angles into the desired range.
    # The `+ 2 * np.pi` handles negative inputs correctly before the final modulo.
    two_pi = 2 * np.pi

    if isinstance(x, (float, int)):
        # Ensure input is treated as float for modulo operation
        x_float = float(x)
        return (x_float % two_pi + two_pi) % two_pi
    elif isinstance(x, np.ndarray):
        return (x % two_pi + two_pi) % two_pi
    elif isinstance(x, torch.Tensor):
        return (x % two_pi + two_pi) % two_pi
    else:
        raise TypeError(f"Unsupported type for to_zero_two_pi: {type(x)}")

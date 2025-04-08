"""
Linear primitives module for neural network operations.

This module contains linear layer implementations and transformations used in the neural
network architecture, focusing on basic building blocks like custom linear layers and
transition modules.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from protenix.openfold_local.model.primitives import LayerNorm
from torch.nn import Linear

# Create a version of Linear with no bias by default
LinearNoBias = partial(Linear, bias=False)


class BiasInitLinear(Linear):
    """Support biasinit for nn.Linear Called just like torch.nn.Linear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        biasinit: float = 0.0,
    ) -> None:
        """
        Args:
            in_features (int): in_features
            out_features (int): out_features
            bias (bool, optional): whether add bias. Defaults to True.
            biasinit (float, optional): the initial bias value. Defaults to 0.0.
        """
        super(BiasInitLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        nn.init.zeros_(tensor=self.weight)
        if bias:
            nn.init.constant_(tensor=self.bias, val=biasinit)


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3
    """

    def __init__(self, c_in: int, n: int) -> None:
        """
        Args:
            c_in (int): the input dimension.
            n (int): factor by which c_in is multiplied to obtain hidden dimension.
        """
        super(Transition, self).__init__()
        self.n = n
        self.c_in = c_in
        self.layernorm1 = LayerNorm(c_in)
        self.linear_no_bias_a = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias_b = LinearNoBias(in_features=c_in, out_features=n * c_in)
        self.linear_no_bias = LinearNoBias(in_features=n * c_in, out_features=c_in)
        self.zero_init()

    def zero_init(self) -> None:
        """Initialize the final linear layer with zeros."""
        nn.init.zeros_(self.linear_no_bias.weight)

    def _process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a single chunk of input data.

        Args:
            chunk (torch.Tensor): Input tensor chunk

        Returns:
            torch.Tensor: Processed chunk
        """
        y = self.layernorm1(chunk)
        a = self.linear_no_bias_a(y)
        a = F.silu(a, True)
        b = self.linear_no_bias_b(y)
        del y
        b *= a
        del a
        return self.linear_no_bias(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        if self.training:
            x = self.layernorm1(x)
            a = self.linear_no_bias_a(x)
            b = self.linear_no_bias_b(x)
            x = self.linear_no_bias(F.silu(a) * b)
            return x
        else:
            # Optimization for inference
            other_dims = x.shape[:-1]
            dim_size = x.shape[-1]
            size = x.shape[-2] if len(x.shape) > 1 else 0

            # Reshape for batch processing
            x = x.reshape(-1, dim_size)

            # Determine chunking based on size
            chunk_num = 1 if size < 3200 else 8
            chunks = torch.chunk(x, chunk_num, dim=-2)

            # Pre-allocate output tensor
            outputs = torch.empty(
                (x.shape[0], self.c_in), dtype=x.dtype, device=x.device
            )

            # Process each chunk
            start = 0
            for chunk in chunks:
                b = self._process_chunk(chunk)
                outputs[start : start + b.shape[0]] = b
                start += b.shape[0]
                del b

            # Reshape back to original dimensions
            outputs = outputs.reshape(*other_dims, self.c_in)
            return outputs

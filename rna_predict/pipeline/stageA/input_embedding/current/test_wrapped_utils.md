
You are given one or more automatically generated Python test files that test various classes and functions. These tests may have issues such as poor naming conventions, inconsistent usage of self, lack of setUp methods, minimal docstrings, redundant or duplicate tests, and limited assertion coverage. They may also fail to leverage hypothesis and unittest.mock effectively, and might not be logically grouped.

Your task is to produce a single, consolidated, high-quality test file from the given input files. The refactored test file should incorporate the following improvements:
	1.	Consolidation and Organization
	•	Combine all tests from the provided files into one coherent Python test file.
	•	Group tests into classes that correspond logically to the functionality they are testing (e.g., separate test classes by the class or function under test).
	•	Within each class, order test methods logically (e.g., basic functionality first, edge cases, error handling, round-trip tests afterward).
	2.	Clean, Readable Code
	•	Use descriptive, PEP 8-compliant class and method names.
	•	Add docstrings to each test class and test method, explaining their purpose and what they verify.
	•	Remove redundant, duplicate, or meaningless tests. Combine or refactor tests that cover the same functionality into a single, comprehensive test method when appropriate.
	3.	Proper Test Fixtures
	•	Utilize setUp methods to instantiate commonly used objects before each test method, reducing redundancy.
	•	Ensure that instance methods of classes under test are called on properly instantiated objects rather than passing self incorrectly as an argument.
	4.	Robust Assertions and Coverage
	•	Include multiple assertions in each test to thoroughly verify behavior and correctness.
	•	Use unittest’s assertRaises for expected exceptions to validate error handling.
	•	Implement at least one round-trip test (e.g., encode then decode a data structure, or transform an object multiple times to ensure idempotency).
	5.	Effective Use of Hypothesis
	•	Employ hypothesis to generate a wide range of input data, ensuring better coverage and exposing edge cases.
	•	Use strategies like st.builds to create complex objects (e.g., custom dataclasses) with varied attribute values.
	•	Enforce constraints (e.g., allow_nan=False) to avoid nonsensical test inputs.
	6.	Mocking External Dependencies
	•	Use unittest.mock where appropriate to simulate external dependencies or environments, ensuring tests are reliable and isolated from external conditions.

⸻

Additional Context: Getting Started with Hypothesis

Below is a practical guide that outlines common use cases and best practices for leveraging hypothesis:
	1.	Basic Usage
	•	Decorate test functions with @given and specify a strategy (e.g., @given(st.text())).
	•	Let hypothesis generate diverse test cases automatically.
	2.	Common Strategies
	•	Use built-in strategies like st.integers(), st.floats(), st.text(), etc.
	•	Combine strategies with st.lists, st.builds, or st.composite to generate complex objects.
	3.	Composing Tests
	•	Employ assume() to filter out unwanted test cases.
	•	Compose or build custom objects to test domain-specific logic.
	4.	Advanced Features
	•	Fine-tune test runs with @settings (e.g., max_examples=1000).
	•	Create reusable strategies via @composite.
	5.	Best Practices
	•	Keep tests focused on one property at a time.
	•	Use explicit examples with @example() for edge cases.
	•	Manage performance by choosing realistic strategy bounds.
	6.	Debugging Failed Tests
	•	Hypothesis shows minimal failing examples and seeds to help reproduce and fix issues.

⸻

Input Format

TEST CODE: 
# ----- test_rot_vec_mul_binary-op.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestBinaryOperationrot_vec_mul(unittest.TestCase):
    rot_vec_mul_operands = st.builds(Tensor)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands, c=rot_vec_mul_operands)
    def test_associative_binary_operation_rot_vec_mul(self, a, b, c) -> None:
        left = utils.rot_vec_mul(r=a, t=utils.rot_vec_mul(r=b, t=c))
        right = utils.rot_vec_mul(r=utils.rot_vec_mul(r=a, t=b), t=c)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands)
    def test_commutative_binary_operation_rot_vec_mul(self, a, b) -> None:
        left = utils.rot_vec_mul(r=a, t=b)
        right = utils.rot_vec_mul(r=b, t=a)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands)
    def test_identity_binary_operation_rot_vec_mul(self, a) -> None:
        identity = tensor([])
        self.assertEqual(a, utils.rot_vec_mul(r=a, t=identity))
        self.assertEqual(a, utils.rot_vec_mul(r=identity, t=a))

# ----- test_uniform_random_rotation_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzUniform_Random_Rotation(unittest.TestCase):

    @given(N_sample=st.integers())
    def test_fuzz_uniform_random_rotation(self, N_sample: int) -> None:
        utils.uniform_random_rotation(N_sample=N_sample)

# ----- test_sample_indices_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import device

class TestFuzzSample_Indices(unittest.TestCase):

    @given(n=st.integers(), device=st.builds(device), lower_bound=st.just(1), strategy=st.text())
    def test_fuzz_sample_indices(self, n: int, device: torch.device, lower_bound, strategy: str) -> None:
        utils.sample_indices(n=n, device=device, lower_bound=lower_bound, strategy=strategy)

# ----- test_sample_msa_feature_dict_random_without_replacement_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzSample_Msa_Feature_Dict_Random_Without_Replacement(unittest.TestCase):

    @given(feat_dict=st.dictionaries(keys=st.text(), values=st.builds(Tensor)), dim_dict=st.dictionaries(keys=st.text(), values=st.integers()), cutoff=st.integers(), lower_bound=st.integers(), strategy=st.text())
    def test_fuzz_sample_msa_feature_dict_random_without_replacement(self, feat_dict: dict, dim_dict: dict, cutoff: int, lower_bound: int, strategy: str) -> None:
        utils.sample_msa_feature_dict_random_without_replacement(feat_dict=feat_dict, dim_dict=dim_dict, cutoff=cutoff, lower_bound=lower_bound, strategy=strategy)

# ----- test_flatten_final_dims_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzFlatten_Final_Dims(unittest.TestCase):

    @given(t=st.builds(Tensor), num_dims=st.integers())
    def test_fuzz_flatten_final_dims(self, t: torch.Tensor, num_dims: int) -> None:
        utils.flatten_final_dims(t=t, num_dims=num_dims)

# ----- test_simple_merge_dict_list_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzSimple_Merge_Dict_List(unittest.TestCase):

    @given(dict_list=st.lists(st.builds(dict)))
    def test_fuzz_simple_merge_dict_list(self, dict_list: list) -> None:
        utils.simple_merge_dict_list(dict_list=dict_list)

# ----- test_expand_at_dim_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzExpand_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), n=st.integers())
    def test_fuzz_expand_at_dim(self, x: torch.Tensor, dim: int, n: int) -> None:
        utils.expand_at_dim(x=x, dim=dim, n=n)

# ----- test_one_hot_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzOne_Hot(unittest.TestCase):

    @given(x=st.builds(Tensor), lower_bins=st.builds(Tensor), upper_bins=st.builds(Tensor))
    def test_fuzz_one_hot(self, x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor) -> None:
        utils.one_hot(x=x, lower_bins=lower_bins, upper_bins=upper_bins)

# ----- test_broadcast_token_to_atom_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzBroadcast_Token_To_Atom(unittest.TestCase):

    @given(x_token=st.builds(Tensor), atom_to_token_idx=st.builds(Tensor))
    def test_fuzz_broadcast_token_to_atom(self, x_token: torch.Tensor, atom_to_token_idx: torch.Tensor) -> None:
        utils.broadcast_token_to_atom(x_token=x_token, atom_to_token_idx=atom_to_token_idx)

# ----- test_batched_gather_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzBatched_Gather(unittest.TestCase):

    @given(data=st.builds(Tensor), inds=st.builds(Tensor), dim=st.integers(), no_batch_dims=st.integers())
    def test_fuzz_batched_gather(self, data: torch.Tensor, inds: torch.Tensor, dim: int, no_batch_dims: int) -> None:
        utils.batched_gather(data=data, inds=inds, dim=dim, no_batch_dims=no_batch_dims)

# ----- test_permute_final_dims_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzPermute_Final_Dims(unittest.TestCase):

    @given(tensor=st.builds(Tensor), inds=st.lists(st.integers()))
    def test_fuzz_permute_final_dims(self, tensor: torch.Tensor, inds: list) -> None:
        utils.permute_final_dims(tensor=tensor, inds=inds)

# ----- test_pad_at_dim_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzPad_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), pad_length=st.one_of(st.lists(st.integers()), st.tuples(st.integers())), value=st.floats())
    def test_fuzz_pad_at_dim(self, x: torch.Tensor, dim: int, pad_length: typing.Union[tuple, list], value: float) -> None:
        utils.pad_at_dim(x=x, dim=dim, pad_length=pad_length, value=value)

# ----- test_rot_vec_mul_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestBinaryOperationrot_vec_mul(unittest.TestCase):
    rot_vec_mul_operands = st.builds(Tensor)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands, c=rot_vec_mul_operands)
    def test_associative_binary_operation_rot_vec_mul(self, a, b, c) -> None:
        left = utils.rot_vec_mul(r=a, t=utils.rot_vec_mul(r=b, t=c))
        right = utils.rot_vec_mul(r=utils.rot_vec_mul(r=a, t=b), t=c)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands)
    def test_commutative_binary_operation_rot_vec_mul(self, a, b) -> None:
        left = utils.rot_vec_mul(r=a, t=b)
        right = utils.rot_vec_mul(r=b, t=a)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands)
    def test_identity_binary_operation_rot_vec_mul(self, a) -> None:
        identity = tensor([])
        self.assertEqual(a, utils.rot_vec_mul(r=a, t=identity))
        self.assertEqual(a, utils.rot_vec_mul(r=identity, t=a))

# ----- test_centre_random_augmentation_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzCentre_Random_Augmentation(unittest.TestCase):

    @given(x_input_coords=st.builds(Tensor), N_sample=st.integers(), s_trans=st.floats(), centre_only=st.booleans(), mask=st.one_of(st.none(), st.builds(Tensor)), eps=st.floats())
    def test_fuzz_centre_random_augmentation(self, x_input_coords: torch.Tensor, N_sample: int, s_trans: float, centre_only: bool, mask: torch.Tensor, eps: float) -> None:
        utils.centre_random_augmentation(x_input_coords=x_input_coords, N_sample=N_sample, s_trans=s_trans, centre_only=centre_only, mask=mask, eps=eps)

# ----- test_move_final_dim_to_dim_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzMove_Final_Dim_To_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers())
    def test_fuzz_move_final_dim_to_dim(self, x: torch.Tensor, dim: int) -> None:
        utils.move_final_dim_to_dim(x=x, dim=dim)

# ----- test_reshape_at_dim_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzReshape_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), target_shape=st.one_of(st.lists(st.integers()), st.tuples(st.integers())))
    def test_fuzz_reshape_at_dim(self, x: torch.Tensor, dim: int, target_shape: typing.Union[tuple, list]) -> None:
        utils.reshape_at_dim(x=x, dim=dim, target_shape=target_shape)

# ----- test_aggregate_atom_to_token_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzAggregate_Atom_To_Token(unittest.TestCase):

    @given(x_atom=st.builds(Tensor), atom_to_token_idx=st.builds(Tensor), n_token=st.one_of(st.none(), st.integers()), reduce=st.text())
    def test_fuzz_aggregate_atom_to_token(self, x_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, n_token: typing.Optional[int], reduce: str) -> None:
        utils.aggregate_atom_to_token(x_atom=x_atom, atom_to_token_idx=atom_to_token_idx, n_token=n_token, reduce=reduce)

FULL SRC CODE: # protenix/model/utils.py
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import numpy as np
import snoop
import torch
import torch.nn as nn
from protenix.utils.scatter_utils import scatter
from scipy.spatial.transform import Rotation


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, N_sample, 3)).to(
        device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]
    return x_augment_coords


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# this is from openfold.utils.rigid_utils import rot_vec_mul
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (torch.Tensor): the rotation matrices
            [..., 3, 3]
        t (torch.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        torch.Tensor: the rotated coordinates
    """
    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(
    x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor
) -> torch.Tensor:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x (torch.Tensor): the input x
            [...]
        lower_bins (torch.Tensor): the lower bounds of bins
            [bins]
        upper_bins (torch.Tensor): the upper bounds of bins
            [bins]
    Returns:
        torch.Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# this is mostly from openfold.utils.torch_utils import batched_gather
def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# #@snoop
def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast token-level embeddings to atom-level embeddings.

    This handles cases where:
      1) atom_to_token_idx is purely 1D -> we unsqueeze to [1, N_atom].
      2) x_token may have one more batch dim than atom_to_token_idx, so we unsqueeze one dim in the index as well.
    """

    # Step 1: If purely 1D => unsqueeze to add a batch dimension
    if atom_to_token_idx.ndim == 1:
        atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

    # Step 2: If x_token has exactly one more dimension than atom_to_token_idx,
    # unsqueeze in the second-last dimension to match the shape logic.
    if len(x_token.shape) == len(atom_to_token_idx.shape) + 1:
        # e.g. x_token is [B, S, N_token, d], atom_to_token_idx is [B, N_atom], so we do:
        atom_to_token_idx = atom_to_token_idx.unsqueeze(-2)

    # Final shape check
    assert atom_to_token_idx.shape[:-1] == x_token.shape[:-2], (
        f"Shape mismatch in broadcast_token_to_atom: "
        f"atom_to_token_idx.shape[:-1]={atom_to_token_idx.shape[:-1]} vs. "
        f"x_token.shape[:-2]={x_token.shape[:-2]}"
    )

    # If still exactly 1D after expansions, do direct indexing
    if atom_to_token_idx.ndim == 1:
        return x_token[..., atom_to_token_idx, :]

    # Otherwise, fall back to batched gather
    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def aggregate_atom_to_token(
    x_atom: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (torch.Tensor): atom-level embedding
            [..., N_atom, d]
        atom_to_token_idx (torch.Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int, optional): number of tokens in total. Defaults to None.
        reduce (str, optional): aggregation method. Defaults to "mean".

    Returns:
        torch.Tensor: token-level embedding
            [..., N_token, d]
    """

    # Broadcasting in the given dim.
    out = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce=reduce
    )

    return out


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> torch.Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        torch.Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, torch.Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict


Where:
	•	
# ----- test_rot_vec_mul_binary-op.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestBinaryOperationrot_vec_mul(unittest.TestCase):
    rot_vec_mul_operands = st.builds(Tensor)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands, c=rot_vec_mul_operands)
    def test_associative_binary_operation_rot_vec_mul(self, a, b, c) -> None:
        left = utils.rot_vec_mul(r=a, t=utils.rot_vec_mul(r=b, t=c))
        right = utils.rot_vec_mul(r=utils.rot_vec_mul(r=a, t=b), t=c)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands)
    def test_commutative_binary_operation_rot_vec_mul(self, a, b) -> None:
        left = utils.rot_vec_mul(r=a, t=b)
        right = utils.rot_vec_mul(r=b, t=a)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands)
    def test_identity_binary_operation_rot_vec_mul(self, a) -> None:
        identity = tensor([])
        self.assertEqual(a, utils.rot_vec_mul(r=a, t=identity))
        self.assertEqual(a, utils.rot_vec_mul(r=identity, t=a))

# ----- test_uniform_random_rotation_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzUniform_Random_Rotation(unittest.TestCase):

    @given(N_sample=st.integers())
    def test_fuzz_uniform_random_rotation(self, N_sample: int) -> None:
        utils.uniform_random_rotation(N_sample=N_sample)

# ----- test_sample_indices_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import device

class TestFuzzSample_Indices(unittest.TestCase):

    @given(n=st.integers(), device=st.builds(device), lower_bound=st.just(1), strategy=st.text())
    def test_fuzz_sample_indices(self, n: int, device: torch.device, lower_bound, strategy: str) -> None:
        utils.sample_indices(n=n, device=device, lower_bound=lower_bound, strategy=strategy)

# ----- test_sample_msa_feature_dict_random_without_replacement_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzSample_Msa_Feature_Dict_Random_Without_Replacement(unittest.TestCase):

    @given(feat_dict=st.dictionaries(keys=st.text(), values=st.builds(Tensor)), dim_dict=st.dictionaries(keys=st.text(), values=st.integers()), cutoff=st.integers(), lower_bound=st.integers(), strategy=st.text())
    def test_fuzz_sample_msa_feature_dict_random_without_replacement(self, feat_dict: dict, dim_dict: dict, cutoff: int, lower_bound: int, strategy: str) -> None:
        utils.sample_msa_feature_dict_random_without_replacement(feat_dict=feat_dict, dim_dict=dim_dict, cutoff=cutoff, lower_bound=lower_bound, strategy=strategy)

# ----- test_flatten_final_dims_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzFlatten_Final_Dims(unittest.TestCase):

    @given(t=st.builds(Tensor), num_dims=st.integers())
    def test_fuzz_flatten_final_dims(self, t: torch.Tensor, num_dims: int) -> None:
        utils.flatten_final_dims(t=t, num_dims=num_dims)

# ----- test_simple_merge_dict_list_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestFuzzSimple_Merge_Dict_List(unittest.TestCase):

    @given(dict_list=st.lists(st.builds(dict)))
    def test_fuzz_simple_merge_dict_list(self, dict_list: list) -> None:
        utils.simple_merge_dict_list(dict_list=dict_list)

# ----- test_expand_at_dim_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzExpand_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), n=st.integers())
    def test_fuzz_expand_at_dim(self, x: torch.Tensor, dim: int, n: int) -> None:
        utils.expand_at_dim(x=x, dim=dim, n=n)

# ----- test_one_hot_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzOne_Hot(unittest.TestCase):

    @given(x=st.builds(Tensor), lower_bins=st.builds(Tensor), upper_bins=st.builds(Tensor))
    def test_fuzz_one_hot(self, x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor) -> None:
        utils.one_hot(x=x, lower_bins=lower_bins, upper_bins=upper_bins)

# ----- test_broadcast_token_to_atom_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzBroadcast_Token_To_Atom(unittest.TestCase):

    @given(x_token=st.builds(Tensor), atom_to_token_idx=st.builds(Tensor))
    def test_fuzz_broadcast_token_to_atom(self, x_token: torch.Tensor, atom_to_token_idx: torch.Tensor) -> None:
        utils.broadcast_token_to_atom(x_token=x_token, atom_to_token_idx=atom_to_token_idx)

# ----- test_batched_gather_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzBatched_Gather(unittest.TestCase):

    @given(data=st.builds(Tensor), inds=st.builds(Tensor), dim=st.integers(), no_batch_dims=st.integers())
    def test_fuzz_batched_gather(self, data: torch.Tensor, inds: torch.Tensor, dim: int, no_batch_dims: int) -> None:
        utils.batched_gather(data=data, inds=inds, dim=dim, no_batch_dims=no_batch_dims)

# ----- test_permute_final_dims_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzPermute_Final_Dims(unittest.TestCase):

    @given(tensor=st.builds(Tensor), inds=st.lists(st.integers()))
    def test_fuzz_permute_final_dims(self, tensor: torch.Tensor, inds: list) -> None:
        utils.permute_final_dims(tensor=tensor, inds=inds)

# ----- test_pad_at_dim_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzPad_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), pad_length=st.one_of(st.lists(st.integers()), st.tuples(st.integers())), value=st.floats())
    def test_fuzz_pad_at_dim(self, x: torch.Tensor, dim: int, pad_length: typing.Union[tuple, list], value: float) -> None:
        utils.pad_at_dim(x=x, dim=dim, pad_length=pad_length, value=value)

# ----- test_rot_vec_mul_basic.py -----
import unittest
import utils
from hypothesis import given, strategies as st

class TestBinaryOperationrot_vec_mul(unittest.TestCase):
    rot_vec_mul_operands = st.builds(Tensor)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands, c=rot_vec_mul_operands)
    def test_associative_binary_operation_rot_vec_mul(self, a, b, c) -> None:
        left = utils.rot_vec_mul(r=a, t=utils.rot_vec_mul(r=b, t=c))
        right = utils.rot_vec_mul(r=utils.rot_vec_mul(r=a, t=b), t=c)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands, b=rot_vec_mul_operands)
    def test_commutative_binary_operation_rot_vec_mul(self, a, b) -> None:
        left = utils.rot_vec_mul(r=a, t=b)
        right = utils.rot_vec_mul(r=b, t=a)
        self.assertEqual(left, right)

    @given(a=rot_vec_mul_operands)
    def test_identity_binary_operation_rot_vec_mul(self, a) -> None:
        identity = tensor([])
        self.assertEqual(a, utils.rot_vec_mul(r=a, t=identity))
        self.assertEqual(a, utils.rot_vec_mul(r=identity, t=a))

# ----- test_centre_random_augmentation_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzCentre_Random_Augmentation(unittest.TestCase):

    @given(x_input_coords=st.builds(Tensor), N_sample=st.integers(), s_trans=st.floats(), centre_only=st.booleans(), mask=st.one_of(st.none(), st.builds(Tensor)), eps=st.floats())
    def test_fuzz_centre_random_augmentation(self, x_input_coords: torch.Tensor, N_sample: int, s_trans: float, centre_only: bool, mask: torch.Tensor, eps: float) -> None:
        utils.centre_random_augmentation(x_input_coords=x_input_coords, N_sample=N_sample, s_trans=s_trans, centre_only=centre_only, mask=mask, eps=eps)

# ----- test_move_final_dim_to_dim_basic.py -----
import torch
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzMove_Final_Dim_To_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers())
    def test_fuzz_move_final_dim_to_dim(self, x: torch.Tensor, dim: int) -> None:
        utils.move_final_dim_to_dim(x=x, dim=dim)

# ----- test_reshape_at_dim_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzReshape_At_Dim(unittest.TestCase):

    @given(x=st.builds(Tensor), dim=st.integers(), target_shape=st.one_of(st.lists(st.integers()), st.tuples(st.integers())))
    def test_fuzz_reshape_at_dim(self, x: torch.Tensor, dim: int, target_shape: typing.Union[tuple, list]) -> None:
        utils.reshape_at_dim(x=x, dim=dim, target_shape=target_shape)

# ----- test_aggregate_atom_to_token_basic.py -----
import torch
import typing
import unittest
import utils
from hypothesis import given, strategies as st
from torch import Tensor

class TestFuzzAggregate_Atom_To_Token(unittest.TestCase):

    @given(x_atom=st.builds(Tensor), atom_to_token_idx=st.builds(Tensor), n_token=st.one_of(st.none(), st.integers()), reduce=st.text())
    def test_fuzz_aggregate_atom_to_token(self, x_atom: torch.Tensor, atom_to_token_idx: torch.Tensor, n_token: typing.Optional[int], reduce: str) -> None:
        utils.aggregate_atom_to_token(x_atom=x_atom, atom_to_token_idx=atom_to_token_idx, n_token=n_token, reduce=reduce)
 is the content of your automatically generated Python test files (potentially multiple files’ content combined or listed).
	•	# protenix/model/utils.py
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import numpy as np
import snoop
import torch
import torch.nn as nn
from protenix.utils.scatter_utils import scatter
from scipy.spatial.transform import Rotation


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, N_sample, 3)).to(
        device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]
    return x_augment_coords


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# this is from openfold.utils.rigid_utils import rot_vec_mul
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (torch.Tensor): the rotation matrices
            [..., 3, 3]
        t (torch.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        torch.Tensor: the rotated coordinates
    """
    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(
    x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor
) -> torch.Tensor:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x (torch.Tensor): the input x
            [...]
        lower_bins (torch.Tensor): the lower bounds of bins
            [bins]
        upper_bins (torch.Tensor): the upper bounds of bins
            [bins]
    Returns:
        torch.Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# this is mostly from openfold.utils.torch_utils import batched_gather
def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


# #@snoop
def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """
    Broadcast token-level embeddings to atom-level embeddings.

    This handles cases where:
      1) atom_to_token_idx is purely 1D -> we unsqueeze to [1, N_atom].
      2) x_token may have one more batch dim than atom_to_token_idx, so we unsqueeze one dim in the index as well.
    """

    # Step 1: If purely 1D => unsqueeze to add a batch dimension
    if atom_to_token_idx.ndim == 1:
        atom_to_token_idx = atom_to_token_idx.unsqueeze(0)

    # Step 2: If x_token has exactly one more dimension than atom_to_token_idx,
    # unsqueeze in the second-last dimension to match the shape logic.
    if len(x_token.shape) == len(atom_to_token_idx.shape) + 1:
        # e.g. x_token is [B, S, N_token, d], atom_to_token_idx is [B, N_atom], so we do:
        atom_to_token_idx = atom_to_token_idx.unsqueeze(-2)

    # Final shape check
    assert atom_to_token_idx.shape[:-1] == x_token.shape[:-2], (
        f"Shape mismatch in broadcast_token_to_atom: "
        f"atom_to_token_idx.shape[:-1]={atom_to_token_idx.shape[:-1]} vs. "
        f"x_token.shape[:-2]={x_token.shape[:-2]}"
    )

    # If still exactly 1D after expansions, do direct indexing
    if atom_to_token_idx.ndim == 1:
        return x_token[..., atom_to_token_idx, :]

    # Otherwise, fall back to batched gather
    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def aggregate_atom_to_token(
    x_atom: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (torch.Tensor): atom-level embedding
            [..., N_atom, d]
        atom_to_token_idx (torch.Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int, optional): number of tokens in total. Defaults to None.
        reduce (str, optional): aggregation method. Defaults to "mean".

    Returns:
        torch.Tensor: token-level embedding
            [..., N_token, d]
    """

    # Broadcasting in the given dim.
    out = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce=reduce
    )

    return out


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> torch.Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        torch.Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, torch.Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict
 is the content of the source code under test (if needed for context).

Output Format

Provide a single Python code block containing the fully refactored, consolidated test file. The output should be ready-to-run with:

python -m unittest

It must exhibit all of the improvements listed above, including:
	•	Logical grouping of tests,
	•	Clear and correct usage of setUp,
	•	Docstrings for test classes and methods,
	•	Consolidated and refactored tests (no duplicates),
	•	Robust assertions and coverage,
	•	Use of hypothesis with one or more examples,
	•	Use of mock where appropriate.

⸻
============
EXTRA USEFUL CONTEXT TO AID YOU IN YOUR TASK:
Hypothesis: A Comprehensive Best-Practice and Reference Guide

Hypothesis is a powerful property-based testing library for Python, designed to help you find subtle bugs by generating large numbers of test inputs and minimizing failing examples. This document combines the strengths and core ideas of three earlier guides. It serves as a broad, in-depth resource: covering Hypothesis usage from the basics to advanced methods, including background on its internal mechanisms (Conjecture) and integration with complex workflows.

⸻

Table of Contents
	1.	Introduction to Property-Based Testing
1.1 What Is Property-Based Testing?
1.2 Why Use Property-Based Testing?
1.3 Installing Hypothesis
	2.	First Steps with Hypothesis
2.1 A Simple Example
2.2 Basic Workflows and Key Concepts
2.3 Troubleshooting the First Failures
	3.	Core Hypothesis Concepts
3.1 The @given Decorator
3.2 Strategies: Building and Composing Data Generators
3.3 Shrinking and Minimizing Failing Examples
3.4 Example Database and Replay
	4.	Advanced Data Generation
4.1 Understanding Strategies vs. Types
4.2 Composing Strategies (map, filter, flatmap)
4.3 Working with Complex or Recursive Data
4.4 Using @composite Functions
4.5 Integration and Edge Cases
	5.	Practical Usage Patterns
5.1 Testing Numeric Code (Floating-Point, Bounds)
5.2 Text and String Generation (Character Sets, Regex)
5.3 Dates, Times, and Time Zones
5.4 Combining Hypothesis with Fixtures and Other Test Tools
	6.	Stateful/Model-Based Testing
6.1 The RuleBasedStateMachine and @rule Decorators
6.2 Designing Operations and Invariants
6.3 Managing Complex State and Multiple Bundles
6.4 Example: Testing a CRUD System or Other Stateful API
	7.	Performance and Health Checks
7.1 Diagnosing Slow Tests with Deadlines
7.2 Common Health Check Warnings and Their Meanings
7.3 Filtering Pitfalls (assume / Over-Filters)
7.4 Tuning Hypothesis Settings (max_examples, phases, etc.)
7.5 Speed vs. Thoroughness
	8.	Multiple Failures and Multi-Bug Discovery
8.1 How Hypothesis Detects and Distinguishes Bugs
8.2 Typical Bug Slippage and the “Threshold Problem”
8.3 Strategies for Handling Multiple Distinct Failures
	9.	Internals: The Conjecture Engine
9.1 Overview of Bytestream-Based Generation
9.2 Integrated Shrinking vs. Type-Based Shrinking
9.3 How Conjecture Tracks and Minimizes Examples
9.4 The Example Database in Depth
	10.	Hypothesis in Real-World Scenarios
10.1 Using Hypothesis in CI/CD
10.2 Collaborative Testing in Teams
10.3 Integrating with Other Tools (pytest, coverage, etc.)
10.4 Best Practices for Large Projects
	11.	Extensibility and Advanced Topics
11.1 Third-Party Extensions (e.g., Hypothesis-Bio, Hypothesis-NetworkX)
11.2 Targeted Property-Based Testing (Scoring)
11.3 Hybrid Approaches (Combining Examples with Generation)
11.4 Glass-Box Testing and Potential Future Work
	12.	Troubleshooting and FAQs
12.1 Common Error Messages
12.2 Reproduce Failures with @reproduce_failure and Seeds
12.3 Overcoming Flaky or Non-Deterministic Tests
12.4 Interpreting Statistics
	13.	Summary and Further Reading
13.1 Key Takeaways and Next Steps
13.2 Recommended Resources and Papers
13.3 Contributing to Hypothesis

⸻

1. Introduction to Property-Based Testing

1.1 What Is Property-Based Testing?

Property-based testing (PBT) shifts your focus from manually enumerating test inputs to describing the properties your code should fulfill for all valid inputs. Instead of hardcoding specific examples (like assert f(2) == 4), you define requirements: e.g., “Sorting a list is idempotent.” Then the library (Hypothesis) generates test inputs to find edge cases or scenarios violating those properties.

Example

from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_sort_idempotent(xs):
    once = sorted(xs)
    twice = sorted(once)
    assert once == twice

Hypothesis tries diverse lists (including empty lists, duplicates, large sizes, negative or positive numbers). If something fails, it shrinks the input to a minimal failing example.

1.2 Why Use Property-Based Testing?
	•	Coverage of Edge Cases: Automatically covers many corner cases—empty inputs, large values, special floats, etc.
	•	Reduced Manual Labor: You specify broad properties, and the tool handles enumerations.
	•	Debugging Aid: Found a failing input? Hypothesis shrinks it to a simpler version, making debug cycles shorter.
	•	Less Test Boilerplate: Fewer individual test cases to write while achieving higher coverage.

1.3 Installing Hypothesis

You can install the base library with pip install hypothesis. For specialized extras (e.g., date/time, Django), consult Hypothesis extras docs.

⸻

2. First Steps with Hypothesis

2.1 A Simple Example

from hypothesis import given
from hypothesis.strategies import integers

@given(integers())
def test_square_is_nonnegative(x):
    assert x*x >= 0

Run with pytest, unittest, or another runner. Hypothesis calls test_square_is_nonnegative multiple times with varied integers (positive, negative, zero).

2.2 Basic Workflows and Key Concepts
	1.	Test Functions: Decorate with @given(<strategies>).
	2.	Generation and Execution: Hypothesis runs tests many times with random values, tries to find failures.
	3.	Shrinking: If a failure occurs, Hypothesis narrows down (shrinks) the input to a minimal failing example.

2.3 Troubleshooting the First Failures
	•	Assertion Errors: If you see Falsifying example: ..., Hypothesis found a failing scenario. Use that scenario to fix your code or refine your property.
	•	Health Check Warnings: If you see warnings like “filter_too_much” or “too_slow,” see the Health Checks section.

⸻

3. Core Hypothesis Concepts

3.1 The @given Decorator

@given ties strategies to a test function’s parameters:

from hypothesis import given
from hypothesis.strategies import text, emails

@given(email=emails(), note=text())
def test_process_email(email, note):
    ...

Hypothesis calls test_process_email() repeatedly with random emails and text. If everything passes, the test is green. Otherwise, you get a shrunk failing example.

3.2 Strategies: Building and Composing Data Generators

Hypothesis’s data generation revolves around “strategies.” Basic ones:
	•	integers(), floats(), text(), booleans(), etc.
	•	Containers: lists(elements, ...), dictionaries(keys=..., values=...)
	•	Map/Filter: Transform or constrain existing strategies.
	•	Composite: Build custom strategies for domain objects.

3.3 Shrinking and Minimizing Failing Examples

If a test fails on a complicated input, Hypothesis tries simpler versions: removing elements from lists, changing large ints to smaller ints, etc. The final reported failing input is minimal by lex ordering.

Falsifying example: test_sort_idempotent(xs=[2, 1, 1])

Hypothesis might have started with [random, complicated list] but ended with [2,1,1].

3.4 Example Database and Replay

Failures are saved in a local .hypothesis/ directory. On subsequent runs, Hypothesis replays known failing inputs before generating fresh ones. This ensures consistent reporting once a failing case is discovered.

⸻

4. Advanced Data Generation

4.1 Understanding Strategies vs. Types

Hypothesis does not rely solely on type information. You can define custom constraints to ensure the data you generate matches your domain. E.g., generating only non-empty lists or restricting floats to finite values:

import math

@given(st.lists(st.floats(allow_infinity=False, allow_nan=False), min_size=1))
def test_mean_in_bounds(xs):
    avg = sum(xs)/len(xs)
    assert min(xs) <= avg <= max(xs)

4.2 Composing Strategies (map, filter, flatmap)
	•	map(f) transforms data after generation:

even_integers = st.integers().map(lambda x: x * 2)


	•	filter(pred) discards values that fail pred; be mindful of over-filtering performance.
	•	flatmap(...) draws a value, then uses it to define a new strategy:

# Draw an int n, then a list of length n
st.integers(min_value=0, max_value=10).flatmap(lambda n: st.lists(st.text(), min_size=n, max_size=n))



4.3 Working with Complex or Recursive Data

For tree-like or nested data, use st.recursive(base_strategy, extend_strategy, max_leaves=...) to limit growth. Also consider the @composite decorator to build logic step by step.

from hypothesis import strategies as st, composite

@composite
def user_records(draw):
    name = draw(st.text(min_size=1))
    age = draw(st.integers(min_value=0))
    return "name": name, "age": age

4.4 Using @composite Functions

@composite is a more explicit style than map/flatmap. It helps define multi-step draws within one function. It’s usually simpler for highly interdependent data.

4.5 Integration and Edge Cases
	•	Ensuring Valid Domain Data: Use composites or partial filtering. Overuse of filter(...) can cause slow tests and health-check failures.
	•	Large/Complex Structures: Limit sizes or use constraints (max_size, bounding integers, etc.) to avoid timeouts.

⸻

5. Practical Usage Patterns

5.1 Testing Numeric Code (Floating-Point, Bounds)

Floating point nuances:

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_floats(x):
    ...

Constrain or skip NaNs/infinities if your domain doesn’t handle them. Keep an eye on overflows if sums get large.

5.2 Text and String Generation (Character Sets, Regex)

Hypothesis can generate ASCII, Unicode, or custom sets:

from hypothesis.strategies import text

@given(text(alphabet="ABCDE", min_size=1))
def test_some_text(s):
    assert s[0] in "ABCDE"

Or use from_regex(r"MyPattern") for more specialized scenarios.

5.3 Dates, Times, and Time Zones

Install hypothesis[datetime] for strategies like dates(), datetimes(), timezones(). These handle cross-timezone issues or restricted intervals.

5.4 Combining Hypothesis with Fixtures and Other Test Tools

With pytest, you can pass both fixture arguments and Hypothesis strategy arguments:

import pytest

@pytest.fixture
def db():
    return init_db()

@given(x=st.integers())
def test_db_invariant(db, x):
    assert my_query(db, x) == ...

Function-scoped fixtures are invoked once per test function, not per example, so plan accordingly or do manual setup for each iteration.

⸻

6. Stateful/Model-Based Testing

6.1 The RuleBasedStateMachine and @rule Decorators

For testing stateful systems, Hypothesis uses a rule-based approach:

from hypothesis.stateful import RuleBasedStateMachine, rule

class SimpleCounter(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.counter = 0

    @rule(increment=st.integers(min_value=1, max_value=100))
    def inc(self, increment):
        self.counter += increment
        assert self.counter >= 0

TestCounter = SimpleCounter.TestCase

Hypothesis runs random sequences of operations, checking for invariant violations.

6.2 Designing Operations and Invariants
	•	Each @rule modifies the system under test.
	•	Use @precondition to ensure certain rules only fire in valid states.
	•	Use @invariant to check conditions after each rule.

6.3 Managing Complex State and Multiple Bundles
	•	Bundle(...) helps track created objects and pass them between rules.
	•	Perfect for simulating CRUD or multi-object interactions.

6.4 Example: Testing a CRUD System or Other Stateful API

class CRUDSystem(RuleBasedStateMachine):
    Records = Bundle('records')

    @rule(target=Records, data=st.text())
    def create(self, data):
        record_id = my_create_fn(data)
        return record_id

    @rule(record=Records)
    def delete(self, record):
        my_delete_fn(record)

Hypothesis will produce sequences of create/delete calls. If a bug arises, it provides a minimal sequence reproducing it.

⸻

7. Performance and Health Checks

7.1 Diagnosing Slow Tests with Deadlines

Hypothesis can treat slow examples as errors:

from hypothesis import settings, HealthCheck

@settings(deadline=100)  # 100ms deadline
@given(st.lists(st.integers()))
def test_something(xs):
    ...

If a single test run exceeds 100 ms, it raises DeadlineExceeded. This helps identify performance bottlenecks quickly.

7.2 Common Health Check Warnings and Their Meanings
	•	filter_too_much: A large proportion of generated data is being thrown away. Fix by refining your strategy or combining strategies (instead of heavy use of filter).
	•	too_slow: The test or generation logic is slow. Lower max_examples or investigate your code’s performance.
	•	data_too_large: Possibly generating very large structures. Restrict sizes.

7.3 Filtering Pitfalls (assume / Over-Filters)

Using assume(condition) forcibly discards any example that doesn’t meet condition. Overdoing it can degrade performance drastically. Instead, refine your data strategies:

# Instead of:
@given(st.lists(st.integers()).filter(lambda xs: sum(xs) < 100))

# Use a better approach:
@given(st.lists(st.integers(max_value=100), max_size=10))

7.4 Tuning Hypothesis Settings (max_examples, phases, etc.)
	•	max_examples: Controls how many examples are generated per test (default ~200).
	•	phases: Choose which parts of the test lifecycle (e.g. “shrink”, “reuse”) run.
	•	suppress_health_check: Silence known but acceptable warnings.

7.5 Speed vs. Thoroughness

Balance thorough coverage with test suite runtime. Trim unhelpful extra complexity in data generation. Use deadline or lower max_examples for large test suites.

⸻

8. Multiple Failures and Multi-Bug Discovery

8.1 How Hypothesis Detects and Distinguishes Bugs

Hypothesis typically shrinks until it finds the smallest failing example. But if a test can fail in multiple ways, Hypothesis 3.29+ tries to keep track of each distinct bug (by exception type and line number).

8.2 Typical Bug Slippage and the “Threshold Problem”
	•	Bug Slippage: Starting with one bug scenario but shrinking to a different scenario. Hypothesis tries to keep track and track distinct failures.
	•	Threshold Problem: When tests fail due to crossing a numeric threshold, shrunk examples tend to be just barely beyond that threshold, potentially obscuring the severity of the issue. Techniques to mitigate this can involve “targeting” or custom test logic.

8.3 Strategies for Handling Multiple Distinct Failures

Hypothesis’s multi-failure mode ensures it shrinks each failing scenario independently. You may see multiple minimal failures reported. This can be turned on automatically if distinct bug states are detected.

⸻

9. Internals: The Conjecture Engine

9.1 Overview of Bytestream-Based Generation

Conjecture is the underlying fuzzing engine. It treats every generated example as a lazily consumed byte stream. Strategies interpret segments of bytes as integers, floats, text, etc. This uniform approach:
	•	Simplifies storing known failures to replay them.
	•	Allows integrated shrinking by reducing or rewriting parts of the byte stream.

9.2 Integrated Shrinking vs. Type-Based Shrinking

Old or simpler property-based systems often rely on “type-based” shrinking. Conjecture’s approach integrates shrinking with data generation. This ensures that if you build data by composition (e.g. mapping or flattening strategies), Hypothesis can still shrink effectively.

9.3 How Conjecture Tracks and Minimizes Examples
	•	Each test run has a “buffer” of bytes.
	•	On failure, Conjecture tries different transformations (removing or reducing bytes).
	•	The result is simpler failing input but consistent with the constraints of your strategy.

9.4 The Example Database in Depth

All interesting examples get stored in .hypothesis/examples by default. On re-run, Hypothesis tries these before generating new data. This yields repeatable failures for regression tests—especially helpful in CI setups.

⸻

10. Hypothesis in Real-World Scenarios

10.1 Using Hypothesis in CI/CD
	•	Run Hypothesis-based tests as part of your continuous integration.
	•	The example database can be committed to share known failures across devs.
	•	Set a deadline or use smaller max_examples to keep test times predictable.

10.2 Collaborative Testing in Teams
	•	Consistent Strategy Definitions: Keep your custom strategies in a shared “strategies.py.”
	•	Version Control: The .hypothesis directory can be versioned to share known failing examples, though watch out for merge conflicts.

10.3 Integrating with Other Tools (pytest, coverage, etc.)
	•	Pytest integration is seamless—just write @given tests, run pytest.
	•	Coverage tools measure tested code as usual, but remember Hypothesis can deeply cover corner cases.

10.4 Best Practices for Large Projects
	•	Modular Strategies: Break them down for maintainability.
	•	Tackle Invariants Early: Short-circuit with assume() or well-structured strategies.
	•	Monitor Performance: Use health checks, deadlines, and max_examples config to scale.

⸻

11. Extensibility and Advanced Topics

11.1 Third-Party Extensions
	•	hypothesis-bio: Specialized for bioinformatics data formats.
	•	hypothesis-networkx: Generate networkx graphs, test graph algorithms.
	•	Many more unofficial or domain-specific libraries exist. Creating your own extension is easy.

11.2 Targeted Property-Based Testing (Scoring)

You can “guide” test generation by calling target(score) in your code. Hypothesis tries to evolve test cases with higher scores, focusing on “interesting” or extreme behaviors (like maximizing error metrics).

from hypothesis import given, target
from hypothesis.strategies import floats

@given(x=floats(-1e6, 1e6))
def test_numerical_stability(x):
    err = some_error_metric(x)
    target(err)
    assert err < 9999

11.3 Hybrid Approaches (Combining Examples with Generation)

You can add “example-based tests” to complement property-based ones. Also, you can incorporate real-world test data as seeds or partial strategies.

11.4 Glass-Box Testing and Potential Future Work

Hypothesis largely treats tests as a black box but can be extended with coverage data or other instrumentation for more advanced test generation. This is an open area of R&D.

⸻

12. Troubleshooting and FAQs

12.1 Common Error Messages
	•	Unsatisfiable: Hypothesis can’t find enough valid examples. Possibly an over-filter or an unrealistic requirement.
	•	DeadlineExceeded: Your test or code is too slow for the set deadline ms.
	•	FailedHealthCheck: Usually means you’re doing too much filtering or the example is too large.

12.2 Reproduce Failures with @reproduce_failure and Seeds

If Hypothesis can’t express your failing data via a standard repr, it shows a snippet like:

@reproduce_failure('3.62.0', b'...')
def test_something():
    ...

Adding that snippet ensures the bug is replayed exactly. Alternatively, you can do:

from hypothesis import seed

@seed(12345)
@given(st.integers())
def test_x(x):
    ...

But seeds alone are insufficient if your .hypothesis database is relevant or if your test uses inline data.

12.3 Overcoming Flaky or Non-Deterministic Tests

If code is time-sensitive or concurrency-based, you may see spurious failures. Try limiting concurrency, raising deadlines, or disabling shrinking for certain tests. Alternatively, fix the non-determinism in the tested code.

12.4 Interpreting Statistics

Running pytest --hypothesis-show-statistics yields info on distribution of generated examples, data-generation time vs. test time, etc. This helps find bottlenecks, excessive filtering, or unexpectedly large inputs.

⸻

13. Summary and Further Reading

13.1 Key Takeaways and Next Steps
	•	Write Clear Properties: A crisp property is simpler for Hypothesis to exploit.
	•	Refine Strategies: Good strategy design yields fewer discards and faster tests.
	•	Use Health Checks: They highlight anti-patterns early.
	•	Explore Stateful Testing: Perfect for integration tests or persistent-state bugs.

13.2 Recommended Resources and Papers
	•	Official Hypothesis Documentation
	•	QuickCheck papers: Claessen and Hughes, 2000
	•	Testing–reduction synergy: Regehr et al. “Test-case Reduction via Delta Debugging” (PLDI 2012)
	•	“Hypothesis: A New Approach to Property-Based Testing” (HypothesisWorks website)

13.3 Contributing to Hypothesis

Hypothesis is open source. If you have ideas or find issues:
	•	Check our GitHub repo
	•	Read the Contributing Guide
	•	Every improvement is welcomed—documentation, bug reports, or code!

⸻

Final Thoughts

We hope this unified, comprehensive guide helps you unlock the power of Hypothesis. From quick introductions to advanced stateful testing, from performance pitfalls to internal design details, you now have a toolkit for robust property-based testing in Python.

Happy testing! If you run into any questions, re-check the relevant sections here or visit the community resources. Once you incorporate Hypothesis into your testing workflow, you might find hidden bugs you never anticipated—and that’s the point!
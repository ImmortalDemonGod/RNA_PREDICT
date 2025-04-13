# protenix/model/utils.py
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

"""
Utility functions for RNA structure prediction.

This module provides utility functions for tensor operations, rotation/transformation,
coordinate processing, and general utilities.

This file is maintained for backward compatibility.
New code should import directly from the utils submodules.
"""

# Re-export all functions for backward compatibility
from .rotation import (
    centre_random_augmentation as centre_random_augmentation,
    uniform_random_rotation as uniform_random_rotation,
    rot_vec_mul as rot_vec_mul,
)
from .tensor_ops import (
    permute_final_dims as permute_final_dims,
    flatten_final_dims as flatten_final_dims,
    one_hot as one_hot,
    batched_gather as batched_gather,
    expand_at_dim as expand_at_dim,
    pad_at_dim as pad_at_dim,
    reshape_at_dim as reshape_at_dim,
    move_final_dim_to_dim as move_final_dim_to_dim,
)
from .coordinate_utils import (
    broadcast_token_to_atom as broadcast_token_to_atom,
    aggregate_atom_to_token as aggregate_atom_to_token,
)
from .general import (
    sample_indices as sample_indices,
    sample_msa_feature_dict_random_without_replacement as sample_msa_feature_dict_random_without_replacement,
    simple_merge_dict_list as simple_merge_dict_list,
)
# For backward compatibility, if this module is run directly
if __name__ == "__main__":
    print("This module is not meant to be run directly.")

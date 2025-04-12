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

This module has been refactored into smaller, more cohesive modules.
This file is maintained for backward compatibility.
New code should import directly from the utils submodules.
"""

# Import all functions from the refactored modules for backward compatibility
# These imports are intentionally unused within this file - they're here to be re-exported
# for any code that imports from this file (noqa justifies the unused import warnings)
# The IDE may show "Module has no attribute" warnings, but these will resolve at runtime
# when the Python interpreter properly loads the modules
from rna_predict.pipeline.stageA.input_embedding.current.utils.rotation import (  # noqa
    centre_random_augmentation,  # noqa
    uniform_random_rotation,  # noqa
    rot_vec_mul,  # noqa
)

from rna_predict.pipeline.stageA.input_embedding.current.utils.tensor_ops import (  # noqa
    permute_final_dims,  # noqa
    flatten_final_dims,  # noqa
    one_hot,  # noqa
    batched_gather,  # noqa
    expand_at_dim,  # noqa
    pad_at_dim,  # noqa
    reshape_at_dim,  # noqa
    move_final_dim_to_dim,  # noqa
)

from rna_predict.pipeline.stageA.input_embedding.current.utils.coordinate_utils import (  # noqa
    broadcast_token_to_atom,  # noqa
    aggregate_atom_to_token,  # noqa
)

from rna_predict.pipeline.stageA.input_embedding.current.utils.general import (  # noqa
    sample_indices,  # noqa
    sample_msa_feature_dict_random_without_replacement,  # noqa
    simple_merge_dict_list,  # noqa
)

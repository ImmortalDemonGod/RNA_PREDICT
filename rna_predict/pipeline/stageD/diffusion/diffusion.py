# protenix/model/modules/diffusion.py
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

import warnings
from typing import Optional, Union, Tuple, Any # Added Any

import torch
import torch.nn as nn
from protenix.model.utils import expand_at_dim

# from protenix.openfold_local.model.primitives import LayerNorm
from rna_predict.pipeline.stageA.input_embedding.current.checkpointing import (
    get_checkpoint_fn,
)
# FourierEmbedding, RelativePositionEncoding are used by the imported DiffusionConditioning
# from rna_predict.pipeline.stageA.input_embedding.current.embedders import (
#     FourierEmbedding,
#     RelativePositionEncoding,
# )
# LayerNorm, LinearNoBias, Transition are used by DiffusionModule and DiffusionConditioning
from rna_predict.pipeline.stageA.input_embedding.current.primitives import (
    LayerNorm,
    LinearNoBias,
    Transition,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
    AtomAttentionDecoder,
    AtomAttentionEncoder,
    DiffusionTransformer,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_decoder import (
    DecoderForwardParams,
)
from rna_predict.pipeline.stageA.input_embedding.current.transformer.atom_attention_encoder import (
    AtomAttentionConfig,
)

# Import from components
from .components.diffusion_conditioning import DiffusionConditioning
# DiffusionSchedule is not directly used by the original DiffusionModule kept here, so not imported for now
# from .components.diffusion_schedule import DiffusionSchedule


# --- Utility functions moved to components/diffusion_utils.py ---

# Original DiffusionConditioning class removed (now imported from components)


# DiffusionSchedule class moved to components/diffusion_schedule.py

# DiffusionModule class moved to components/diffusion_module.py

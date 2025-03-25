# rna_predict/__init__.py

import torch
from protenix.model.modules import embedders
import snoop

_original_forward = embedders.InputFeatureEmbedder.forward

@snoop
def patched_forward(self, input_feature_dict: dict[str, any], inplace_safe: bool = False, chunk_size: any = None):
    print("[PATCH] Running patched forward method.")
    # Check if "deletion_mean" is present
    if "deletion_mean" in input_feature_dict:
        deletion = input_feature_dict["deletion_mean"]
        print(f"[PATCH] Original deletion_mean shape: {deletion.shape}")

        # Unsqueeze if it's still 2D
        if deletion.ndim == 2:
            deletion = deletion.unsqueeze(-1)
            print(f"[PATCH] Unsqueezed deletion_mean shape: {deletion.shape}")

        # Determine expected tokens from 'restype'
        expected_tokens = input_feature_dict["restype"].shape[1]

        if deletion.shape[1] != expected_tokens:
            print(
                f"[PATCH] Forcing fresh allocation from {deletion.shape} "
                f"to ([{deletion.shape[0]}, {expected_tokens}, {deletion.shape[-1]}])."
            )
            new_deletion = torch.empty(
                (deletion.shape[0], expected_tokens, deletion.shape[-1]),
                dtype=deletion.dtype,
                device=deletion.device,
            )
            # Copy over the relevant slice
            new_deletion.copy_(deletion[:, :expected_tokens, :])
            deletion = new_deletion

        # Assign back
        input_feature_dict["deletion_mean"] = deletion

    # Then call the original
    return _original_forward(self, input_feature_dict, inplace_safe, chunk_size)

# Override
embedders.InputFeatureEmbedder.forward = patched_forward
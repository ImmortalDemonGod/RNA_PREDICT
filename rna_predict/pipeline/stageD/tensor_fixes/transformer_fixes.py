"""
Transformer-related tensor shape compatibility fixes.
"""

from functools import wraps


def fix_atom_transformer():
    """
    Fix the atom transformer to handle shape mismatches by adjusting q and s, not p.
    """
    from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
        atom_transformer,
    )

    # Check if already patched with this specific fix
    if hasattr(atom_transformer.AtomTransformer, "_patch_applied_forward_fix"):
        print("[DEBUG] AtomTransformer.forward already patched (fix). Skipping.")
        return

    original_forward = atom_transformer.AtomTransformer.forward

    @wraps(original_forward)
    def patched_forward(self, q, s, p, inplace_safe=False, chunk_size=None):
        # Determine the relevant sequence dimension index based on tensor dimensions
        # Assuming q: [..., N_seq_q, C_q], s: [..., N_seq_s, C_s]
        seq_dim_idx = -2

        # Check for sequence dimension mismatch between q and s
        # Add safety check for dimensions being large enough
        if (
            q.dim() > abs(seq_dim_idx)
            and s.dim() > abs(seq_dim_idx)
            and q.shape[seq_dim_idx] != s.shape[seq_dim_idx]
        ):
            print(
                f"[DEBUG][Patch] Shape mismatch detected: q.shape={q.shape}, s.shape={s.shape}. Adjusting q and s."
            )
            min_len = min(q.shape[seq_dim_idx], s.shape[seq_dim_idx])

            # Slice q
            slices_q = [slice(None)] * q.dim()
            slices_q[seq_dim_idx] = slice(0, min_len)
            q = q[tuple(slices_q)]

            # Slice s
            slices_s = [slice(None)] * s.dim()
            slices_s[seq_dim_idx] = slice(0, min_len)
            s = s[tuple(slices_s)]
            print(
                f"[DEBUG][Patch] Adjusted shapes: q.shape={q.shape}, s.shape={s.shape}"
            )

            # Important: DO NOT slice p here

        # Call the original forward method with potentially adjusted q and s, but original p
        return original_forward(self, q, s, p, inplace_safe, chunk_size)

    # atom_transformer.AtomTransformer.forward = patched_forward # Disable faulty patch
    # atom_transformer.AtomTransformer._patch_applied_forward_fix = True  # Add flag
    print("[INFO] Faulty AtomTransformer patch disabled.")


def fix_atom_attention_encoder():
    """
    Fix the atom attention encoder to handle shape mismatches and maintain backward compatibility.
    Adds a check to prevent re-patching.
    """
    from rna_predict.pipeline.stageA.input_embedding.current.transformer import (
        atom_attention_encoder,
    )

    # Check if already patched
    if hasattr(atom_attention_encoder.AtomAttentionEncoder, "_patch_applied_forward"):
        print("[DEBUG] AtomAttentionEncoder.forward already patched. Skipping.")
        return

    original_forward = atom_attention_encoder.AtomAttentionEncoder.forward
    original_init = atom_attention_encoder.AtomAttentionEncoder.__init__

    # Capture the original function in the closure
    _original_forward_captured = original_forward
    # Remove the potentially problematic class attribute assignment
    # atom_attention_encoder.AtomAttentionEncoder._original_forward = original_forward

    @wraps(original_forward)
    def patched_forward(self, *args, **kwargs):
        # Handle both old and new interface
        if len(args) == 2 and not kwargs:  # Old interface
            input_feature_dict, params = args
            # Call the captured original function directly, passing self
            return _original_forward_captured(
                self,  # Pass self explicitly
                input_feature_dict=input_feature_dict,
                r_l=params.get("r_l"),
                s=params.get("s"),
                z=params.get("z"),
                inplace_safe=params.get("inplace_safe", False),
                chunk_size=params.get("chunk_size"),
            )
        else:  # New interface
            # Ensure shapes are compatible
            if len(args) >= 4:  # We have s and z
                s, z = args[2], args[3]
                if s is not None and z is not None:
                    if s.shape[1] != z.shape[1]:
                        min_len = min(s.shape[1], z.shape[1])
                        s = s[:, :min_len, :]
                        z = z[:, :min_len, :]
                        args = list(args)
                        args[2] = s
                        args[3] = z
            # Call the captured original function directly, passing self
            return _original_forward_captured(self, *args, **kwargs)

    # Patch the forward method
    # atom_attention_encoder.AtomAttentionEncoder.forward = patched_forward # Disable potentially problematic patch
    # Set the flag to indicate the patch has been applied
    # atom_attention_encoder.AtomAttentionEncoder._patch_applied_forward = True
    print("[INFO] AtomAttentionEncoder forward patch disabled.")

    # Ensure backward compatibility for initialization
    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        # Handle both old and new initialization
        if "c_atom" in kwargs:  # Old interface
            kwargs["hidden_dim"] = kwargs.pop("c_atom")
        return original_init(self, *args, **kwargs)

    atom_attention_encoder.AtomAttentionEncoder.__init__ = patched_init


def fix_adaptive_layernorm():
    """
    Fix adaptive layer norm to handle shape mismatches.
    """
    # Corrected import from primitives instead of common
    from rna_predict.pipeline.stageA.input_embedding.current import primitives

    original_aln_forward = primitives.AdaptiveLayerNorm.forward

    @wraps(original_aln_forward)
    def patched_aln_forward(self, *args, **kwargs):
        try:
            return original_aln_forward(self, *args, **kwargs)
        except RuntimeError as e:
            if "The size of tensor a" in str(
                e
            ) and "must match the size of tensor b" in str(e):
                # Handle shape mismatch in adaptive layer norm
                x, conditioning = args[0], args[1]
                if x.shape[1] != conditioning.shape[1]:
                    min_len = min(x.shape[1], conditioning.shape[1])
                    x = x[:, :min_len, :]
                    conditioning = conditioning[:, :min_len, :]
                return original_aln_forward(self, x, conditioning, *args[2:], **kwargs)
            raise

    # Assign patched function back to the correct module
    primitives.AdaptiveLayerNorm.forward = patched_aln_forward

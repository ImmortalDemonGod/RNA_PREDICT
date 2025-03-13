import os
import sys
import torch
from rna_predict.models.encoder.input_feature_embedding import InputFeatureEmbedder
from rna_predict.dataset_loader import stream_bprna_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

###############################################################################
# Example Usage
###############################################################################


def demo_run_input_embedding():
    """
    Demonstration of how to run the input embedding module on toy data.
    """
    # Define number of atoms, tokens, and local attention block size.
    N_atom = 100
    N_token = 30
    block_size = 16

    # Build a feature dictionary.
    f = {}
    # Per-atom features.
    f["ref_pos"] = torch.randn(N_atom, 3)
    f["ref_charge"] = torch.randint(-2, 3, (N_atom,)).float()
    f["ref_element"] = torch.randn(N_atom, 128)
    f["ref_atom_name_chars"] = torch.randn(N_atom, 16)
    f["atom_to_token"] = torch.randint(0, N_token, (N_atom,))

    # Per-token features.
    f["restype"] = torch.randn(N_token, 32)
    f["profile"] = torch.randn(N_token, 32)
    f["deletion_mean"] = torch.randn(N_token)

    # For each atom, randomly select neighbor indices (for local attention).
    block_index = torch.randint(0, N_atom, (N_atom, block_size))

    # Instantiate the high-level embedder, toggling use_optimized to True/False
    embedder = InputFeatureEmbedder(
        c_token=384,
        restype_dim=32,
        profile_dim=32,
        c_atom=128,
        c_pair=32,
        num_heads=4,
        num_layers=3,
        use_optimized=True,  # <--- toggle here
    )

    # Forward pass.
    single_emb = embedder(f, trunk_sing=None, trunk_pair=None, block_index=block_index)
    print(
        "Output single-token embedding shape:", single_emb.shape
    )  # Expected: [N_token, c_token]


def demo_stream_bprna():
    """
    Demonstration of how to call stream_bprna_dataset() and iterate a few rows.
    """
    bprna_stream = stream_bprna_dataset(split="train")
    for idx, row in enumerate(bprna_stream):
        seq_len = len(row["sequence"])
        print(f"Row {idx}: id={row['id']}, sequence length={seq_len}")
        if idx >= 4:  # stop after 5 rows
            break

if __name__ == "__main__":
    print("Running demo_run_input_embedding()...")
    demo_run_input_embedding()
    print("\nNow streaming the bprna-spot dataset...")
    demo_stream_bprna()

import pandas as pd
import pathlib
import textwrap
import sys
import itertools
import numpy as np
import logging # Added for consistency if we want to log from here

from rna_predict.kaggle.data_utils import auto_column

# Helper function (previously local to the sanity check cell)
def preview(s, n=5):
    lst = list(s)
    return ", ".join(lst[:n]) + (" …" if len(lst) > n else "")

# Helper function (previously local to the sanity check cell)
def unique_triplet_count(row, tolerance: float):
    """Return #unique (x,y,z) triplets in a 5×3 slice."""
    uniq = []
    for v in row:
        if not any(np.allclose(v, u, atol=tolerance) for u in uniq):
            uniq.append(v)
    return len(uniq)

def run_sanity_checks(test_csv_path: str, submission_csv_path: str, tolerance: float = 1e-5):
    """
    Performs a series of sanity checks on a submission.csv file against a test_sequences.csv file.
    """
    logging.info(f"Running sanity checks: Test CSV='{test_csv_path}', Submission CSV='{submission_csv_path}'")

    # ── 1)  load / basic info ───────────────────────────────────────────────
    for f_path_str in (test_csv_path, submission_csv_path):
        if not pathlib.Path(f_path_str).is_file():
            logging.error(f"[ERROR] File not found: {f_path_str}")
            sys.exit(f"[ERROR] {f_path_str} not found!") # Or raise an error

    test_sequences = pd.read_csv(test_csv_path)
    submission     = pd.read_csv(submission_csv_path)

    id_col_test = auto_column(test_sequences, ["ID", "id", "seq_id", "sequence_id"])
    id_col_sub  = auto_column(submission,     ["ID", "id", "seq_id", "sequence_id"])

    # ── 2)  expected vs actual rows ─────────────────────────────────────────
    expected_rows = test_sequences["sequence"].str.len().sum()
    print("\n━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Expected rows        : {expected_rows:,}")
    print(f"submission.csv rows  : {len(submission):,}")
    dupes = submission[id_col_sub].duplicated().sum()
    print(f"Duplicate {id_col_sub!r} rows : {dupes:,}")

    # ── 3)  build the *full* ID set   \"<sequenceID>_<resIdx>\"  ─────────────
    full_id_set = {
        f"{sid}_{idx}"
        for sid, seq in zip(test_sequences[id_col_test], test_sequences["sequence"])
        for idx in range(1, len(seq) + 1)
    }
    sub_id_set = set(submission[id_col_sub].astype(str))

    missing = full_id_set - sub_id_set
    extra   = sub_id_set  - full_id_set

    print("\n━━ ID reconciliation ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"IDs missing from submission : {len(missing):,}")
    print(f"Unexpected extra IDs        : {len(extra):,}")
    if missing: print("  → first few missing :", preview(missing))
    if extra:   print("  → first few extras  :", preview(extra))

    # ── 4)  per-sequence coverage (how many residues per sequence?) ────────
    seq_len_series = test_sequences.set_index(id_col_test)["sequence"].str.len()

    prefixes = (
        submission[id_col_sub]
        .astype(str)
        .str.rsplit("_", n=1, expand=True)[0]
    )

    coverage = prefixes.value_counts().reindex(seq_len_series.index).fillna(0).astype(int)
    bad_cov  = coverage[coverage != seq_len_series]

    print("\n━━ Per-sequence coverage ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Sequences with wrong #rows : {len(bad_cov):,}")
    if len(bad_cov):
        print("  id  | expected | got")
        for sid, got in itertools.islice(bad_cov.items(), 5):
            print(f" {sid:<6}| {seq_len_series[sid]:>8} | {got}")

    # ── 5)  column sanity ───────────────────────────────────────────────────
    REQ_COLS = ["ID", "resname", "resid"] + [f"{ax}_{i}" for i in range(1, 6) for ax in "xyz"]
    missing_cols = [c for c in REQ_COLS if c not in submission.columns]

    print("\n━━ Column sanity ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Missing required columns   : {len(missing_cols)}")
    if missing_cols:
        print(textwrap.fill(", ".join(missing_cols), width=88))

    # ── 6)  structure-repeat uniqueness ────────────────────────────────────
    trip_cols = np.array([[f"{ax}_{i}" for ax in "xyz"] for i in range(1, 6)])
    # Ensure all required columns are present before trying to access them
    if not missing_cols:
        coords = submission[trip_cols.flatten()].values.reshape(len(submission), 5, 3)
        
        # Pass tolerance to unique_triplet_count
        uniq_counts = np.array([unique_triplet_count(row, tolerance) for row in coords])

        all_identical = (uniq_counts == 1).sum()
        truly_unique  = (uniq_counts > 1).sum()

        print("\n━━ Structure-repeat uniqueness ━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Rows where 5 structures are identical : {all_identical:,}")
        print(f"Rows with ≥2 distinct triplets         : {truly_unique:,}")

        sub_seq_id = prefixes.to_numpy()
        per_seq_unique = (
            pd.Series(uniq_counts > 1, index=sub_seq_id)
              .groupby(level=0).mean()
              .sort_values(ascending=False)
        )

        print("\nTop 5 sequences with most unique repeats:")
        for sid, frac in per_seq_unique.head(5).items():
            print(f"  {sid:<6}: {frac:6.1%} rows diversified")
    else:
        print("\n━━ Structure-repeat uniqueness (SKIPPED due to missing columns) ━━")


    print("\n✅  Sanity check finished.")

if __name__ == '__main__':
    # Example usage if run as a script
    # This part would require setting up TEST_CSV and SUB_CSV paths appropriately
    # For now, it's primarily designed to be imported and called as a function.
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Replace with actual paths or command-line argument parsing
    test_file = "/kaggle/input/stanford-rna-3d-folding/test_sequences.csv" 
    submission_file = "submission.csv"

    if pathlib.Path(test_file).exists() and pathlib.Path(submission_file).exists():
        run_sanity_checks(test_file, submission_file)
    else:
        print(f"Error: Ensure '{test_file}' and '{submission_file}' exist to run the standalone validator.")

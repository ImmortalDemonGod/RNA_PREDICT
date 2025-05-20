"""
Entry point for RNA_PREDICT inference. Loads a checkpoint, runs batch prediction, saves outputs.
Hydra-configurable. Supports partial checkpoint loading (M2 requirement).
"""

import torch
import hydra
from omegaconf import DictConfig
import pandas as pd
import os
from typing import List, Optional, Dict, Any

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.utils.submission import coords_to_df, extract_atom, reshape_coords

class RNAPredictor:
    """High-level interface for the RNA_PREDICT pipeline."""
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes an RNAPredictor instance with the provided configuration.
        
        Validates the presence of required configuration sections for model inference, including model stageC, prediction parameters, and torsion_bert settings. Sets up the computation device and initializes the torsion angle predictor.
        
        Raises:
            ValueError: If required configuration sections or fields are missing.
        """
        self.device = cfg.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageC"):
            raise ValueError("Configuration must contain model.stageC section")
        self.stageC_config = cfg.model.stageC
        # Prediction config retrieved via Hydra structured config
        if not hasattr(cfg, "prediction"):
            raise ValueError("Configuration must contain prediction section")
        self.prediction_config = cfg.prediction
        self.default_repeats = self.prediction_config.repeats
        self.default_atom_choice = self.prediction_config.residue_atom_choice
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'torsion_bert'):
            torsion_bert_cfg = cfg.model.stageB.torsion_bert
        else:
            torsion_bert_cfg = cfg
        if not hasattr(torsion_bert_cfg, 'model_name_or_path'):
            raise ValueError("torsion_bert_cfg must specify model_name_or_path in the Hydra config.")
        if not hasattr(torsion_bert_cfg, 'device'):
            raise ValueError("torsion_bert_cfg must specify device in the Hydra config.")
        self.torsion_predictor = StageBTorsionBertPredictor(torsion_bert_cfg)

    #@snoop
    def predict_3d_structure(self, sequence: str, stochastic_pass: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Predicts the 3D structure of an RNA sequence using torsion angle prediction and reconstruction.
        
        If the input sequence is empty, returns empty coordinate tensors and zero atom count. Otherwise, predicts torsion angles (optionally with stochastic inference and a random seed), prepares the reconstruction configuration, and generates 3D atomic coordinates.
        
        Args:
            sequence: RNA sequence to predict the structure for.
            stochastic_pass: Whether to enable stochastic inference for torsion angle prediction.
            seed: Optional random seed for reproducibility during stochastic inference.
        
        Returns:
            A dictionary containing predicted atom coordinates, 3D coordinates, and atom count.
        """
        print(f"[DEBUG][predict_3d_structure] sequence type: {type(sequence)}, value: {sequence}")
        if not sequence:
            return {
                "coords": torch.empty((0, 3), device=self.device),
                "coords_3d": torch.empty((0, 0, 3), device=self.device),
                "atom_count": 0,
            }
        torsion_output = self.torsion_predictor(sequence, stochastic_pass=stochastic_pass, seed=seed)
        torsion_angles = torsion_output["torsion_angles"]
        from omegaconf import OmegaConf
        # Construct config with model.stageC for run_stageC
        base_stagec = self.stageC_config
        # Convert dataclass to dict if needed
        if hasattr(base_stagec, '__dataclass_fields__'):
            base_stagec = OmegaConf.structured(base_stagec)
        if OmegaConf.is_config(base_stagec):
            base_stagec = OmegaConf.to_container(base_stagec, resolve=True)
        stageC_full_config = OmegaConf.create({"model": {"stageC": base_stagec}})
        print("[DEBUG] type(stageC_full_config):", type(stageC_full_config))
        print("[DEBUG] stageC_full_config keys:", list(stageC_full_config.keys()))
        print("[DEBUG] stageC_full_config['model'] keys:", list(stageC_full_config['model'].keys()))
        return run_stageC(cfg=stageC_full_config, sequence=sequence, torsion_angles=torsion_angles)

    def predict_submission(self, sequence: str, prediction_repeats: Optional[int] = None, residue_atom_choice: Optional[int] = None, repeat_seeds: Optional[list] = None) -> pd.DataFrame:
        """
        Generates a DataFrame containing multiple predicted RNA 3D structures for a given sequence, supporting stochastic inference and repeat-specific seeding.
        
        If stochastic inference is enabled in the configuration, each repeat uses a unique random seed for reproducibility. For each repeat, the function predicts atom coordinates, extracts the specified atom type, and collects results into a DataFrame with separate coordinate columns for each repeat. Handles empty sequences by returning an empty DataFrame with the appropriate structure.
        
        Args:
            sequence: RNA sequence to predict.
            prediction_repeats: Number of prediction repeats. If not provided, uses the default from config.
            residue_atom_choice: Index of the atom type to extract per residue. Defaults to config value if not provided.
            repeat_seeds: Optional list of seeds for each repeat to ensure reproducibility.
        
        Returns:
            pd.DataFrame: DataFrame with atom IDs, residue names, residue indices, and x/y/z coordinates for each repeat.
        """
        import logging
        import random
        import numpy as np
        logger = logging.getLogger("rna_predict.predict_submission")

        # Read stochastic config from Hydra
        enable_stochastic = self.prediction_config.enable_stochastic_inference_for_submission
        logger.info(f"[DEBUG-CONFIG-PROPAGATION] enable_stochastic_inference_for_submission={enable_stochastic} (from config), passing as stochastic_pass={enable_stochastic}")
        repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats

        def set_all_seeds(seed):
            """
            Sets the random seed for Python, NumPy, and PyTorch to ensure reproducible results.
            
            Args:
            	seed: The seed value to use for all random number generators.
            """
            random.seed(seed)
            np.random.seed(seed)
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if not sequence:
            df = coords_to_df("", torch.empty(0, 3, device=self.device), repeats)
            import os
            if os.environ.get("DEBUG_PREDICT_SUBMISSION") == "1":
                print("[DEBUG] predict_submission DataFrame shape:", df.shape)
                print("[DEBUG] predict_submission DataFrame columns:", list(df.columns))
                print("[DEBUG] predict_submission DataFrame head:\n", df.head())
            logger.warning(f"[DEBUG] Returning DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            return df

        results = []
        atom_names = None
        residue_indices = None
        for i in range(repeats):
            # Determine seed for this repeat if stochastic enabled
            seed = None
            if enable_stochastic:
                if repeat_seeds and i < len(repeat_seeds):
                    seed = repeat_seeds[i]
                else:
                    seed = i  # Use index as seed for reproducibility
                set_all_seeds(seed)
            elif repeat_seeds and i < len(repeat_seeds):
                seed = repeat_seeds[i]
                set_all_seeds(seed)
            # Run prediction with stochastic_pass and seed
            result = self.predict_3d_structure(sequence, stochastic_pass=enable_stochastic, seed=seed)
            coords = reshape_coords(result['coords'], len(sequence))
            # Select atom coords
            if coords.dim() == 3:
                atom_choice = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice
                if atom_choice < 0 or atom_choice >= coords.shape[1]:
                    raise IndexError(f"Invalid residue_atom_choice {atom_choice} for coords shape {coords.shape}")
                atom_coords = extract_atom(coords, atom_choice).detach().cpu()
            elif coords.dim() == 2:
                atom_coords = coords.detach().cpu()
            else:
                raise ValueError(f"Unexpected coords shape: {coords.shape}")
            results.append(atom_coords)
            # Save atom metadata from first repeat
            if atom_names is None or residue_indices is None:
                atom_metadata = result.get('atom_metadata') or {}
                atom_names = atom_metadata.get('atom_names', ['P'] * atom_coords.shape[0])
                residue_indices = atom_metadata.get('residue_indices', list(range(atom_coords.shape[0])))
                if residue_indices is None:
                    residue_indices = list(range(atom_coords.shape[0]))
                atom_names = atom_names[:atom_coords.shape[0]]
                residue_indices = residue_indices[:atom_coords.shape[0]]

        n_atoms = results[0].shape[0] if results else 0

        # Clamp residue_indices to valid range
        if residue_indices is None:
            residue_indices = list(range(atom_coords.shape[0]))
        valid_indices = [i if i < len(sequence) else len(sequence)-1 for i in residue_indices]
        if any(i >= len(sequence) for i in residue_indices):
            print(f"[WARN] Clamped out-of-range residue_indices: {residue_indices} (sequence len={len(sequence)})")
        base_data = {
            'ID': list(range(1, n_atoms + 1)),
            'resname': [sequence[i] for i in valid_indices],
            'resid': [i + 1 for i in valid_indices],
        }
        # Coordinate columns for each repeat
        if results is not None:
            for i, atom_coords in enumerate(results):
                base_data[f'x_{i+1}'] = atom_coords[:, 0].tolist()
                base_data[f'y_{i+1}'] = atom_coords[:, 1].tolist()
                base_data[f'z_{i+1}'] = atom_coords[:, 2].tolist()
        return pd.DataFrame(base_data)

    def predict_submission_original(self, sequence: str, prediction_repeats: Optional[int] = None, residue_atom_choice: Optional[int] = None, repeat_seeds: Optional[list] = None) -> pd.DataFrame:
        """
        Generates a DataFrame with repeated 3D atom coordinates for a given RNA sequence.
        
        If the sequence is empty, returns an empty DataFrame with the appropriate columns for the specified number of repeats. Otherwise, predicts the 3D structure once and replicates the predicted atom coordinates across all repeats. Atom coordinates are selected based on the specified or default residue atom choice.
        
        Args:
            sequence: RNA sequence to predict.
            prediction_repeats: Number of times to repeat the predicted coordinates in the output DataFrame. If not provided, uses the default from configuration.
            residue_atom_choice: Index of the atom to extract per residue. If not provided, uses the default from configuration.
            repeat_seeds: Ignored in this method.
        
        Returns:
            pd.DataFrame: DataFrame containing atom IDs, residue names, residue indices, and x/y/z coordinates for each repeat.
        """
        import logging
        logger = logging.getLogger("rna_predict.predict_submission")
        if not sequence:
            repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
            df = coords_to_df("", torch.empty(0, 3, device=self.device), repeats)
            import os
            if os.environ.get("DEBUG_PREDICT_SUBMISSION") == "1":
                print("[DEBUG] predict_submission DataFrame shape:", df.shape)
                print("[DEBUG] predict_submission DataFrame columns:", list(df.columns))
                print("[DEBUG] predict_submission DataFrame head:\n", df.head())
            logger.warning(f"[DEBUG] Returning DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            return df
        
        # Single predict_3d_structure call
        result = self.predict_3d_structure(sequence)
        coords = reshape_coords(result['coords'], len(sequence))
        # Select atom coords
        if coords.dim() == 3:
            atom_choice = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice
            if atom_choice < 0 or atom_choice >= coords.shape[1]:
                raise IndexError(f"Invalid residue_atom_choice {atom_choice} for coords shape {coords.shape}")
            atom_coords = extract_atom(coords, atom_choice).detach().cpu()
        elif coords.dim() == 2:
            atom_coords = coords.detach().cpu()
        else:
            raise ValueError(f"Unexpected coords shape: {coords.shape}")

        n_atoms = atom_coords.shape[0]
        # Metadata fallback
        atom_metadata = result.get('atom_metadata') or {}
        atom_names = atom_metadata.get('atom_names', ['P'] * n_atoms)
        residue_indices = atom_metadata.get('residue_indices', list(range(n_atoms)))
        atom_names = atom_names[:n_atoms]
        residue_indices = residue_indices[:n_atoms]

        # Build DataFrame
        base_data = {
            'ID': list(range(1, n_atoms + 1)),
            'resname': [sequence[i] for i in residue_indices],
            'resid': [i + 1 for i in residue_indices],
        }
        # Coordinate columns
        x_vals = atom_coords[:, 0].tolist()
        y_vals = atom_coords[:, 1].tolist()
        z_vals = atom_coords[:, 2].tolist()
        repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
        for i in range(1, repeats + 1):
            base_data[f'x_{i}'] = x_vals
            base_data[f'y_{i}'] = y_vals
            base_data[f'z_{i}'] = z_vals
        return pd.DataFrame(base_data)


def load_partial_checkpoint(model, checkpoint_path):
    """
    Loads model parameters from a checkpoint, updating only matching keys and shapes.
    
    Only parameters present in both the checkpoint and the model with identical shapes are loaded; others are ignored. The model is updated in-place and returned.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    return model


def batch_predict(predictor: RNAPredictor, sequences: List[str], output_dir: str):
    """
    Generates and saves 3D structure predictions for a batch of RNA sequences.
    
    For each input sequence, predicts its 3D structure using the provided predictor, then saves the coordinates for each prediction repeat as separate CSV and PDB files in the specified output directory. Also writes a summary CSV with metadata for all predictions, including atom counts and number of repeats.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for i, seq in enumerate(sequences):
        print(f"[DEBUG][batch_predict] seq type: {type(seq)}, value: {seq}")
        submission_df = predictor.predict_submission(seq)
        # Save each repeat's coordinates as separate CSV and PDB files
        n_repeats = (submission_df.shape[1] - 3) // 3  # x/y/z per repeat, columns are: ID, resname, resid, x_1, y_1, z_1, ...
        for rep in range(n_repeats):
            x_col = f"x_{rep+1}"
            y_col = f"y_{rep+1}"
            z_col = f"z_{rep+1}"
            rep_df = submission_df[["ID", "resname", "resid", x_col, y_col, z_col]].copy()
            rep_df.rename(columns={x_col: "x", y_col: "y", z_col: "z"}, inplace=True)
            csv_path = os.path.join(output_dir, f"prediction_{i}_repeat_{rep+1}.csv")
            rep_df.to_csv(csv_path, index=False)
            # Save as PDB
            def write_pdb(df, pdb_path):
                """
                Writes RNA atom coordinates from a DataFrame to a PDB file.
                
                Each row in the DataFrame should contain 'resname', 'resid', 'x', 'y', and 'z' columns representing atom metadata and 3D coordinates.
                """
                with open(pdb_path, "w") as f:
                    for idx, row in df.iterrows():
                        atom = row["resname"]
                        res_idx = row["resid"]
                        f.write(f"ATOM  {idx+1:5d} {atom:>4} RNA A{res_idx:4d}    {row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}  1.00  0.00           {atom[0]:>2}\n")
            pdb_path = os.path.join(output_dir, f"prediction_{i}_repeat_{rep+1}.pdb")
            write_pdb(rep_df, pdb_path)
        # For summary, use atom count from first repeat
        atom_count = rep_df.shape[0] if n_repeats > 0 else 0
        results.append({"index": i, "atom_count": atom_count, "repeats": n_repeats})

    pd.DataFrame(results).to_csv(os.path.join(output_dir, "summary.csv"), index=False)


@hydra.main(version_base=None, config_path="conf", config_name="predict")
def main(cfg: DictConfig):
    """
    Entry point for RNA_PREDICT inference. Loads a checkpoint, runs batch prediction, saves outputs.
    Hydra-configurable. Supports partial checkpoint loading (M2 requirement).
    Now supports fast_dev_run/limit_n for quick debugging runs.
    """
    print("[INFO] Starting RNA_PREDICT inference...")
    fast_dev_run = getattr(cfg, "fast_dev_run", False)
    limit_n = getattr(cfg, "limit_n", None)
    if fast_dev_run:
        print("[INFO] fast_dev_run is enabled; limiting to 1 sequence.")
        limit_n = 1
    if limit_n is not None:
        print(f"[INFO] Limiting inference to first {limit_n} sequence(s).")

    import hydra
    import os
    # Hydra best practice: resolve all paths relative to original project root
    original_cwd = hydra.utils.get_original_cwd()
    print("[DEBUG] Hydra runtime CWD:", os.getcwd())
    print("[DEBUG] Hydra original_cwd:", original_cwd)
    # Resolve paths
    input_csv = cfg.get("input_csv", "validation.csv")
    if not os.path.isabs(input_csv):
        input_csv = os.path.join(original_cwd, input_csv)
    output_dir = cfg.get("output_dir", "outputs/")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(original_cwd, output_dir)
    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path and not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(original_cwd, checkpoint_path)
    print("[DEBUG] Resolved input_csv:", input_csv)
    print("[DEBUG] Resolved output_dir:", output_dir)
    print("[DEBUG] Resolved checkpoint_path:", checkpoint_path)
    df = pd.read_csv(input_csv)
    sequences = []
    seq_paths = []
    if "sequence" in df.columns:
        print("[DEBUG] Using 'sequence' column from CSV.")
        sequences = df["sequence"].tolist()
    elif "sequence_path" in df.columns:
        print("[DEBUG] Using 'sequence_path' column from CSV. Reading sequences from files...")
        for seq_path in df["sequence_path"].tolist():
            try:
                with open(seq_path, "r") as f:
                    lines = f.readlines()
                if fast_dev_run or limit_n is not None:
                    print(f"[DIAG] First 10 lines of {seq_path}:")
                    for line in lines[:10]:
                        print(f"[DIAG] {line.rstrip()}")
                    print(f"[DIAG] Total lines in {seq_path}: {len(lines)}")
                # Find all valid RNA sequences (continuous A/C/G/U runs)
                import re
                valid_seqs = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith(">"):
                        continue
                    # Only keep lines with all A/C/G/U
                    if re.fullmatch(r"[ACGU]+", line):
                        valid_seqs.append(line)
                if not valid_seqs:
                    print(f"[ERROR] No valid RNA sequence found in {seq_path}!")
                    sequences.append("")
                    seq_paths.append(seq_path)
                elif len(valid_seqs) == 1:
                    sequences.append(valid_seqs[0])
                    seq_paths.append(seq_path)
                else:
                    print(f"[WARNING] Multiple valid RNA sequences found in {seq_path}. Using only the first.")
                    sequences.append(valid_seqs[0])
                    seq_paths.append(seq_path)
            except Exception as e:
                print(f"[ERROR] Could not read sequence file at {seq_path}: {e}")
                sequences.append("")
                seq_paths.append(seq_path)
    else:
        raise ValueError("Input CSV must contain either a 'sequence' or 'sequence_path' column.")
    if limit_n is not None:
        sequences = sequences[:limit_n]
        seq_paths = seq_paths[:limit_n]
        print(f"[DEBUG] Processing {len(sequences)} sequence(s) in dev/limit mode.")
        for i, (seq, seq_path) in enumerate(zip(sequences, seq_paths)):
            print(f"[DEBUG] Sequence {i}: {seq}")
            # Validate sequence contains only A/C/G/U
            invalid = [c for c in seq if c not in "ACGU"]
            if invalid:
                print(f"[ERROR] Invalid character(s) found in sequence {i}: {invalid}")
                print(f"[ERROR] Offending sequence: {seq}")
                if seq_path:
                    print(f"[ERROR] Sequence file path: {seq_path}")
                raise ValueError(f"Sequence {i} contains invalid character(s): {invalid}. Must only contain A, C, G, U.")
    predictor = RNAPredictor(cfg)
    if checkpoint_path:
        predictor.torsion_predictor = load_partial_checkpoint(predictor.torsion_predictor, checkpoint_path)
    batch_predict(predictor, sequences, output_dir)
    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    main()

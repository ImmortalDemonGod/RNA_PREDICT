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
import logging # Added import
import numpy as np # Add this import

from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
from rna_predict.pipeline.stageC.stage_c_reconstruction import run_stageC
from rna_predict.utils.submission import coords_to_df, extract_atom, reshape_coords

logger = logging.getLogger(__name__) # Added logger initialization

class RNAPredictor:
    """High-level interface for the RNA_PREDICT pipeline."""
    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the RNAPredictor with configuration for model inference.
        
        Args:
            cfg: Hydra DictConfig containing model, device, and prediction parameters.
        
        Raises:
            ValueError: If required configuration sections or fields are missing.
        """
        # --- Debug logging: print full config and prediction section ---
        try:
            from omegaconf import OmegaConf
            logger.info("[DEBUG] RNAPredictor received config:\n" + OmegaConf.to_yaml(cfg))
            if hasattr(cfg, 'prediction'):
                logger.info("[DEBUG] RNAPredictor prediction config section:\n" + OmegaConf.to_yaml(cfg.prediction))
            else:
                logger.info("[DEBUG] No 'prediction' section in config!")
        except Exception as e:
            logger.info(f"[DEBUG] Could not print config in RNAPredictor.__init__: {e}")

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
        
        If the input sequence is empty, returns empty coordinate tensors and zero atom count. Otherwise, predicts torsion angles, prepares the configuration for stage C reconstruction, and generates 3D coordinates and related outputs.
        
        Args:
            sequence: RNA sequence string to predict the structure for.
        
        Returns:
            A dictionary containing predicted coordinates, 3D coordinates, and atom count.
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

    def _extract_per_residue_coords_from_result(self, result: dict, sequence: str, atom_choice: int) -> torch.Tensor:
        """
        Processes raw coordinates from a single predict_3d_structure result to extract
        one set of [x, y, z] coordinates per residue in the sequence.

        Handles:
        - Flat input coordinates (e.g., [N_total_atoms, 3]) by selecting 'P' atom or fallback.
        - Reshaped input coordinates (e.g., [L, n_atoms_per_res, 3]).
        - Ensures output tensor is always [len(sequence), 3].
        - Fills with NaNs for residues entirely missing from input data.
        
        Args:
            result: Output dictionary from self.predict_3d_structure.
            sequence: The RNA sequence string (to determine L).
            atom_choice: Default atom index to pick if coords are [L, n_atoms_per_res, 3].

        Returns:
            torch.Tensor: Coordinates of shape [len(sequence), 3] on CPU.
        """
        coords_raw = result['coords']
        metadata = result.get('atom_metadata', {})
        atom_names_meta = metadata.get('atom_names', [])
        residue_indices_meta = metadata.get('residue_indices', []) # 0-based
        sequence_len = len(sequence)

        # Ensure coords_raw is detached before potential numpy conversion or further processing
        if coords_raw.requires_grad:
            coords_raw = coords_raw.detach()

        final_per_residue_coords = torch.full((sequence_len, 3), float('nan'), dtype=coords_raw.dtype, device='cpu')

        # Determine if we need to process flat coordinates
        needs_flat_processing = (coords_raw.dim() == 2 and coords_raw.shape[0] != sequence_len)

        if not needs_flat_processing:
            # Try reshaping first; this might also lead to flat processing if reshape fails
            coords_reshaped = reshape_coords(coords_raw, sequence_len)

            if coords_reshaped.dim() == 2 and coords_reshaped.shape[0] != sequence_len:
                logger.warning(
                    f"""[COORD_EXTRACTION] reshape_coords resulted in flat coordinates ({coords_reshaped.shape}) 
                    that are not per-residue for sequence length {sequence_len}. Switching to flat processing logic."""
                )
                coords_raw = coords_reshaped # Update coords_raw to be the output of failed reshape
                needs_flat_processing = True
            elif coords_reshaped.dim() == 3:
                if atom_choice < 0 or atom_choice >= coords_reshaped.shape[1]:
                    raise IndexError(f"Invalid atom_choice {atom_choice} for reshaped coords shape {coords_reshaped.shape}")
                final_per_residue_coords = extract_atom(coords_reshaped, atom_choice).cpu()
                needs_flat_processing = False # Successfully processed
            elif coords_reshaped.dim() == 2: # Assumed to be [sequence_len, 3]
                final_per_residue_coords = coords_reshaped.cpu()
                needs_flat_processing = False # Successfully processed
            else:
                raise ValueError(f"Unexpected coords shape after reshape: {coords_reshaped.shape}")
        
        if needs_flat_processing:
            logger.info(f"""[COORD_EXTRACTION] 
                Processing flat coordinates of shape {coords_raw.shape} 
                for sequence length {sequence_len}.""")
            if not atom_names_meta or not residue_indices_meta:
                logger.error(
                    f"""[COORD_EXTRACTION_ERROR] 
                    Missing atom metadata (names or indices) for flat coordinates. 
                    Shape was {coords_raw.shape}, seq_len {sequence_len}. 
                    Output for this repeat will be NaNs."""
                )
                # final_per_residue_coords is already NaNs, so we can return
                return final_per_residue_coords

            if len(atom_names_meta) != coords_raw.shape[0] or len(residue_indices_meta) != coords_raw.shape[0]:
                logger.error(
                    f"""[COORD_EXTRACTION_ERROR] 
                    Metadata length mismatch with flat coordinates. 
                    Names: {len(atom_names_meta)}, 
                    Indices: {len(residue_indices_meta)}, 
                    Coords: {coords_raw.shape[0]}. 
                    Cannot reliably select atoms. 
                    Output for this repeat will be NaNs."""
                )
                return final_per_residue_coords

            # Ensure coords_raw is on CPU for numpy conversion if not already
            coords_raw_cpu = coords_raw.cpu()
            tmp_df = pd.DataFrame({
                "atom_name": atom_names_meta,
                "res0": residue_indices_meta, # 0-based residue index
                "x": coords_raw_cpu[:, 0].numpy(),
                "y": coords_raw_cpu[:, 1].numpy(),
                "z": coords_raw_cpu[:, 2].numpy(),
            })

            for i in range(sequence_len):
                residue_atoms = tmp_df[tmp_df["res0"] == i]
                if residue_atoms.empty:
                    logger.warning(f"[COORD_EXTRACTION] No atoms found for residue index {i} in flat data. Coords will be NaN.")
                    continue # NaN already in final_per_residue_coords[i]

                p_atom = residue_atoms[residue_atoms["atom_name"] == "P"]
                if not p_atom.empty:
                    chosen_atom_series = p_atom.iloc[0]
                else:
                    chosen_atom_series = residue_atoms.iloc[0] # Fallback to first atom for this residue
                
                # Extract as numpy array, ensure float32, then convert to tensor of the target dtype
                xyz_numpy = chosen_atom_series[["x", "y", "z"]].values.astype(np.float32)
                xyz_tensor = torch.tensor(xyz_numpy, dtype=final_per_residue_coords.dtype)
                final_per_residue_coords[i] = xyz_tensor
        
        # Validate shape before returning
        if final_per_residue_coords.shape[0] != sequence_len or final_per_residue_coords.shape[1] != 3:
            # This should not happen if logic is correct, but as a safeguard:
            logger.error(f"[COORD_EXTRACTION_CRITICAL] Final selected coords shape {final_per_residue_coords.shape} is not [{sequence_len}, 3]. Returning NaNs.")
            return torch.full((sequence_len, 3), float('nan'), dtype=coords_raw.dtype, device='cpu')
            
        return final_per_residue_coords.to(dtype=torch.float32) # Ensure consistent float32 output

    def predict_submission(self, sequence: str, prediction_repeats: Optional[int] = None, residue_atom_choice: Optional[int] = None, repeat_seeds: Optional[list] = None) -> pd.DataFrame:
        """
        Generate a submission DataFrame with multiple unique structure predictions using stochastic inference if enabled in config.
        Ensures one row per residue in the output DataFrame.
        """
        import logging
        import random
        import numpy as np
        # logger for this method specifically
        method_logger = logging.getLogger("rna_predict.RNAPredictor.predict_submission") 

        # Read stochastic config from Hydra
        enable_stochastic = self.prediction_config.enable_stochastic_inference_for_submission
        method_logger.info(f"[DEBUG] Stochastic inference for submission: {enable_stochastic}")
        repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
        current_atom_choice = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice
        method_logger.info(f"[DEBUG] prediction_repeats: {prediction_repeats}, repeats used: {repeats}")
        method_logger.info(f"[DEBUG] residue_atom_choice: {residue_atom_choice}, atom_choice used: {current_atom_choice}")
        if hasattr(self.prediction_config, 'submission_seeds'):
            method_logger.info(f"[DEBUG] submission_seeds from config: {self.prediction_config.submission_seeds}")
        else:
            method_logger.info("[DEBUG] No submission_seeds in prediction config!")
        method_logger.info(f"[DEBUG] repeat_seeds argument: {repeat_seeds}")

        def set_all_seeds(seed_val):
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_val)

        if not sequence:
            method_logger.warning("Empty sequence provided. Returning empty DataFrame with submission columns.")
            # Create an empty DataFrame with expected columns for an empty sequence
            cols = ['ID', 'resname', 'resid']
            for r_idx in range(1, repeats + 1):
                cols.extend([f'x_{r_idx}', f'y_{r_idx}', f'z_{r_idx}'])
            return pd.DataFrame(columns=cols)

        results_coords_list = [] # List to store [L,3] tensors for each repeat
        
        for i in range(repeats):
            seed_for_repeat = None
            if enable_stochastic:
                if repeat_seeds and i < len(repeat_seeds):
                    seed_for_repeat = repeat_seeds[i]
                elif hasattr(self.prediction_config, 'submission_seeds') and i < len(self.prediction_config.submission_seeds):
                    seed_for_repeat = self.prediction_config.submission_seeds[i]
                else:
                    seed_for_repeat = i # Default seed progression
                method_logger.info(f"[DEBUG] Repeat {i+1}/{repeats} with seed: {seed_for_repeat} (stochastic enabled)")
                set_all_seeds(seed_for_repeat)
            elif repeat_seeds and i < len(repeat_seeds): # Seeds provided but stochastic disabled, still use them
                seed_for_repeat = repeat_seeds[i]
                method_logger.info(f"[DEBUG] Repeat {i+1}/{repeats} with provided seed (stochastic disabled): {seed_for_repeat}")
                set_all_seeds(seed_for_repeat)
            else:
                method_logger.info(f"[DEBUG] Repeat {i+1}/{repeats} (deterministic or default seed).")

            # Run prediction for one repeat
            method_logger.info(f"[DEBUG] Calling predict_3d_structure with stochastic_pass={enable_stochastic}, seed={seed_for_repeat}")
            prediction_result = self.predict_3d_structure(sequence, stochastic_pass=enable_stochastic, seed=seed_for_repeat)
            
            # Process coordinates to ensure [len(sequence), 3]
            per_residue_coords_for_repeat = self._extract_per_residue_coords_from_result(
                prediction_result, 
                sequence, 
                current_atom_choice
            )
            results_coords_list.append(per_residue_coords_for_repeat)

        # Construct the final DataFrame - now guaranteed one row per residue
        sequence_len = len(sequence)
        base_data = {
            'ID': list(range(1, sequence_len + 1)),
            'resname': list(sequence), # Each character in sequence is a resname
            'resid': list(range(1, sequence_len + 1)), # Residue IDs are 1-based
        }

        if not results_coords_list: # Should not happen if repeats > 0 and sequence not empty
            method_logger.warning("No coordinate results generated. Returning base DataFrame.")
            return pd.DataFrame(base_data)

        for i, atom_coords_tensor in enumerate(results_coords_list):
            if atom_coords_tensor.shape[0] != sequence_len:
                method_logger.error(
                    f"""Repeat {i+1} produced coords of shape {atom_coords_tensor.shape} 
                    instead of expected [{sequence_len}, 3]. Submission might be misaligned."""
                )
                # Fill with NaNs to maintain DataFrame structure if a repeat failed badly
                base_data[f'x_{i+1}'] = [float('nan')] * sequence_len
                base_data[f'y_{i+1}'] = [float('nan')] * sequence_len
                base_data[f'z_{i+1}'] = [float('nan')] * sequence_len
            else:
                base_data[f'x_{i+1}'] = atom_coords_tensor[:, 0].tolist()
                base_data[f'y_{i+1}'] = atom_coords_tensor[:, 1].tolist()
                base_data[f'z_{i+1}'] = atom_coords_tensor[:, 2].tolist()
        
        final_df = pd.DataFrame(base_data)
        if os.environ.get("DEBUG_PREDICT_SUBMISSION") == "1":
            method_logger.info(f"[DEBUG] predict_submission final DataFrame shape: {final_df.shape}")
            method_logger.info(f"[DEBUG] predict_submission final DataFrame columns: {list(final_df.columns)}")
            method_logger.info(f"[DEBUG] predict_submission final DataFrame head:\n{final_df.head()}")

        return final_df


    def predict_submission_original(self, sequence: str, prediction_repeats: Optional[int] = None, residue_atom_choice: Optional[int] = None, repeat_seeds: Optional[list] = None) -> pd.DataFrame:
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
    import logging
    logger = logging.getLogger("rna_predict.checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state = model.state_dict()
    # --- Debug: print first 10 keys from both dicts ---
    logger.info(f"[CHECKPOINT-DEBUG] Checkpoint state_dict keys (first 10): {list(state_dict.keys())[:10]}")
    logger.info(f"[CHECKPOINT-DEBUG] Model state_dict keys (first 10): {list(model_state.keys())[:10]}")
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    logger.info(f"[CHECKPOINT-LOAD] Loading checkpoint from: {checkpoint_path}")
    logger.info(f"[CHECKPOINT-LOAD] Keys loaded: {list(filtered.keys())[:10]} ... total: {len(filtered)}")
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
    logger.info("[CHECKPOINT-LOAD] Model state_dict updated with checkpoint.")
    return model


def batch_predict(predictor: RNAPredictor, sequences: List[str], output_dir: str):
    """
    Generates 3D structure predictions for a batch of RNA sequences and saves results.
    
    For each input sequence, predicts its 3D structure using the provided predictor, then saves the output in three formats: a PyTorch tensor file (.pt), a CSV file with atom coordinates, and a PDB file. Also writes a summary CSV with metadata for all predictions.
    
    Args:
        predictor: An RNAPredictor instance used for structure prediction.
        sequences: List of RNA sequences to process.
        output_dir: Directory where prediction outputs will be saved.
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
        print(f"[CHECKPOINT-LOAD] Successfully loaded checkpoint from: {checkpoint_path}")
    else:
        print("[CHECKPOINT-LOAD] No checkpoint path provided; using default/random weights.")
    batch_predict(predictor, sequences, output_dir)
    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    main()

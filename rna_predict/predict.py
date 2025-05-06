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
import snoop

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
        self.device = cfg.device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        if not hasattr(cfg, "model") or not hasattr(cfg.model, "stageC"):
            raise ValueError("Configuration must contain model.stageC section")
        self.stageC_config = cfg.model.stageC
        self.prediction_config = getattr(cfg, "prediction", {})
        self.default_repeats = getattr(self.prediction_config, "repeats", 5)
        self.default_atom_choice = getattr(self.prediction_config, "residue_atom_choice", 1)
        if hasattr(cfg, 'model') and hasattr(cfg.model, 'stageB') and hasattr(cfg.model.stageB, 'torsion_bert'):
            torsion_bert_cfg = cfg.model.stageB.torsion_bert
        else:
            torsion_bert_cfg = cfg
        if not hasattr(torsion_bert_cfg, 'model_name_or_path'):
            raise ValueError("torsion_bert_cfg must specify model_name_or_path in the Hydra config.")
        if not hasattr(torsion_bert_cfg, 'device'):
            raise ValueError("torsion_bert_cfg must specify device in the Hydra config.")
        self.torsion_predictor = StageBTorsionBertPredictor(torsion_bert_cfg)

    @snoop
    def predict_3d_structure(self, sequence: str) -> Dict[str, Any]:
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
        torsion_output = self.torsion_predictor(sequence)
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

    def predict_submission(self, sequence: str, prediction_repeats: Optional[int] = None, residue_atom_choice: Optional[int] = None) -> pd.DataFrame:
        if not sequence:
            repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
            return coords_to_df("", torch.empty(0, 3, device=self.device), repeats)
        result = self.predict_3d_structure(sequence)
        coords = result["coords"]
        if coords.dim() == 2 and coords.shape[0] != len(sequence):
            repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
            base_data = {
                "ID": range(1, coords.shape[0] + 1),
                "resname": ["X"] * coords.shape[0],
                "resid": range(1, coords.shape[0] + 1)
            }
            coords_np = coords.detach().cpu().numpy()
            for i in range(1, repeats + 1):
                base_data[f"x_{i}"] = coords_np[:, 0]
                base_data[f"y_{i}"] = coords_np[:, 1]
                base_data[f"z_{i}"] = coords_np[:, 2]
            df = pd.DataFrame(base_data)
            return df
        coords = reshape_coords(coords, len(sequence))
        repeats = prediction_repeats if prediction_repeats is not None else self.default_repeats
        atom_choice = residue_atom_choice if residue_atom_choice is not None else self.default_atom_choice
        if coords.dim() != 3 or atom_choice < 0 or atom_choice >= coords.shape[1]:
            raise IndexError(f"Invalid residue_atom_choice {atom_choice} for coords shape {coords.shape} (expected shape [N, atoms, 3])")
        atom_coords = extract_atom(coords, atom_choice)
        return coords_to_df(sequence, atom_coords, repeats)


def load_partial_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state, strict=False)
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
        out = predictor.predict_3d_structure(seq)
        # --- Save as .pt for internal/debugging (optional, can remove if not wanted) ---
        torch.save(out, os.path.join(output_dir, f"prediction_{i}.pt"))
        # --- Save as CSV: atom-level coordinates ---
        coords = out["coords"]
        atom_metadata = out.get("atom_metadata", {})
        atom_names = atom_metadata.get("atom_names", ["?"] * coords.shape[0])
        residue_indices = atom_metadata.get("residue_indices", [None] * coords.shape[0])
        import pandas as pd
        df = pd.DataFrame({
            "atom_name": atom_names,
            "residue_index": residue_indices,
            "x": coords[:, 0].detach().cpu().numpy(),
            "y": coords[:, 1].detach().cpu().numpy(),
            "z": coords[:, 2].detach().cpu().numpy(),
        })
        df.to_csv(os.path.join(output_dir, f"prediction_{i}.csv"), index=False)
        # --- Save as PDB ---
        def write_pdb(df, pdb_path):
            with open(pdb_path, "w") as f:
                for idx, row in df.iterrows():
                    atom = row["atom_name"]
                    res_idx = row["residue_index"] if row["residue_index"] is not None else 1
                    f.write(f"ATOM  {idx+1:5d} {atom:>4} RNA A{res_idx:4d}    {row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}  1.00  0.00           {atom[0]:>2}\n")
        write_pdb(df, os.path.join(output_dir, f"prediction_{i}.pdb"))
        results.append({"index": i, "atom_count": out["atom_count"]})
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

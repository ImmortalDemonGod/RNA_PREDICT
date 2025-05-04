# RNA_PREDICT Training & Loss Function: Hypothesis-Driven Analysis

**Date:** 2025-05-03  
**Branch:** `fix/training-loss-function-implementation`

## Purpose
This document systematically analyzes the current state of the training and loss function implementation in the RNA_PREDICT pipeline. The analysis follows a hypothesis-driven approach to confirm or reject key concerns about the pipeline's training signal and gradient flow, based on code inspection and evidence.

---

## Key Hypotheses & Evidence

### H1: The current training step only computes loss on Stage C outputs, not on Stage D outputs.
**Evidence:**
- The `training_step` method in `rna_lightning_module.py` computes loss using `predicted_coords = output["coords"]` (output from Stage C).
- No evidence of Stage D output being used for loss.
- The loss is MSE between these coordinates and ground truth.
**Conclusion:**
- **Accepted.** Loss is only on Stage C outputs.

### H2: Stage D (Diffusion) is not called at all during training.
**Evidence:**
- In both `forward` and `training_step`, there is no call to `self.stageD`.
- Stage D is instantiated but not invoked in the forward or training path.
**Conclusion:**
- **Accepted.** Stage D is not called during training.

### H3: Pairformer (Stage B) is not trained, as its outputs do not contribute to the loss.
**Evidence:**
- Pairformer outputs are calculated, but only Stage C output is used for loss.
- No direct or indirect use of Pairformer outputs in the loss computation.
- Gradients would not flow to Pairformer parameters.
**Conclusion:**
- **Accepted.** Pairformer is not being trained.

### H4: TorsionBERT is trained, but only indirectly via Stage C’s coordinate reconstruction.
**Evidence:**
- TorsionBERT outputs torsion angles, which are used by Stage C to generate coordinates.
- The loss is on coordinates, so gradients flow back through Stage C to TorsionBERT.
- There is no direct angle supervision.
**Conclusion:**
- **Accepted.** TorsionBERT is trained, but only indirectly.

### H5: There is no direct angle supervision loss (`L_angle`) in the current training step.
**Evidence:**
- No angle-based loss found in `training_step`.
- Only coordinate-based MSE loss is present.
**Conclusion:**
- **Accepted.** No direct angle supervision loss is implemented.

### H6: The loss function used for Stage D is not the correct denoising/diffusion objective.
**Evidence:**
- No diffusion/denoising loss is implemented.
- No noise sampling, no call to Stage D, no diffusion loss logic.
**Conclusion:**
- **Accepted.** Diffusion loss is not implemented.

---

## Summary Table

| Hypothesis | Status    | Evidence/Notes                                              |
|------------|-----------|------------------------------------------------------------|
| H1         | Accepted  | Loss is only on Stage C output coords                      |
| H2         | Accepted  | Stage D not called in forward/training_step                |
| H3         | Accepted  | Pairformer outputs not used in loss                        |
| H4         | Accepted  | TorsionBERT trained indirectly via coords                  |
| H5         | Accepted  | No direct angle loss present                               |
| H6         | Accepted  | No diffusion loss or Stage D integration                   |

---

## Conclusions & Recommendations

- The current training implementation is misaligned with the intended multi-stage pipeline design.
- Only TorsionBERT receives a training signal, and only indirectly.
- Pairformer and Stage D receive no training signal at all.
- No direct angle supervision or diffusion loss is present.

**Actionable Steps:**
1. Implement direct angle supervision (`L_angle`).
2. Integrate Stage D into the forward and training path.
3. Implement the correct diffusion loss objective.
4. Ensure gradients flow to all intended components (TorsionBERT, Pairformer, Stage D).
5. Test with debug output and gradient checks after each incremental change.

This analysis provides a clear, evidence-based foundation for the required refactoring and implementation work.
=======
Okay, let's systematically break down the training and loss function situation in your `RNA_PREDICT` pipeline. Your analysis that the current setup might be incorrect is spot-on, particularly given the staged nature and the different model types involved.

**Refined Hypothesis:**

The current training implementation in `RNALightningModule` is fundamentally misaligned with the intended pipeline design described in the documentation (internal design docs, TorsionBERT paper, AF3 paper). Specifically:

1.  The loss is calculated based on the 3D coordinate output of **Stage C (MP-NeRF)**, not the final output of **Stage D (Diffusion)**.
2.  This results in **Stage D (Diffusion model) and Stage B (Pairformer)** receiving **no training signal (zero gradients)**.
3.  **Stage B (TorsionBERT)** is being trained, but **indirectly and suboptimally**, via gradients flowing back through the Stage C geometric reconstruction, rather than through direct angle supervision or via gradients from the final Stage D output.
4.  The current MSE loss on coordinates is **inappropriate** for training the Stage D diffusion model, which requires a specific denoising objective.

This discrepancy means the core components responsible for global context (Pairformer) and final structure refinement (Diffusion) are not learning, and TorsionBERT is learning from a potentially noisy, indirect signal.

**Detailed Analysis:**

1.  **Analysis of `RNALightningModule` Training Code:**
    *   **`forward` method:** Executes Stages A, B (Torsion & Pairformer), and C sequentially. It produces `coords` from Stage C. **Crucially, it does *not* execute `self.stageD` (ProtenixDiffusionManager).**
    *   **`training_step` method:**
        *   Calls `self.forward(batch)`.
        *   Extracts `predicted_coords = output["coords"]` (output of Stage C).
        *   Calculates `loss = MSE(predicted_coords, real_target_coords)`.
        *   The complex masking logic attempts to align atoms for this Stage C vs Ground Truth comparison.
        *   It checks `predicted_coords.requires_grad`, confirming the *intent* to train upstream models.

2.  **Analysis of Gradient Flow:**
    *   Loss originates from Stage C's output MSE.
    *   Gradients flow backward: `Loss -> predicted_coords`.
    *   **Stage C (MP-NeRF):** Since it's differentiable geometric operations based on input angles (`rna_predict.pipeline.stageC.mp_nerf.rna.rna_fold`), gradients flow *through* it to its input (`torsion_angles`). As stated, Stage C has no trainable parameters, so it's just a conduit.
    *   **Stage B (TorsionBERT):** Receives gradients because `torsion_angles` (its output) is the input to the differentiable Stage C. **It IS being trained**, but the signal is "produce angles that, after MP-NeRF reconstruction, match the ground truth 3D coords". This is indirect.
    *   **Stage B (Pairformer):** Its outputs (`s_embeddings`, `z_embeddings`) are calculated but **do not contribute to the `predicted_coords` used in the loss**. They only feed the `latent_merger`, whose output is also ignored by the loss. **Pairformer is NOT being trained.**
    *   **Stage D (Diffusion):** The `self.stageD` module is **never called** in the training forward path. **Stage D is NOT being trained.**
    *   **Stage A (RFold):** No trainable parameters, not involved in loss. Not trained.

3.  **Cross-Referencing with Documentation:**
    *   **TorsionBERT Paper (`torsionBert_full_paper.md`):**
        *   Focus: Predicting angles accurately. Metrics: MCQ, MAE on angles.
        *   Training: Direct supervision comparing predicted angles (or sin/cos) to ground truth angles.
        *   *Discrepancy:* Current loss trains indirectly via 3D coords, lacking direct angle supervision.
    *   **AlphaFold 3 Paper (`AF3_paper.md` Supplement):**
        *   *Pairformer (Trunk):* Produces `s_i`, `z_ij` to condition the Diffusion module. Trained *implicitly* via gradients from the final diffusion loss (AF3 Eq 15).
        *   *Discrepancy:* Current setup provides **no gradient** to Pairformer.
        *   *Diffusion Module:* Trained using a denoising objective (AF3 Eq 6 - weighted MSE, LDDT loss, etc.) on noisy coordinates, conditioned on trunk embeddings.
        *   *Discrepancy:* Current loss (MSE on Stage C output) is fundamentally different. Stage D isn't run.
    *   **Internal Design Docs (`Integrated_RNA_3D...`, `full_pipeline_spec...`, `core_framework...`):**
        *   Flow: Stages A/B -> Merger -> Stage D -> Final Coords.
        *   Combined Loss: Explicitly mention `L_3D` (on Stage D output) + `L_angle` (for TorsionBERT) + optional `L_pair`.
        *   *Discrepancy:* Current loss is on Stage C output, lacks Stage D integration, and lacks `L_angle`.

4.  **Conclusion Confirmation:**
    *   The current training loss setup is significantly flawed. It trains only TorsionBERT, and does so indirectly. It completely ignores Pairformer and the entire Stage D Diffusion model, which are critical components according to the design documents and underlying model papers.

**Actionable Recommendations (Refined Implementation Plan):**

To correctly train the intended pipeline, the `RNALightningModule` needs substantial refactoring, primarily in the `training_step`.

**Phase 1: Prerequisites & Direct Angle Supervision**

1.  **Enable Ground Truth Angle Loading:**
    *   **File:** `rna_predict/conf/data/default.yaml`
        *   **Action:** Set `load_ang: true`.
        *   **Action:** Ensure paths/patterns for loading angle data are correctly configured if not derivable from `index_csv`.
    *   **File:** `rna_predict/dataset/loader.py` (`RNADataset`)
        *   **Action:** Implement `_load_angles` to read angle data (e.g., from `.pt`, `.npy`). Handle missing data.
        *   **Action:** In `__getitem__`, call `_load_angles` and add the tensor as `"angles_true"` to the sample dictionary. Ensure correct padding/dtype.
    *   **File:** `rna_predict/dataset/collate.py` (`rna_collate_fn`)
        *   **Action:** Ensure the collate function correctly handles and batches the `"angles_true"` tensor.
    *   **Verification:** Load a batch and confirm `"angles_true"` exists with shape `[B, N, 7]` (or similar) and correct dtype/device.

2.  **Implement Direct Angle Loss (`L_angle`) in `training_step`:**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`)
    *   **Action:** Inside `training_step`, after `output = self.forward(batch)`:
        *   `predicted_angles_sincos = output["torsion_angles"]` # Shape [B, N, 14]
        *   `true_angles_rad = batch["angles_true"]` # Shape [B, N, 7]
        *   Implement/import `angles_rad_to_sin_cos(true_angles_rad)` -> `true_angles_sincos` # Shape [B, N, 14]
        *   `loss_angle = F.mse_loss(predicted_angles_sincos, true_angles_sincos)` (apply masking if needed based on sequence length).
        *   Store `loss_angle`.
    *   **Verification:** Temporarily return `{"loss": loss_angle}`. Train 1 step. Check `self.stageB_torsion` parameters have non-None `.grad`.

**Phase 2: Integrating Stage D into Training**

3.  **Prepare Inputs for Stage D within `training_step`:**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`)
    *   **Action:** Gather inputs needed by `self.stageD.forward` (or a dedicated train method).
        *   `coords_true = batch["coords_true"]` # Shape [B, N_atom, 3]
        *   Sample noise level `sigma_t` (shape `[B]`) based on a schedule (e.g., AF3's exponential or simpler linear).
        *   Generate noise `epsilon` (shape like `coords_true`).
        *   `coords_noisy = coords_true + epsilon * sigma_t.view(-1, 1, 1)`.
        *   Retrieve Stage B outputs: `s_embeddings` (`[B, N_res, C_s]`), `z_embeddings` (`[B, N_res, N_res, C_z]`).
        *   **(Bridging Required):** Bridge residue-level `s_embeddings` (and potentially `z_embeddings`) to atom-level using `rna_predict.utils.tensor_utils.embedding.residue_to_atoms` and `batch["atom_to_token_idx"]`. Let the result be `s_embeddings_atom`, `z_embeddings_atom`. This is critical if Stage D expects atom-level conditioning.
        *   **(Optional Merger):** If using `self.latent_merger`, feed it the necessary (potentially bridged) embeddings, angles, adjacency to get `unified_latent`. Ensure its output level (residue/atom) matches Stage D's expectation.
        *   Define `conditioning_signal` (e.g., `s_embeddings_atom`, `z_embeddings_atom`, or `unified_latent`).

4.  **Execute Stage D Training Step:**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`)
    *   **Action:** Call `self.stageD`'s forward/training method.
        *   `stage_d_pred = self.stageD(coords_noisy=coords_noisy, conditioning=conditioning_signal, noise_level=sigma_t, **other_stageD_args)`
    *   **Verification:** Check input/output shapes and types. Ensure no runtime errors.

5.  **Implement Diffusion Loss (`L_diffusion`)**:
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`)
    *   **Action:** Calculate loss based on `stage_d_pred`.
        *   **If `stage_d_pred` is predicted noise `\hat{\epsilon}`:**
            *   `loss_diffusion = F.mse_loss(stage_d_pred, epsilon)` (apply masking).
        *   **If `stage_d_pred` is predicted denoised coords `\hat{x}_0` (like AF3):**
            *   Requires implementing `weighted_rigid_align` (Algo 28) and `SmoothLDDTLoss` (Algo 27).
            *   `aligned_coords_true = weighted_rigid_align(coords_true, stage_d_pred, weights)`
            *   `loss_mse = weighted_mse(stage_d_pred, aligned_coords_true, weights)`
            *   `loss_lddt = SmoothLDDTLoss(stage_d_pred, aligned_coords_true)`
            *   `loss_diffusion = weight_factor * (loss_mse + w_lddt * loss_lddt)` (weight_factor depends on `sigma_t` per AF3 Eq 6).
            *   *Recommendation:* Start with the simpler noise prediction objective if possible, unless Stage D is explicitly designed for coordinate prediction.
    *   **Verification:** Check `loss_diffusion` is scalar. Temporarily return `{"loss": loss_diffusion}`. Train 1 step. Verify gradients flow to `self.stageD` parameters *and* back to the conditioning parameters (Pairformer, TorsionBERT, Merger).

**Phase 3: Final Integration**

6.  **Combine and Log Losses:**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`)
    *   **Action:** Define loss weights in config (e.g., `cfg.training.w_diffusion`, `cfg.training.w_angle`).
    *   **Action:** Calculate `total_loss = cfg.training.w_diffusion * loss_diffusion + cfg.training.w_angle * loss_angle`.
    *   **Action:** Use `self.log(...)` to log `total_loss`, `loss_diffusion`, `loss_angle`.
    *   **Action:** Return `{"loss": total_loss}`.

7.  **Configuration & Dimension Review:**
    *   **Files:** `rna_predict/conf/**`
    *   **Action:** Systematically ensure all embedding dimensions (`c_s`, `c_z`, `c_atom`, `num_angles`, merger dims, Stage D conditioning dims) are consistent across stage configurations and the `config_schema.py`. Pay close attention to residue-vs-atom level dimensions during bridging.

8.  **Testing Strategy:**
    *   **Unit Tests:** For angle conversion, loss calculations (diffusion, angle), merger.
    *   **Integration Test (`test_lightning_trainer.py`):** Train for a few steps on 1-2 samples. Assert non-None gradients for parameters in TorsionBERT (LoRA), Pairformer (LoRA), and Stage D.
    *   **Monitor Training:** Use TensorBoard/W&B to watch loss components decrease on a small dataset. Use loss weights to isolate components if debugging is needed.

This detailed plan addresses the identified discrepancies and provides a concrete path to implement a correct and effective training loop for your multi-stage RNA prediction pipeline. The most complex parts will be the residue-to-atom bridging for Stage D conditioning and implementing the chosen diffusion loss objective accurately.
=======
Okay, let's perform a detailed, critical deep dive into **Phase 1: Enable Ground Truth Angle Loading** of the proposed plan. This is a foundational step, and getting the data loading right is crucial for implementing the direct angle loss later.

**Goal of Phase 1:** Modify the data pipeline (`RNADataset` and `rna_collate_fn`) to reliably load, process (pad/truncate), and batch ground truth torsion angles, making them available as `batch["angles_true"]` with the correct shape, dtype, and device in the `RNALightningModule.training_step`.

---

**1. File: `rna_predict/conf/data/default.yaml`**

*   **Action 1.1: Enable Angle Loading Flag**
    *   **Code Change:**
        ```yaml
        # rna_predict/conf/data/default.yaml
        # ... (other keys) ...
        load_adj: false
        load_ang: true   # <<< CHANGE THIS LINE from false to true
        # ... (other keys like coord_fill_value, coord_dtype) ...
        # Optional: Add flag for detailed batch inspection during debugging
        debug_inspect_batch: false # Set to true via CLI or test config to log first batch details
        ```
    *   **Rationale:** This boolean flag acts as a switch for the `RNADataset` to activate the angle loading logic. It allows flexibility to train with or without angle supervision without changing code.
    *   **Critical Consideration:** This flag *only* enables the *attempt* to load angles. It doesn't guarantee the data exists or is correctly formatted.

*   **Action 1.2: Define Angle Data Location and Format Strategy**
    *   **Strategy Decision:** Assume angle data for a structure (e.g., `data_root/1abc.cif`) is stored in a corresponding file (e.g., `data_root/1abc.pt`). This requires a consistent naming convention.
        *   **File Extension:** Assume `.pt` (PyTorch tensor file).
        *   **Data Format:** Assume the `.pt` file directly contains a `torch.Tensor`.
        *   **Tensor Shape:** Assume shape `[L, 7]`, where `L` is the number of residues in the sequence, and 7 corresponds to the standard torsions (α, β, γ, δ, ε, ζ, χ).
        *   **Units:** Assume angles are stored in **radians**. This is the standard for trigonometric functions in PyTorch/NumPy.
        *   **Dtype:** Assume `torch.float32`.
    *   **Configuration Impact:** *No new configuration keys are needed immediately* if this convention is followed. The existing `root_dir` and the file path derived from `index_csv` (e.g., `row['filepath']` or similar) will be used to construct the angle file path.
    *   **Alternative Strategies (If needed later):**
        *   Add `data.angle_dir` key if angles are in a separate directory.
        *   Add `data.angle_suffix` key if the extension isn't `.pt`.
        *   Add a column like `angle_filepath` to `index_csv`.
        *   Store angles within a larger `.pt` or `.hdf5` file containing multiple data types per structure.
    *   **Documentation:** This chosen strategy (same base name, `.pt` extension, `[L, 7]` shape, radians) *must* be clearly documented for anyone preparing the training data.

---

**2. File: `rna_predict/dataset/loader.py` (`RNADataset`)**

*   **Action 2.1: Imports**
    *   **Code Change:**
        ```python
        from pathlib import Path # Ensure Path is imported
        from typing import Optional # Add Optional for type hinting
        # ... other imports ...
        ```

*   **Action 2.2: Implement `_load_angles` Method**
    *   **Code Implementation:**
        ```python
        # Inside RNADataset class

        def _get_structure_filepath(self, row) -> Path:
            """Helper to consistently get the structure filepath."""
            # !! ADAPT THIS based on your actual index_csv structure !!
            # Example 1: If CSV has a 'filepath' column
            if 'filepath' in row and pd.notna(row['filepath']):
                file_path_str = row['filepath']
            # Example 2: If CSV has 'pdb_id' and 'chain' (e.g., "1abc_A")
            elif 'target_id' in row and pd.notna(row['target_id']):
                 # Assuming target_id might be like "PDBID_CHAIN" or just "PDBID"
                 pdb_id = row['target_id'].split('_')[0]
                 # Assuming extension is .cif, could be .pdb
                 file_path_str = f"{pdb_id}.cif"
            else:
                raise ValueError(f"Cannot determine structure file path from row: {row.to_dict()}")

            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                file_path = Path(self.root_dir) / file_path # Assumes root_dir is absolute or relative to CWD

            if not file_path.exists():
                 raise FileNotFoundError(f"Structure file not found at resolved path: {file_path}")
            return file_path

        def _load_angles(self, row) -> Optional[torch.Tensor]:
            """
            Loads ground truth torsion angles from a .pt file based on the
            structure file path derived from the row.
            Assumes angle file has the same base name as structure file but with .pt extension.
            Returns None if file not found or data is invalid.
            """
            angle_path = None # Initialize for error logging
            try:
                structure_path = self._get_structure_filepath(row)
                angle_path = structure_path.with_suffix(".pt")

                if not angle_path.is_file():
                    if self.verbose:
                        logger.warning(f"Angle file not found for {structure_path.name} at expected location: {angle_path}")
                    return None

                # Load the tensor data
                angles_data = torch.load(angle_path, map_location='cpu') # Load to CPU first

                # --- Data Validation ---
                if not isinstance(angles_data, torch.Tensor):
                    logger.error(f"Invalid data type in angle file {angle_path}. Expected torch.Tensor, got {type(angles_data)}.")
                    return None

                if angles_data.dim() != 2:
                    logger.error(f"Invalid dimensions in angle file {angle_path}. Expected 2D tensor [L, num_angles], got {angles_data.dim()}D.")
                    return None

                num_angles_loaded = angles_data.shape[1]
                # Check if number of angles is reasonable (e.g., 7 standard, maybe more if pseudo included)
                if num_angles_loaded < 7: # Allow for more than 7 if pseudo-torsions are included
                     logger.warning(f"Unexpected number of angles ({num_angles_loaded}) in {angle_path}. Expected at least 7. Using loaded data.")
                     # Decide if this should be an error or just a warning

                # Length check will happen in __getitem__ after sequence is loaded
                # Ensure dtype is float32
                return angles_data.to(torch.float32)

            except FileNotFoundError as e:
                 # This is handled by _get_structure_filepath now, but keep as fallback
                 logger.error(f"Structure file error for row {row.get('id', 'N/A')}: {e}")
                 return None
            except Exception as e:
                logger.error(f"Error loading or validating angles for {row.get('id', 'N/A')} from {angle_path if angle_path else 'unknown path'}: {e}", exc_info=self.verbose)
                return None
        ```
    *   **Critical Considerations:**
        *   Robust Path Finding: The `_get_structure_filepath` needs to correctly interpret your `index_csv` to find the base path. The example provided covers common scenarios but might need adjustment. Ensure it correctly resolves relative paths using `self.root_dir`.
        *   Validation: The checks for tensor type and dimensions are crucial. Decide how strictly to enforce the number of columns (e.g., exactly 7, or at least 7).
        *   Error Logging: Provide informative logs, especially when files are missing or data is invalid. Include `exc_info=self.verbose` for detailed tracebacks during debugging.

*   **Action 2.3: Modify `__getitem__`**
    *   **Code Implementation:**
        ```python
        # Inside RNADataset.__getitem__:
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            item_id = row.get('id', idx) # For logging

            # --- Load Sequence (Assume _load_sequence exists) ---
            sequence = self._load_sequence(row)
            if sequence is None:
                 logger.error(f"Failed to load sequence for item {item_id}. Skipping sample.")
                 # Returning an empty dict or raising an error might be necessary depending on DataLoader robustness
                 return {} # Or raise appropriate error
            L_res = len(sequence)

            # --- Load Coords & Mask (Assume _load_coords exists) ---
            coords_true, atom_mask, L_atoms = self._load_coords(row) # Assume returns tuple
            if coords_true is None or atom_mask is None:
                 logger.error(f"Failed to load coordinates/mask for item {item_id}. Skipping sample.")
                 return {} # Or raise

            # --- Load Angles (Conditional) ---
            angles_true_loaded = None
            if self.load_ang:
                angles_true_loaded = self._load_angles(row)

            # --- Process Angles (Validation, Placeholder Creation, Padding) ---
            angles_true_processed = None
            expected_angle_dim = 7 # Define expected dimension

            if angles_true_loaded is not None:
                # Validate length against sequence length
                if angles_true_loaded.shape[0] != L_res:
                    logger.error(f"Angle length mismatch for {item_id}: Angles({angles_true_loaded.shape[0]}) != Sequence({L_res}). Using zero placeholder.")
                    angles_true_processed = torch.zeros((L_res, expected_angle_dim), dtype=torch.float32)
                else:
                    # Use loaded angles, ensure correct dim if needed
                    if angles_true_loaded.shape[1] != expected_angle_dim:
                         logger.warning(f"Angle dim mismatch for {item_id}: Expected {expected_angle_dim}, got {angles_true_loaded.shape[1]}. Slicing/Padding last dim.")
                         # Slice or pad the feature dimension (dim 1)
                         if angles_true_loaded.shape[1] > expected_angle_dim:
                              angles_true_processed = angles_true_loaded[:, :expected_angle_dim]
                         else:
                              pad_needed = expected_angle_dim - angles_true_loaded.shape[1]
                              padding = torch.zeros((L_res, pad_needed), dtype=torch.float32)
                              angles_true_processed = torch.cat([angles_true_loaded, padding], dim=1)
                    else:
                         angles_true_processed = angles_true_loaded.to(torch.float32) # Ensure dtype
            elif self.load_ang:
                 # Loading was enabled but failed or file not found
                 logger.warning(f"Could not load angles for {item_id}. Using zero placeholder.")
                 angles_true_processed = torch.zeros((L_res, expected_angle_dim), dtype=torch.float32)
            # else: load_ang is False, angles_true_processed remains None

            # Apply padding/truncation to max_res
            if angles_true_processed is not None:
                if L_res > self.max_res:
                    angles_true_final = angles_true_processed[:self.max_res, :]
                elif L_res < self.max_res:
                    pad_len = self.max_res - L_res
                    padding = torch.zeros((pad_len, angles_true_processed.shape[1]), dtype=torch.float32)
                    angles_true_final = torch.cat([angles_true_processed, padding], dim=0)
                else:
                    angles_true_final = angles_true_processed
            else:
                 angles_true_final = None # Keep as None if load_ang was false

            # --- Construct Final Sample Dictionary ---
            sample = {
                'id': item_id,
                'sequence': sequence, # Already loaded & processed
                'coords_true': coords_true, # Already loaded & processed
                'atom_mask': atom_mask, # Already loaded & processed
                # ... include ALL other keys needed by the model/collate_fn ...
                # 'atom_to_token_idx': ...,
                # 'ref_element': ...,
                # 'ref_atom_name_chars': ...,
                # 'residue_indices': ... # Add this if needed
            }

            if angles_true_final is not None:
                 sample['angles_true'] = angles_true_final

            # Example: Add residue_indices if not already loaded
            if 'residue_indices' not in sample and 'atom_to_token_idx' in sample:
                 # Assuming atom_to_token_idx maps atom index to residue index
                 # This is a placeholder - adapt based on your actual data structure
                 sample['residue_indices'] = sample['atom_to_token_idx']

            return sample

        ```
    *   **Critical Considerations:**
        *   Order of Operations: Ensure `L_res` (sequence length) is known *before* creating placeholder tensors or performing length validation for angles.
        *   Placeholder Strategy: Using `torch.zeros` as a placeholder for missing/failed loads is common, but relies on the loss function correctly handling masking (e.g., using an attention mask derived from sequence length). Alternatively, samples with missing crucial data could be skipped entirely.
        *   Dimension Handling: Explicitly check and handle potential mismatches in the number of angles (dimension 1) if your data source might vary.
        *   Completeness: Ensure *all* keys required by the `rna_collate_fn` and the model are present in the final `sample` dictionary.

---

**3. File: `rna_predict/dataset/collate.py` (`rna_collate_fn`)**

*   **Action 3.1: Add Handling for `angles_true` Key**
    *   **Code Implementation:**
        ```python
        import torch
        from torch.nn.utils.rnn import pad_sequence
        import logging

        logger = logging.getLogger(__name__)

        def rna_collate_fn(batch, debug_logging=False):
            collated_batch = {}
            # Determine keys present in the first sample to handle optional keys
            if not batch:
                return {}
            sample_keys = batch[0].keys()

            # Keys expected to be stacked (typically already padded in dataset)
            keys_to_stack = ['coords_true', 'atom_mask', 'angles_true'] # Add angles_true here
            # Keys needing padding (if variable length in dataset, less common now)
            keys_to_pad = []
            # Keys to collect into lists
            keys_to_list = ['id', 'sequence'] # Add other non-tensor keys

            for key in sample_keys:
                if key in keys_to_stack:
                    try:
                        # Check if all items have this key and it's a tensor
                        if all(key in item and isinstance(item[key], torch.Tensor) for item in batch):
                             tensors = [item[key] for item in batch]
                             # Check if shapes match before stacking (common error source)
                             if len(set(t.shape for t in tensors)) > 1:
                                 logger.error(f"Collate Error: Inconsistent shapes for key '{key}'. Shapes: {[t.shape for t in tensors]}")
                                 collated_batch[key] = None # Or raise error
                                 continue
                             collated_batch[key] = torch.stack(tensors, dim=0)
                        else:
                             logger.warning(f"Collate Warning: Key '{key}' missing or not a tensor in some batch items. Skipping.")
                             collated_batch[key] = None
                    except Exception as e:
                        logger.error(f"Error stacking key '{key}': {e}", exc_info=debug_logging)
                        collated_batch[key] = None
                elif key in keys_to_pad:
                     # Add padding logic here if needed
                     pass
                elif key in keys_to_list:
                     collated_batch[key] = [item.get(key) for item in batch]
                # Handle other keys if necessary, or ignore them
                # else:
                #    if debug_logging: logger.debug(f"Collate: Ignoring key '{key}'")

            # --- Add Debug Logging for final batch shapes ---
            if debug_logging:
                 logger.debug("--- Final Collated Batch Shapes/Types ---")
                 for k, v in collated_batch.items():
                     if isinstance(v, torch.Tensor):
                         logger.debug(f"Key: '{k}', Shape: {v.shape}, Dtype: {v.dtype}, Device: {v.device}")
                     elif isinstance(v, list):
                          logger.debug(f"Key: '{k}', Type: list, Length: {len(v)}")
                     else:
                         logger.debug(f"Key: '{k}', Type: {type(v)}")
                 logger.debug("------------------------------------------")

            return collated_batch
        ```
    *   **Critical Considerations:**
        *   Shape Consistency: The most common failure in collation is attempting to stack tensors of different shapes. The added check `len(set(t.shape for t in tensors)) > 1` helps catch this. This relies on the `RNADataset` correctly padding/truncating all `angles_true` tensors to `[max_res, 7]`.
        *   Handling Missing Keys: The code now checks if the key exists and is a tensor *before* attempting to stack, logging a warning if it's inconsistent across the batch.
        *   Completeness: Ensure the collation handles *all* necessary keys from the dataset samples.

---

**4. Verification Plan:**

*   **Setup:**
    *   Create a `test_data` directory.
    *   Inside, create `test_index.csv` with at least two rows pointing to structure files:
        ```csv
        id,filepath
        sample1,test_data/sample1.pdb
        sample2,test_data/sample2.cif
        # Add a row that might lack an angle file if testing error handling
        # sample3,test_data/sample3.pdb
        ```
    *   Create dummy `sample1.pdb` (e.g., 10 residues) and `sample2.cif` (e.g., 20 residues). Exact content doesn't matter much, just needs to be parsable by your `_load_coords` / `_load_sequence`.
    *   Create `sample1.pt` containing `torch.randn(10, 7, dtype=torch.float32)`.
    *   Create `sample2.pt` containing `torch.randn(20, 7, dtype=torch.float32)`.
    *   *Do not* create `sample3.pt` if testing missing file handling.
    *   Create `rna_predict/conf/test_load_angles.yaml`:
        ```yaml
        defaults:
          - data: default
          # Add minimal model configs ONLY if RNADataset depends on them
          # - model: ...
        data:
          index_csv: test_data/test_index.csv # Relative to where test script runs
          root_dir: ./ # Assumes test script runs from project root
          load_ang: true
          max_res: 50 # Choose a value > largest L (e.g., 20)
          batch_size: 2
          num_workers: 0 # CRITICAL for debugging
          verbose: true # Enable dataset verbose logging
          debug_inspect_batch: true # Enable detailed batch logging in train.py (if used)
        # Add minimal training config if needed by LightningModule
        # training:
        #   w_diffusion: 1.0
        #   w_angle: 0.1
        ```

*   **Execution Script (`tests/integration/test_angle_loading.py`):**
    ```python
    import hydra
    from omegaconf import DictConfig, OmegaConf
    from rna_predict.dataset.loader import RNADataset
    from rna_predict.dataset.collate import rna_collate_fn
    from torch.utils.data import DataLoader
    import torch
    import os
    import pytest

    # Assuming this test runs from the project root directory
    CONFIG_PATH = "../rna_predict/conf"
    CONFIG_NAME = "test_load_angles.yaml"

    @pytest.fixture(scope="module")
    def hydra_config():
        """Loads the Hydra configuration."""
        with hydra.initialize(config_path=CONFIG_PATH, version_base=None):
            cfg = hydra.compose(config_name=CONFIG_NAME)
        # Resolve paths relative to original CWD if needed by dataset
        # Example: cfg.data.index_csv = os.path.join(hydra.utils.get_original_cwd(), cfg.data.index_csv)
        # Example: cfg.data.root_dir = os.path.join(hydra.utils.get_original_cwd(), cfg.data.root_dir)
        # NOTE: RNADataset might resolve root_dir itself, check its implementation
        print("Loaded Hydra Config:\n", OmegaConf.to_yaml(cfg))
        return cfg

    def test_dataset_loading(hydra_config):
        """Tests dataset instantiation and single item retrieval."""
        print("\n--- Testing RNADataset ---")
        # Pass cfg directly to RNADataset
        dataset = RNADataset(cfg=hydra_config, load_adj=False, load_ang=True, verbose=True)
        print(f"Dataset size: {len(dataset)}")
        assert len(dataset) > 0, "Dataset is empty!"

        print("\nFetching first sample (sample1)...")
        sample1 = dataset[0] # Corresponds to sample1.pdb
        print("Sample 1 keys:", sample1.keys())
        assert 'angles_true' in sample1, "'angles_true' key missing from sample1"
        assert isinstance(sample1['angles_true'], torch.Tensor), "'angles_true' is not a Tensor"
        assert sample1['angles_true'].shape == (hydra_config.data.max_res, 7), f"Sample 1 shape mismatch: Expected {(hydra_config.data.max_res, 7)}, got {sample1['angles_true'].shape}"
        assert sample1['angles_true'].dtype == torch.float32, f"Sample 1 dtype mismatch: Expected float32, got {sample1['angles_true'].dtype}"
        # Check that padded values are zero
        assert torch.all(sample1['angles_true'][10:] == 0), "Padding values are not zero for sample1"
        print("Sample 1 verification PASSED.")

        print("\nFetching second sample (sample2)...")
        sample2 = dataset[1] # Corresponds to sample2.cif
        assert 'angles_true' in sample2
        assert isinstance(sample2['angles_true'], torch.Tensor)
        assert sample2['angles_true'].shape == (hydra_config.data.max_res, 7)
        assert torch.all(sample2['angles_true'][20:] == 0), "Padding values are not zero for sample2"
        print("Sample 2 verification PASSED.")

        # Optional: Test missing angle file handling if sample3 was included
        # print("\nFetching third sample (sample3 - missing angle file)...")
        # sample3 = dataset[2]
        # assert 'angles_true' in sample3
        # assert isinstance(sample3['angles_true'], torch.Tensor)
        # assert sample3['angles_true'].shape == (hydra_config.data.max_res, 7)
        # assert torch.all(sample3['angles_true'] == 0), "Placeholder for missing angle file is not zeros"
        # print("Sample 3 (missing file) verification PASSED.")


    def test_dataloader_collation(hydra_config):
        """Tests DataLoader and collation function."""
        print("\n--- Testing DataLoader & Collation ---")
        dataset = RNADataset(cfg=hydra_config, load_adj=False, load_ang=True, verbose=False) # Less verbose here
        # Use debug_logging=True in collate_fn for verbose output
        dataloader = DataLoader(dataset, batch_size=hydra_config.data.batch_size, collate_fn=lambda b: rna_collate_fn(b, debug_logging=True), num_workers=0)

        print("\nFetching first batch...")
        batch = next(iter(dataloader))
        print("Batch keys:", batch.keys())

        assert 'angles_true' in batch, "'angles_true' key missing from batch"
        assert isinstance(batch['angles_true'], torch.Tensor), f"'angles_true' is not a Tensor in batch, got {type(batch['angles_true'])}"
        expected_shape = (hydra_config.data.batch_size, hydra_config.data.max_res, 7)
        assert batch['angles_true'].shape == expected_shape, f"Expected batch shape {expected_shape}, got {batch['angles_true'].shape}"
        assert batch['angles_true'].dtype == torch.float32, f"Expected batch dtype torch.float32, got {batch['angles_true'].dtype}"
        print("Batch collation verification PASSED.")

    # To run this test: pytest tests/integration/test_angle_loading.py -s
    ```
*   **Run:** Execute the test script using `pytest -s tests/integration/test_angle_loading.py` (the `-s` shows print statements). Examine the output logs from the dataset and collate function, and check if the assertions pass.

---

This detailed breakdown for Phase 1 provides specific code changes, assumptions, critical considerations, and a verification plan to ensure ground truth angles are correctly loaded and batched, setting the stage for implementing the direct angle loss in the next phase.
=====
Okay, you are absolutely correct. We cannot just *assume* the ground truth torsion angle data exists in the required format (`.pt` files with `[L, 7]` tensors in radians). That was a placeholder assumption in the previous plan. We need a concrete strategy to **generate** this ground truth data from the existing 3D coordinate files (PDB/CIF).

This requires analyzing the candidate tools (X3DNA-DSSR, MDAnalysis) and selecting the best one for **pre-computing highly accurate torsion angles** for the entire dataset.

**Deep Dive: Selecting the Best Tool for Pre-computing Ground Truth Torsion Angles**

**Goal:** Choose a tool that accurately calculates standard RNA torsion angles (α, β, γ, δ, ε, ζ, χ), handles real-world PDB/CIF complexities (modified residues, missing atoms), is scriptable for batch processing, and provides output in a usable format (e.g., numerical values).

**Analysis of Candidate Tools:**

1.  **X3DNA-DSSR:**
    *   **Pros:**
        *   **Specialized for Nucleic Acids:** Designed specifically for RNA/DNA, deeply understands conventions (torsion definitions, base pairing, Leontis-Westhof, Saenger).
        *   **High Accuracy:** Widely regarded as a gold standard for RNA structural analysis. The documentation explicitly mentions correcting errors from other tools.
        *   **Robust Handling of Modifications:** Automatically maps dozens of common modified residues to standard parents, ensuring calculations don't fail on non-standard bases.
        *   **Handles Edge Cases:** Specific flags for NMR ensembles (`--nmr`) and symmetric assemblies (`--symm`). Robust to missing atoms (outputs `---`).
        *   **Scriptable:** Excellent command-line interface.
        *   **Convenient Output:** `--json` flag provides structured, machine-readable output containing torsion angles (typically in degrees) which is easy to parse in Python. The `--torsion-file` provides a dedicated text file.
        *   **Fast:** As a compiled binary, it's very efficient for processing individual files.
    *   **Cons:**
        *   **Licensing:** Requires obtaining a license (free for academics, paid for commercial). This is a one-time setup step.
        *   **Installation:** Not a simple `pip install`. Requires downloading a binary and potentially setting PATH.
        *   **Output Units:** Outputs angles in **degrees** by default, requiring conversion to radians for use with PyTorch trigonometric functions (`torch.sin`, `torch.cos`).

2.  **MDAnalysis:**
    *   **Pros:**
        *   **Python Native:** Integrates seamlessly into Python data pipelines (`pip install mdanalysis`).
        *   **Flexible:** The general `analysis.dihedrals.Dihedral` class can calculate *any* dihedral if the 4 atoms are correctly selected.
        *   **Open Source:** LGPL license, easy installation.
        *   **Trajectory Handling:** Excellent for analyzing MD simulations (though not our primary need here).
        *   **Output Units:** Returns angles in **radians** directly, which is convenient for PyTorch.
    *   **Cons:**
        *   **Not RNA-Specialized:** Lacks built-in knowledge of RNA-specific conventions (α-ζ, χ definitions, sugar puckers ν0-ν4, modified base handling). The user must manually define the correct 4-atom selections for *every* angle type, including tricky inter-residue ones (α, ε, ζ). This is error-prone.
        *   **Robustness Dependent on User Code:** Handling missing atoms, altLocs, or modified residues requires explicit, careful Python logic within the selection process. Less robust out-of-the-box compared to DSSR's built-in handling.
        *   **Potential Complexity:** Scripting the selection logic for all 7 standard torsions across all residues, handling termini and chain breaks correctly, is significantly more complex than using DSSR's automated calculation.
        *   **Potential BAT Class Issue:** The reported issue with the `BAT` class modifying coordinates unexpectedly (though potentially fixed or specific) raises a minor flag about tool subtleties.

**Conclusion: Recommendation**

For the specific task of **pre-computing accurate ground truth torsion angles for a dataset of RNA structures**, **X3DNA-DSSR is the strongly recommended tool.**

*   **Reasoning:** Its specialization in nucleic acids ensures higher accuracy and robustness concerning RNA-specific conventions, modified residues, and common structural artifacts found in PDB/CIF files. While MDAnalysis *can* calculate these angles, the burden of correctly implementing and validating the selections for all RNA angles (especially inter-residue ones) falls entirely on the user, increasing the risk of errors in the "ground truth" data. DSSR automates this complex process reliably. The convenience of JSON output for parsing outweighs the minor inconvenience of the licensing step and degree-to-radian conversion. Speed is also likely superior for batch processing static files.

**Revised Plan for Phase 1: Pre-computation using DSSR**

This replaces the previous assumption of existing `.pt` files with a concrete generation step.

**Phase 1a: Generate Ground Truth Angle Data (One-Time Pre-computation)**

1.  **Obtain DSSR:** Secure the appropriate DSSR binary and license for your operating system. Install it and ensure it's executable (e.g., in your PATH or provide the full path).
2.  **Create a Pre-computation Script:** Write a Python script (`scripts/preprocessing/compute_ground_truth_angles.py` or similar).
    *   **Input:** Path to the main `index_csv` used by `RNADataset`, path to the data `root_dir`.
    *   **Logic:**
        *   Read the `index_csv` into a pandas DataFrame.
        *   Iterate through each row of the DataFrame.
        *   For each row:
            *   Determine the full path to the structure file (PDB or CIF) using the same logic as `RNADataset._get_structure_filepath`.
            *   Define the corresponding output `.pt` file path (e.g., `structure_path.with_suffix(".pt")`).
            *   **Check if output `.pt` file already exists. If yes, skip (allows incremental runs).**
            *   Construct the DSSR command: `dssr_cmd = ["x3dna-dssr", "--json", f"--input={structure_path}"]`.
            *   Run DSSR using `subprocess.run(dssr_cmd, capture_output=True, text=True, check=True)`. Use `check=True` to raise an error if DSSR fails.
            *   Use a `try...except subprocess.CalledProcessError` block to catch DSSR failures and log them (e.g., file parsing errors, DSSR internal errors).
            *   Inside the `try` block (after successful DSSR run):
                *   Parse the JSON output: `dssr_data = json.loads(result.stdout)`.
                *   Check if `"nts"` (nucleotides) key exists and is not empty. If empty, log a warning (e.g., protein-only file) and continue to the next structure.
                *   Initialize an empty list `all_angles = []`.
                *   Iterate through the nucleotides in `dssr_data["nts"]`:
                    *   Extract the 7 standard torsion angles: `alpha`, `beta`, `gamma`, `delta`, `epsilon`, `zeta`, `chi`.
                    *   **Handle Missing Values:** DSSR JSON might use `null` for undefined angles (e.g., alpha/zeta at termini). Replace `null` with a chosen placeholder, typically `0.0` or `np.nan`. Using `0.0` is often simpler if subsequent masking in the loss function is handled correctly. Using `np.nan` requires handling NaNs during loss calculation. **Let's choose `0.0` for simplicity initially.**
                    *   **Convert Degrees to Radians:** `angle_rad = angle_deg * np.pi / 180.0` for each extracted angle.
                    *   Append the list/tuple of 7 radian values for this nucleotide to `all_angles`.
                *   Convert the list of lists to a NumPy array: `angle_array_np = np.array(all_angles, dtype=np.float32)`. Shape should be `[L, 7]`.
                *   Convert to a PyTorch tensor: `angle_tensor = torch.from_numpy(angle_array_np)`.
                *   **Save the Tensor:** `torch.save(angle_tensor, output_pt_path)`.
                *   Log success for this file.
        *   Include overall progress reporting (e.g., using `tqdm` for the main loop).
3.  **Run the Script:** Execute this script once over your entire training/validation dataset. This populates the data directory with the necessary `.pt` angle files.

**Phase 1b: Adapt Data Loading (As Before, but Simpler)**

Now that the `.pt` files are guaranteed to exist (or were skipped with logs during pre-computation), the implementation in `RNADataset` becomes simpler.

1.  **File: `rna_predict/conf/data/default.yaml`**
    *   **Action:** Set `load_ang: true`. (No change from previous plan here).

2.  **File: `rna_predict/dataset/loader.py` (`RNADataset`)**
    *   **Action:** Implement `_load_angles` to *only load* the pre-computed `.pt` file. The complex DSSR call is removed.
        ```python
        # Inside RNADataset class
        def _load_angles(self, row) -> Optional[torch.Tensor]:
            """Loads pre-computed ground truth torsion angles from a .pt file."""
            angle_path = None
            try:
                structure_path = self._get_structure_filepath(row) # Assumes this helper exists and works
                angle_path = structure_path.with_suffix(".pt")

                if not angle_path.is_file():
                    # This case should be less frequent now if pre-computation was run
                    if self.verbose:
                        logger.warning(f"Pre-computed angle file not found: {angle_path}. Returning None.")
                    return None

                # Load the tensor data
                angles_tensor = torch.load(angle_path, map_location='cpu')

                # Basic validation (optional, as pre-computation should ensure this)
                if not isinstance(angles_tensor, torch.Tensor) or angles_tensor.dim() != 2 or angles_tensor.shape[1] != 7:
                     logger.error(f"Invalid data in pre-computed angle file {angle_path}. Shape: {angles_tensor.shape}, Type: {type(angles_tensor)}")
                     return None

                return angles_tensor.to(torch.float32) # Ensure correct dtype

            except Exception as e:
                logger.error(f"Error loading pre-computed angles for {row.get('id', 'N/A')} from {angle_path if angle_path else 'unknown path'}: {e}", exc_info=self.verbose)
                return None
        ```
    *   **Action:** Modify `__getitem__` to call the *simplified* `_load_angles` and handle padding/placeholder creation as outlined in the previous plan's step 2.3. The logic remains the same, but it now relies on the pre-computed files.

3.  **File: `rna_predict/dataset/collate.py` (`rna_collate_fn`)**
    *   **Action:** Ensure collation handles `angles_true` correctly (no change from previous plan's step 3.1).

4.  **Verification Plan:**
    *   Run the DSSR pre-computation script on your test data subset. Verify `.pt` files are created.
    *   Run the `tests/integration/test_angle_loading.py` script (from the previous plan). It should now pass by loading the pre-computed `.pt` files.

---

**Summary of Refined Phase 1:**

1.  **Pre-computation:** Use a dedicated script leveraging X3DNA-DSSR (`--json` output) to iterate through all structure files, calculate standard torsion angles (α-ζ, χ), convert them to **radians**, and save them as `[L, 7]` `torch.float32` tensors in `.pt` files (e.g., `structure.pt`) alongside the original structure files. Handle errors and missing atoms gracefully during this step.
2.  **Data Loading:** Modify `RNADataset` to simply load these pre-computed `.pt` files when `cfg.data.load_ang` is true. Implement padding/truncation and placeholder logic for missing files within `__getitem__`.
3.  **Collation:** Ensure `rna_collate_fn` stacks the loaded `angles_true` tensors correctly into batches.

This approach ensures accuracy by using a specialized tool (DSSR), improves efficiency by pre-computing angles once, and simplifies the data loading logic within `RNADataset`.
=====

Okay, let's perform a meticulous deep dive into **Phase 2: Integrating Stage D into Training** within the `RNALightningModule`. This phase is the most complex part of the refactoring, as it connects multiple stages and introduces the core diffusion training logic.

**Goal:** Modify `RNALightningModule.training_step` to correctly prepare inputs for Stage D, execute its training step, compute the appropriate diffusion loss (`L_diffusion`), and ensure gradients propagate back to Stage D and the upstream conditioning modules (TorsionBERT, Pairformer, LatentMerger).

---

**Detailed Implementation Steps & Considerations:**

**3. File: `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`) - Prepare Inputs for Stage D**

*   **Action 3.1: Define Helpers for Noise Sampling and Application.**
    *   **Rationale:** Encapsulate the logic for noise schedule sampling and adding noise to coordinates for better readability and potential reuse.
    *   **Implementation:** Add `_sample_noise_level` and `_add_noise` methods to `RNALightningModule`.
        ```python
        # Inside RNALightningModule class
        def _sample_noise_level(self, batch_size: int) -> torch.Tensor:
            """Samples noise level sigma_t for each item in the batch based on config."""
            # Access noise schedule config safely
            noise_schedule_cfg = getattr(getattr(getattr(self.cfg, 'model', {}), 'stageD', {}), 'diffusion', {}).get('noise_schedule', {})
            p_mean = noise_schedule_cfg.get('p_mean', -1.2)
            p_std = noise_schedule_cfg.get('p_std', 1.5)
            # Access sigma_data safely from model_architecture
            model_arch_cfg = getattr(getattr(getattr(self.cfg, 'model', {}), 'stageD', {}), 'diffusion', {}).get('model_architecture', {})
            sigma_data = model_arch_cfg.get('sigma_data', 1.0) # Default to 1.0 for EDM-like schedule

            # Log the parameters being used
            logger.debug(f"Noise sampling params: p_mean={p_mean}, p_std={p_std}, sigma_data={sigma_data}")

            # AF3-like exponential schedule (log-normal distribution for sigma)
            # log_snr = torch.randn(batch_size, device=self.device_) * p_std + p_mean # SNR = 1/sigma^2 -> log(SNR) = -2*log(sigma)
            # log_sigma = -0.5 * log_snr
            # sigma_t = sigma_data * torch.exp(log_sigma)

            # Simpler Placeholder: Uniform sampling in log space
            min_log_sigma = torch.log(torch.tensor(noise_schedule_cfg.get('s_min', 0.002), device=self.device_)) # Use s_min from config
            max_log_sigma = torch.log(torch.tensor(noise_schedule_cfg.get('s_max', 80.0), device=self.device_)) # Use s_max from config
            log_sigma_t = torch.rand(batch_size, device=self.device_) * (max_log_sigma - min_log_sigma) + min_log_sigma
            sigma_t = torch.exp(log_sigma_t)

            logger.debug(f"Sampled sigma_t (noise levels, shape {sigma_t.shape}): {sigma_t}")
            return sigma_t.to(self.device_) # Ensure device

        def _add_noise(self, coords_true: torch.Tensor, sigma_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Adds Gaussian noise based on sampled noise levels sigma_t."""
            if coords_true.numel() == 0: # Handle empty coordinates
                 return coords_true.clone(), torch.zeros_like(coords_true)
            epsilon = torch.randn_like(coords_true)
            # Ensure sigma_t can broadcast to coords_true shape [B, N_atom, 3]
            # sigma_t is [B], need [B, 1, 1]
            sigma_t_reshaped = sigma_t.view(-1, *([1] * (coords_true.dim() - 1)))
            coords_noisy = coords_true + epsilon * sigma_t_reshaped
            logger.debug(f"Added noise: coords_true.shape={coords_true.shape}, sigma_t.shape={sigma_t.shape}, epsilon.shape={epsilon.shape}, coords_noisy.shape={coords_noisy.shape}")
            return coords_noisy, epsilon
        ```
    *   **Critical Check:** The specific noise sampling strategy (`_sample_noise_level`) directly impacts training stability and performance. The AF3 schedule is complex; starting with a simpler schedule (like linear or uniform log-space) might be easier initially. **Verify `sigma_data`, `s_min`, `s_max` are correctly read from the config.**

*   **Action 3.2: Gather and Prepare Inputs within `training_step`.**
    *   **Code Snippet (Inside `training_step`):**
        ```python
        # (Ensure self.forward(batch) has been called to get 'output' dict)
        # ... (L_angle calculation from Phase 1) ...

        logger.debug("--- Preparing Stage D Inputs ---")

        # 1. Ground Truth Coords & Mask
        # Ensure keys exist in the batch, handle potential errors
        if "coords_true" not in batch or "atom_mask" not in batch:
            logger.error("Batch missing 'coords_true' or 'atom_mask'. Cannot proceed with Stage D training.")
            # Return a minimal loss or raise error
            return {"loss": loss_angle if 'loss_angle' in locals() else torch.tensor(0.0, device=self.device_, requires_grad=True)}

        coords_true = batch["coords_true"].to(self.device_) # Shape [B, N_atom_padded, 3]
        atom_mask = batch["atom_mask"].to(self.device_)     # Shape [B, N_atom_padded] bool

        # 2. Sample Noise Level & Add Noise
        batch_size = coords_true.shape[0]
        sigma_t = self._sample_noise_level(batch_size) # Shape [B]
        coords_noisy, epsilon = self._add_noise(coords_true, sigma_t) # Shapes [B, N_atom_padded, 3]

        # 3. Retrieve Stage B Outputs (Residue Level) from `output` dict
        if "s_embeddings" not in output or "z_embeddings" not in output:
             logger.error("Missing 's_embeddings' or 'z_embeddings' from self.forward output dict.")
             return {"loss": loss_angle} # Fallback

        s_embeddings_res = output["s_embeddings"].to(self.device_) # Shape [B, N_res_padded, C_s]
        z_embeddings_res = output["z_embeddings"].to(self.device_) # Shape [B, N_res_padded, N_res_padded, C_z]

        # 4. Bridging from Residue to Atom Level (CRITICAL)
        if 'atom_to_token_idx' not in batch:
            raise KeyError("Batch missing 'atom_to_token_idx' needed for residue-to-atom bridging.")
        atom_to_token_idx = batch['atom_to_token_idx'].to(self.device_) # Shape [B, N_atom_padded]

        # --- Bridging s_embeddings ---
        from rna_predict.utils.tensor_utils import residue_to_atoms, derive_residue_atom_map
        s_embeddings_atom_list = []
        n_atoms_padded = coords_true.shape[1]
        for b_idx in range(batch_size):
            # Derive map for this specific item (using its actual atom mask)
            current_atom_mask = atom_mask[b_idx]
            # Provide sequence if available for better mapping
            sequence_item = batch["sequence"][b_idx] if "sequence" in batch else ""
            residue_atom_map_item = derive_residue_atom_map(sequence=sequence_item, atom_mask=current_atom_mask)

            if not residue_atom_map_item:
                 logger.warning(f"Empty residue_atom_map derived for batch item {b_idx}. Using zeros.")
                 s_embeddings_atom_list.append(torch.zeros((n_atoms_padded, s_embeddings_res.shape[-1]), device=self.device_, dtype=s_embeddings_res.dtype))
                 continue

            # Bridge using the derived map
            try:
                # residue_to_atoms expects [N_res, C] and returns [N_atom_actual, C]
                s_atom_actual = residue_to_atoms(s_embeddings_res[b_idx], residue_atom_map_item)
                # Pad back to N_atom_padded
                padded_s_atom = torch.zeros((n_atoms_padded, s_atom_actual.shape[-1]), device=self.device_, dtype=s_atom_actual.dtype)
                real_indices = torch.where(current_atom_mask)[0] # Indices of real atoms
                num_real_atoms = len(real_indices)
                # Ensure slicing matches actual number of atoms returned by bridging
                padded_s_atom[real_indices] = s_atom_actual[:num_real_atoms]
                s_embeddings_atom_list.append(padded_s_atom)
            except Exception as e:
                logger.error(f"Error bridging s_embeddings for batch item {b_idx}: {e}", exc_info=True)
                s_embeddings_atom_list.append(torch.zeros((n_atoms_padded, s_embeddings_res.shape[-1]), device=self.device_, dtype=s_embeddings_res.dtype))

        s_embeddings_atom = torch.stack(s_embeddings_atom_list, dim=0) # Shape [B, N_atom_padded, C_s]
        logger.debug(f"Bridged s_embeddings_atom shape: {s_embeddings_atom.shape}")

        # --- Bridging z_embeddings (Optional but potentially needed by Stage D) ---
        # If Stage D needs atom-level pair features, implement bridging here.
        # This typically involves replicating residue-pair features to corresponding atom-pairs.
        # Example (Simple Replication - adapt if Stage D uses a more complex scheme):
        # z_embeddings_atom = torch.zeros(batch_size, n_atoms_padded, n_atoms_padded, z_embeddings_res.shape[-1], device=self.device_, dtype=z_embeddings_res.dtype)
        # for b_idx in range(batch_size):
        #    residue_map_item = derive_residue_atom_map(...) # Derive map again or reuse
        #    for res_i, atoms_i in enumerate(residue_map_item):
        #        for res_j, atoms_j in enumerate(residue_map_item):
        #            if atoms_i and atoms_j: # Only if both residues have atoms
        #                 # Get indices for atoms in res_i and res_j
        #                 idx_i = torch.tensor(atoms_i, device=self.device_)
        #                 idx_j = torch.tensor(atoms_j, device=self.device_)
        #                 # Assign residue-pair value to all atom-pairs
        #                 # Need meshgrid for multi-dim assignment
        #                 mesh_i, mesh_j = torch.meshgrid(idx_i, idx_j, indexing='ij')
        #                 z_embeddings_atom[b_idx, mesh_i, mesh_j, :] = z_embeddings_res[b_idx, res_i, res_j, :]
        # Placeholder: Keep z as residue-level for now, assume Stage D handles it
        z_embeddings_atom = None # Set to None if not bridged
        logger.debug(f"z_embeddings (residue level) shape: {z_embeddings_res.shape}")

        # 5. Prepare Conditioning Signal Dictionary for StageDContext
        # Adapt keys based on what StageDContext / run_stageD expects
        # Typically: s_trunk (residue), s_inputs (atom), pair/z_trunk (residue or atom)
        conditioning_signal = {
            "s_trunk": s_embeddings_res, # Keep residue level for potential use
            "pair": z_embeddings_res,    # Keep residue level for potential use
            "s_inputs": s_embeddings_atom,# Pass atom level
            "z_atom": z_embeddings_atom, # Pass atom level if bridged, else None
            # Pass other batch features needed by Stage D / Context
            "atom_mask": atom_mask,
            "atom_to_token_idx": atom_to_token_idx,
            # Add input_feature_dict content needed by ProtenixDiffusionManager
            # e.g., ref_element, ref_charge if used by conditioning
            "ref_element": batch.get("ref_element"),
            "ref_atom_name_chars": batch.get("ref_atom_name_chars"),
            # etc.
        }

        # Optional: Unified Latent Merger
        unified_latent = None
        if getattr(self.cfg, "merge_latent", True): # Default to True based on design docs
            logger.debug("--- Running Latent Merger ---")
            # Prepare inputs for the merger (ensure correct device)
            # Note: SimpleLatentMerger expects residue-level inputs based on its forward sig
            merge_inputs = LatentInputs(
                adjacency=output["adjacency"].to(self.device_),
                angles=output["torsion_angles"].to(self.device_), # Shape [B, N_res, C_angle]
                s_emb=s_embeddings_res,                           # Shape [B, N_res, C_s]
                z_emb=z_embeddings_res,                           # Shape [B, N_res, N_res, C_z]
                # partial_coords is optional, needs careful handling if atom-level
                partial_coords=None # Or pass output["coords"] if merger handles atom-level coords
            )
            unified_latent = self.latent_merger(merge_inputs) # Expected output [B, N_res, C_latent]
            conditioning_signal["unified_latent"] = unified_latent # Add to conditioning dict
            logger.debug(f"Unified Latent shape: {unified_latent.shape if unified_latent is not None else None}")
            # CRITICAL: If Stage D needs atom-level unified_latent, bridge it here.
            # Example: unified_latent_atom = residue_to_atoms(unified_latent, residue_map_list_of_lists_batched)
            # conditioning_signal["unified_latent_atom"] = unified_latent_atom

        logger.debug("--- Stage D Input Preparation Complete ---")
        ```
    *   **Critical Considerations:**
        *   **Bridging Accuracy:** The correctness of `derive_residue_atom_map` and `residue_to_atoms` is paramount. Errors here will propagate incorrect conditioning signals. Test this bridging extensively.
        *   **`z_embeddings` Bridging:** Determine if Stage D requires atom-level pair embeddings (`z_atom`). If so, implement the bridging (e.g., the replication logic sketched above). If not, pass `z_embeddings_res`.
        *   **Conditioning Dictionary Keys:** Ensure the keys in `conditioning_signal` match precisely what `StageDContext` and `run_stageD` expect. Rename keys if necessary (e.g., maybe Stage D expects `"pair"` instead of `"z_trunk"`).
        *   **Unified Latent Level:** Decide if the `unified_latent` should be residue-level (as produced by `SimpleLatentMerger`) or atom-level. Bridge if needed. Ensure Stage D's conditioning logic handles whichever level is provided.

---

**4. File: `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`) - Execute Stage D**

*   **Action 4.1:** Call `run_stageD` using `StageDContext`.
    *   **Code Snippet (Inside `training_step`, after input prep):**
        ```python
        logger.debug("--- Executing Stage D ---")
        from rna_predict.pipeline.stageD.context import StageDContext
        from rna_predict.pipeline.stageD.run_stageD import run_stageD

        # Create the context object, mapping prepared signals to context fields
        # Ensure atom_metadata contains at least residue_indices for bridging inside run_stageD
        atom_metadata_for_stageD = {
             "residue_indices": atom_to_token_idx.squeeze(0) # Assuming batch size 1 for now, needs adjustment if B > 1
             # Add other metadata like atom_names if available and needed
        }
        if 'atom_names' in batch: atom_metadata_for_stageD['atom_names'] = batch['atom_names'][0] # Assuming B=1

        stage_d_context = StageDContext(
            cfg=self.cfg,                      # Pass the main Hydra config
            coords=coords_noisy,               # Noisy coordinates are the primary input
            s_trunk=conditioning_signal["s_trunk"], # Residue-level trunk
            z_trunk=conditioning_signal["pair"],    # Residue-level pair
            s_inputs=conditioning_signal["s_inputs"],# ATOM-level inputs
            input_feature_dict=conditioning_signal, # Pass the whole dict for flexibility
            atom_metadata=atom_metadata_for_stageD,
            unified_latent=unified_latent,     # Pass if merger is used
            mode='train',                      # Explicitly set train mode
            noise_level=sigma_t,               # Pass noise level for potential use in loss weighting
            device=str(self.device_),          # Pass device string
            debug_logging=getattr(self.cfg.model.stageD, 'debug_logging', False)
        )

        try:
            # Call run_stageD, which should handle training mode internally
            stage_d_pred_output = run_stageD(stage_d_context) # Assuming run_stageD is adapted

            # --- Determine what Stage D returned ---
            # Check the type and content of stage_d_pred_output
            # Ideally, it returns predicted noise epsilon_hat or denoised coords x_hat_0
            if isinstance(stage_d_pred_output, tuple) and len(stage_d_pred_output) == 3:
                 # Example: Assume returns (x_denoised, sigma, x_gt_augment) like ProtenixDiffusionManager.train_diffusion_step
                 # We need either noise or denoised coords for loss calculation
                 # If it returns denoised coords:
                 stage_d_pred = stage_d_pred_output[1] # Assuming x_denoised is the second element
                 prediction_type = 'coords' # Flag for loss calculation
            elif isinstance(stage_d_pred_output, torch.Tensor):
                 # Assume it returned the direct prediction (e.g., noise)
                 stage_d_pred = stage_d_pred_output
                 # ****** ASSUMPTION: Need to know if this tensor is noise or coords ******
                 # For now, assume noise prediction based on simpler loss path
                 prediction_type = 'noise'
            else:
                 logger.error(f"Unexpected output type from run_stageD in train mode: {type(stage_d_pred_output)}")
                 stage_d_pred = None
                 prediction_type = None

            if stage_d_pred is None:
                 raise ValueError("Stage D training step did not return a usable prediction.")

            logger.debug(f"Stage D prediction type: {prediction_type}, shape: {stage_d_pred.shape if stage_d_pred is not None else 'None'}")

        except Exception as e:
            logger.error(f"Error during Stage D execution (run_stageD call) in training_step: {e}", exc_info=True)
            stage_d_pred = None
            prediction_type = None
            # Assign a dummy loss that requires grad to avoid Lightning errors
            loss_diffusion = torch.tensor(0.0, device=self.device_, requires_grad=True)

        logger.debug("--- Stage D Execution Complete ---")
        ```
    *   **Critical Considerations:**
        *   **Adapt `run_stageD` for Training:** The primary dependency is ensuring `run_stageD` (and `_run_stageD_impl`) correctly handles `mode='train'`. It needs to:
            *   Recognize the training mode.
            *   Pass the `noise_level` (`sigma_t`) to the underlying `DiffusionModule` or loss calculation.
            *   Return the appropriate prediction (noise or denoised coords) needed for the loss. Currently, `_run_stageD_impl` seems inference-focused and might just return the final coordinates without loss calculation. This **must** be adapted. Consider adding a dedicated `train_stageD` function or modifying `run_stageD` with conditional logic based on `context.mode`.
        *   **`StageDContext` Completeness:** Double-check that `StageDContext` correctly bundles *all* information needed by the diffusion model's conditioning mechanism and loss function (including potentially required elements from the original `input_feature_dict` like masks, residue types, etc.).
        *   **Batch Size > 1:** The current snippet assumes batch size 1 in places (like squeezing `atom_metadata['residue_indices']`). This needs generalization if `batch_size > 1`. `StageDContext` should likely hold batched tensors.

---

**5. File: `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`) - Implement Diffusion Loss**

*   **Action 5.1:** Calculate `L_diffusion`.
    *   **Code Snippet (Inside `training_step`, after Stage D execution):**
        ```python
        import torch.nn.functional as F # Ensure F is imported

        logger.debug("--- Calculating Diffusion Loss ---")
        loss_diffusion = torch.tensor(0.0, device=self.device_) # Default zero loss

        # Use the prediction_type determined in Step 4
        if stage_d_pred is not None and prediction_type is not None:
            if prediction_type == 'noise':
                logger.debug("Calculating noise prediction loss (MSE)")
                target_noise = epsilon # The noise added in step 3.2

                # Apply mask to compute loss only on non-padded atoms
                # Ensure masks have compatible shapes (e.g., [B, N_atom_padded])
                if atom_mask.shape != target_noise.shape[:-1]:
                     # Attempt to fix common case: mask is [B, N], target is [B, N, 3]
                     if atom_mask.shape == target_noise.shape[:-1]: # Should match if padding is correct
                          mask_for_loss = atom_mask.unsqueeze(-1) # -> [B, N, 1]
                     else:
                          logger.warning(f"Atom mask shape {atom_mask.shape} incompatible with target noise shape {target_noise.shape}. Using unmasked loss.")
                          mask_for_loss = torch.ones_like(target_noise[..., 0], dtype=torch.bool) # Unmasked
                else:
                     mask_for_loss = atom_mask.unsqueeze(-1) # -> [B, N, 1]

                # Ensure prediction shape matches target shape
                if stage_d_pred.shape != target_noise.shape:
                     logger.error(f"Shape mismatch between predicted noise {stage_d_pred.shape} and target noise {target_noise.shape}")
                     # Fallback to zero loss
                     loss_diffusion = torch.tensor(0.0, device=self.device_, requires_grad=True)
                else:
                     # Calculate masked MSE loss
                     error = (stage_d_pred - target_noise)**2
                     masked_error = error * mask_for_loss # Apply boolean mask
                     loss_diffusion = masked_error.sum() / (mask_for_loss.sum() * 3 + 1e-8) # Normalize by number of valid atom coordinates

                logger.debug(f"Loss Diffusion (Noise MSE): {loss_diffusion.item()}")

            elif prediction_type == 'coords':
                logger.debug("Calculating coordinate prediction loss (AF3-style - Placeholder MSE)")
                target_coords = coords_true # Ground truth coordinates

                # --- Placeholder: Implement AF3 Loss Here ---
                # Requires:
                # 1. weighted_rigid_align function (from AF3 supplement Algo 28)
                # 2. Atom weights wl (from AF3 Eq 4 - needs atom type info)
                # 3. weighted_mse loss
                # 4. SmoothLDDTLoss function (from AF3 supplement Algo 27)
                # 5. sigma_data from config
                # 6. Weight factor based on sigma_t (from AF3 Eq 6)

                # Simplified Placeholder: Masked MSE on *unaligned* coordinates (less accurate)
                logger.warning("Using simplified coordinate MSE loss (NO ALIGNMENT). Implement AF3 loss for best results.")
                mask_for_loss = atom_mask.unsqueeze(-1) # [B, N, 1]
                if stage_d_pred.shape != target_coords.shape:
                     logger.error(f"Shape mismatch between predicted coords {stage_d_pred.shape} and target coords {target_coords.shape}")
                     loss_diffusion = torch.tensor(0.0, device=self.device_, requires_grad=True)
                else:
                     error = (stage_d_pred - target_coords)**2
                     masked_error = error * mask_for_loss
                     loss_diffusion = masked_error.sum() / (mask_for_loss.sum() * 3 + 1e-8)

                logger.debug(f"Loss Diffusion (Coord MSE - No Align): {loss_diffusion.item()}")
            else:
                logger.error(f"Internal error: Unknown prediction_type '{prediction_type}'")
                loss_diffusion = torch.tensor(0.0, device=self.device_, requires_grad=True) # Dummy grad

        # Ensure loss requires grad if Stage D requires grad
        if not loss_diffusion.requires_grad and stage_d_pred is not None and stage_d_pred.requires_grad:
             loss_diffusion = loss_diffusion.clone().requires_grad_(True)

        logger.debug(f"--- Diffusion Loss Calculation Complete: loss={loss_diffusion.item()} ---")
        ```
    *   **Critical Considerations:**
        *   **Loss Choice:** Noise prediction loss is significantly simpler to implement than the full AF3 coordinate loss. Start with noise prediction unless the `ProtenixDiffusionManager` is specifically designed *only* for coordinate prediction.
        *   **Masking:** Correctly applying `atom_mask` is essential. Ensure its shape is broadcastable to the error tensor before multiplication and normalization.
        *   **AF3 Loss Implementation:** If implementing the coordinate loss, `weighted_rigid_align` and `SmoothLDDTLoss` are non-trivial and must be implemented correctly based on the AF3 supplement. Atom type information is needed for `wl`. The noise-level weighting (`weight_factor`) is also important.

*   **Action 5.2:** Verification of Gradient Flow.
    *   **Details:** Perform the gradient check meticulously.
    *   **Procedure:**
        1.  Ensure `self.stageD` and relevant parts of `self.stageB_pairformer` and `self.stageB_torsion` (e.g., LoRA adapters if used) have `requires_grad=True`. Freeze other parts if necessary (`self.stageA`, base weights of LoRA models).
        2.  In `training_step`, temporarily set `total_loss = loss_diffusion`.
        3.  Run `trainer.fit(model, dataloader, max_steps=1)`.
        4.  Iterate through `model.stageD.named_parameters()`: Check `p.grad is not None` and `p.grad.abs().sum() > 0`.
        5.  Iterate through relevant conditioning parameters (e.g., `model.stageB_pairformer.lora_params()`, `model.stageB_torsion.lora_params()`, `model.latent_merger.parameters()`): Check `p.grad is not None` and `p.grad.abs().sum() > 0`.
    *   **Debugging Missing Gradients:**
        *   Check for `.detach()` calls anywhere in the path from conditioning inputs -> Stage D -> loss.
        *   Ensure the `conditioning_signal` passed to Stage D is correctly derived from trainable upstream modules.
        *   Verify `run_stageD` preserves the computation graph when in training mode.
        *   Check the diffusion loss calculation maintains the graph connection to `stage_d_pred`.

---

**Final Checkpoints for Phase 2:**

1.  **Data Flow Confirmed:** Stage B outputs -> Bridging -> Conditioning Signal -> Stage D -> Prediction (`stage_d_pred`).
2.  **Stage D Training Mode:** `run_stageD` or equivalent must correctly execute the *training* step of the diffusion model, using `coords_noisy` and `sigma_t`, and returning the appropriate prediction (noise or denoised coords).
3.  **Loss Function Correctness:** The chosen diffusion loss (`L_diffusion`) must match the `prediction_type` returned by Stage D and correctly use the target (`epsilon` or `coords_true`) and `atom_mask`.
4.  **Gradient Propagation Verified:** Gradients from `L_diffusion` must successfully reach parameters in Stage D *and* the upstream modules providing the conditioning signal.

Completing these steps thoroughly is essential for successfully training the Stage D component and enabling end-to-end learning driven by the diffusion process.
=====
Okay, let's perform a deep dive into the inputs required for the Stage D diffusion process in the `RNA_PREDICT` pipeline, paying close attention to the role of Stage A's input embedding.

**1. Stage D Overview**

Stage D (`rna_predict/pipeline/stageD/`) is responsible for refining the 3D coordinates of the RNA structure, likely starting from coordinates generated by Stage C. It employs a diffusion model, inspired by techniques like those in AlphaFold 3, which iteratively denoises coordinates conditioned on various structural and sequence features.

**2. Main Entry Point and Top-Level Inputs (`run_stageD_unified.py`)**

The primary entry point seems to be `run_stageD_diffusion` within `run_stageD_unified.py`. This function expects a configuration object, specifically an instance of `DiffusionConfig` (defined in `rna_predict/pipeline/stageD/diffusion/utils/config_types.py`), which bundles the main inputs:

*   **`partial_coords` (torch.Tensor):** The initial 3D coordinates to be refined. These likely come from Stage C's reconstruction. Crucially, the diffusion model itself (`DiffusionModule`) operates at the **atom level**. Shape is expected to be `[Batch, N_atom, 3]`.
*   **`trunk_embeddings` (Dict[str, torch.Tensor]):** A dictionary containing core feature embeddings. These are expected *before* the bridging step and are likely **residue-level**. The bridging function converts them to atom-level. Key embeddings typically include:
    *   `s_trunk`: Single representation from upstream (e.g., Pairformer in Stage B). Expected shape *before* bridging: `[Batch, N_residue, C_s]`.
    *   `s_inputs`: Single representation derived from input features (potentially including Stage A output). Expected shape *before* bridging: `[Batch, N_residue, C_s_inputs]`.
    *   `pair` (or `z_trunk`): Pair representation from upstream (e.g., Pairformer). Expected shape *before* bridging: `[Batch, N_residue, N_residue, C_z]`.
*   **`diffusion_config` (DictConfig / Dict):** A nested configuration structure containing parameters specific to the diffusion process itself (noise schedule, inference steps, model architecture details, etc.).
*   **`input_features` (Optional[Dict[str, Any]]):** A dictionary containing various other input features, potentially a mix of atom-level and residue-level data *before* bridging. The bridging step standardizes these. Important keys include:
    *   `atom_metadata`: Contains `residue_indices`, `atom_names`, etc. Crucial for mapping atoms to residues.
    *   `ref_element`, `ref_atom_name_chars`, `ref_charge`, `ref_mask`: Atom-level reference features.
    *   `restype`, `profile`: Residue-level features.
    *   `sequence`: The RNA sequence string.
*   **`mode` (str):** "inference" or "train".
*   **`device` (str):** "cpu", "cuda", or "mps".
*   **`debug_logging` (bool):** Controls logging level.
*   **`cfg` (DictConfig):** The overall Hydra configuration object, passed down for accessing various parameters.
*   **`atom_metadata` (Optional[Dict[str, Any]]):** Explicitly passed metadata (overlaps with `input_features`). Required for bridging.

**3. The Crucial Bridging Step (`bridging/residue_atom_bridge.py`)**

Before the core diffusion model (`DiffusionModule`) sees the inputs, the `bridge_residue_to_atom` function is called within `run_stageD_unified.py`. This is a critical step:

*   **Input:** Takes `BridgingInput` containing the *residue-level* `trunk_embeddings` (`s_trunk`, `s_inputs`, `pair`/`z_trunk`), `partial_coords` (atom-level), `input_features`, and `sequence`.
*   **Process:**
    *   Uses `atom_metadata` (specifically `residue_indices`) to create or validate a `residue_atom_map`.
    *   Calls `process_trunk_embeddings`: Expands the **residue-level** `s_trunk`, `s_inputs`, and `pair`/`z_trunk` embeddings into **atom-level** embeddings using the map. For example, `s_trunk` `[B, N_residue, C]` becomes `[B, N_atom, C]`. `pair` `[B, N_residue, N_residue, C]` becomes `[B, N_atom, N_atom, C]`.
    *   Calls `process_input_features`: Ensures other features (like `restype`, `profile`) are expanded to the atom level and creates the essential `atom_to_token_idx` tensor (mapping each atom to its residue index).
*   **Output:** Returns atom-level `partial_coords`, atom-level `trunk_embeddings`, and atom-level `input_features`.

**Therefore, the inputs *directly* consumed by the core diffusion machinery (`ProtenixDiffusionManager` -> `DiffusionModule`) are primarily at the atom level, having been transformed by this bridging step.**

**4. Core Diffusion Inputs (`ProtenixDiffusionManager` & `DiffusionModule`)**

The `ProtenixDiffusionManager` orchestrates the diffusion steps, calling the `DiffusionModule`. The *effective* inputs to the `DiffusionModule` (after bridging and internal processing by the manager/conditioning layers) are:

*   **`x_noisy` (torch.Tensor):** Noisy atom coordinates. Shape `[Batch, N_sample, N_atom, 3]`. Derived from the initial `partial_coords`.
*   **`t_hat_noise_level` (torch.Tensor):** The noise level for the current step. Shape `[Batch, N_sample]` or broadcastable.
*   **`input_feature_dict` (Dict[str, Any]):** Contains *atom-level* features after bridging. Crucially includes:
    *   `atom_to_token_idx`: Map from atom index to residue index. Shape `[Batch, N_atom]`.
    *   `ref_pos`: Reference positions (can be the initial `partial_coords`). Shape `[Batch, N_atom, 3]`.
    *   `ref_mask`: Mask indicating valid atoms. Shape `[Batch, N_atom, 1]`.
    *   `ref_element`, `ref_atom_name_chars`, `ref_charge`: Atom properties.
    *   `restype`, `profile`: Residue-level features *broadcasted to the atom level*.
*   **`s_inputs` (torch.Tensor):** Single representation derived from input features, now at the **atom level**. Shape `[Batch, N_sample, N_atom, C_s_inputs]`.
*   **`s_trunk` (torch.Tensor):** Single representation from upstream stages, now at the **atom level**. Shape `[Batch, N_sample, N_atom, C_s]`.
*   **`z_trunk` (torch.Tensor):** Pair representation from upstream stages, now at the **atom level**. Shape `[Batch, N_sample, N_atom, N_atom, C_z]`. (Note: Referred to as `pair` in some parts of the bridging code, but likely corresponds to `z_trunk` conceptually).
*   **`unified_latent` (Optional[torch.Tensor]):** An optional conditioning vector potentially combining information from multiple upstream stages (A, B, C). Its exact processing depends on how `DiffusionConditioning` uses it.

**5. The Role of Stage A Input Embedding**

Now, let's address the connection to Stage A's input embedding:

*   **Stage A Output:** The `InputFeatureEmbedder` (in `rna_predict/pipeline/stageA/input_embedding/current/embedders.py`), which uses the `AtomAttentionEncoder`, processes atom features (like element type, charge) and produces a **token-level (residue-level)** embedding. Its output dimension is `c_token`.
*   **Indirect Connection:** This Stage A output *does not directly feed* into the core Stage D `DiffusionModule` which operates at the atom level.
*   **How it Feeds In:** The token-level embedding produced by Stage A likely contributes to one or both of the **residue-level** `trunk_embeddings` (`s_trunk`, `s_inputs`) *before* they are passed to the Stage D bridging function.
    *   It might be directly used as `s_inputs` (residue-level).
    *   It might be passed to Stage B (e.g., Pairformer) which then produces the `s_trunk` and `pair`/`z_trunk` embeddings (still residue-level).
*   **Transformation:** The key is that the residue-level information, potentially influenced by Stage A's output, is explicitly converted to the atom-level representations required by Stage D's `DiffusionModule` via the `bridge_residue_to_atom` function.

**In Summary:**

Stage D diffusion requires **atom-level** inputs:

1.  **Initial/Noisy Atom Coordinates:** (`partial_coords` -> `x_noisy`)
2.  **Atom-Level Single Embeddings:** (`s_trunk`, `s_inputs` *after* bridging) - These originated as residue-level embeddings potentially derived from Stage A/B/C outputs.
3.  **Atom-Level Pair Embeddings:** (`pair`/`z_trunk` *after* bridging) - Originated as residue-level embeddings from Stage B/C.
4.  **Atom-Level Reference Features:** (`ref_mask`, `ref_element`, etc. from `input_features` *after* bridging).
5.  **Atom-to-Residue Mapping:** (`atom_to_token_idx` from `input_features` *after* bridging).
6.  **Noise Level:** (`t_hat_noise_level`).
7.  **Optional Unified Latent Conditioning:** (From merger).

Stage A's input embedding produces residue-level features. These features influence the residue-level `trunk_embeddings` passed *into* the Stage D module, but they must first be transformed into atom-level representations by the crucial bridging step (`bridge_residue_to_atom`) before being consumed by the core atom-level diffusion model (`DiffusionModule`).
====
Okay, let's perform a deep dive into **Phase 3: Final Integration**, focusing on combining the losses, reviewing configurations, and establishing a robust testing strategy.

**Goal:** Finalize the `RNALightningModule.training_step` by combining the angle and diffusion losses, ensure the entire configuration is consistent (especially dimensions), and define tests to verify end-to-end gradient flow and training stability.

---

**6. File: `rna_predict/training/rna_lightning_module.py` (`RNALightningModule`) - Combine and Log Losses**

*   **Action 6.1: Define Loss Weights in Configuration.**
    *   **Rationale:** Loss weights should be configurable hyperparameters for easy tuning.
    *   **File:** `rna_predict/conf/config_schema.py`
        *   **Action:** Add a `TrainingConfig` dataclass (if not already present) to hold training-specific hyperparameters like loss weights.
            ```python
            from dataclasses import dataclass, field
            # ... other imports ...

            @dataclass
            class TrainingConfig:
                """Configuration for training-related parameters."""
                checkpoint_dir: str = field(default="outputs/checkpoints", metadata={"help": "Directory to save checkpoints"})
                w_diffusion: float = field(default=1.0, metadata={"help": "Weight for the diffusion loss term"})
                w_angle: float = field(default=0.1, metadata={"help": "Weight for the direct angle loss term"})
                # Add other training params like learning_rate, epochs etc. if managing them here

            # ... Add TrainingConfig to the main RNAConfig ...
            @dataclass
            class RNAConfig:
                # ... other fields ...
                training: TrainingConfig = field(default_factory=TrainingConfig)
                # ... other fields ...

            # ... Add TrainingConfig registration in register_configs() ...
            def register_configs() -> None:
                # ... other cs.store calls ...
                cs.store(group="training", name="default", node=TrainingConfig)
                # ...
            ```
    *   **File:** Create `rna_predict/conf/training/default.yaml`
        *   **Action:** Define the default weights.
            ```yaml
            # rna_predict/conf/training/default.yaml
            defaults:
              - _self_

            checkpoint_dir: outputs/checkpoints # Default checkpoint dir
            w_diffusion: 1.0
            w_angle: 0.1
            # Add other training defaults (e.g., learning_rate: 0.001)
            ```
    *   **File:** `rna_predict/conf/default.yaml`
        *   **Action:** Ensure the training config group is included in the main defaults list.
            ```yaml
            # rna_predict/conf/default.yaml
            defaults:
              - _self_
              - data: default
              - model/stageA@model.stageA
              # ... other model stages ...
              - training: default # <<< ADD THIS LINE
              - test_data@test_data
            # ... rest of the file ...
            ```

*   **Action 6.2: Calculate Combined Loss in `training_step`.**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`RNALightningModule.training_step`)
    *   **Action:** Combine `loss_angle` and `loss_diffusion` using weights from `self.cfg.training`. Handle cases where a loss might not have been computed (e.g., `load_ang=False`).
    *   **Code Snippet (End of `training_step`):**
        ```python
        # (Assuming loss_angle and loss_diffusion are calculated above if enabled/successful)
        # Initialize total loss
        total_loss = torch.tensor(0.0, device=self.device_, requires_grad=True) # Start with a zero tensor that requires grad

        # Access weights safely with defaults
        w_diffusion = getattr(getattr(self.cfg, 'training', {}), 'w_diffusion', 1.0)
        w_angle = getattr(getattr(self.cfg, 'training', {}), 'w_angle', 0.1)

        # Add diffusion loss if computed
        diffusion_loss_computed = 'loss_diffusion' in locals() and loss_diffusion is not None and torch.is_tensor(loss_diffusion)
        if diffusion_loss_computed:
            # Ensure loss_diffusion requires grad if its inputs did
            if not loss_diffusion.requires_grad and 'stage_d_pred' in locals() and stage_d_pred is not None and stage_d_pred.requires_grad:
                 loss_diffusion = loss_diffusion.clone().requires_grad_(True)
            if w_diffusion > 0:
                 total_loss = total_loss + w_diffusion * loss_diffusion
                 self.log('train/loss_diffusion', loss_diffusion, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            else:
                 # Log zero if weight is zero but loss was computed
                 self.log('train/loss_diffusion', 0.0, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Add angle loss if computed (depends on load_ang and successful calculation)
        angle_loss_computed = 'loss_angle' in locals() and loss_angle is not None and torch.is_tensor(loss_angle)
        if angle_loss_computed:
             # Ensure loss_angle requires grad if its inputs did
             if not loss_angle.requires_grad and 'output' in locals() and output["torsion_angles"].requires_grad:
                  loss_angle = loss_angle.clone().requires_grad_(True)
             if w_angle > 0:
                  total_loss = total_loss + w_angle * loss_angle
                  self.log('train/loss_angle', loss_angle, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
             else:
                 # Log zero if weight is zero but loss was computed
                 self.log('train/loss_angle', 0.0, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Log total loss
        self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Return the total loss for the optimizer
        return {"loss": total_loss}
        ```
    *   **Critical Considerations:**
        *   Gradient Requirement: Ensure `total_loss` retains `requires_grad=True` if any component loss had it. Starting `total_loss` as a zero tensor with `requires_grad=True` helps if the first added loss term happens to be zero but had graph history. Cloning component losses before adding might also be necessary if they were potentially modified in-place during logging.
        *   Logging: Use `sync_dist=True` for distributed training environments.
        *   Zero Weights: The logic handles cases where a weight is zero, ensuring the corresponding component isn't added to `total_loss` but is still logged (as 0.0) for monitoring consistency.

---

**7. File: `rna_predict/conf/**` - Configuration & Dimension Review**

*   **Action 7.1: Establish Central Dimension Definitions.**
    *   **File:** `rna_predict/conf/config_schema.py`
    *   **Action:** Define a `DimensionsConfig` dataclass to hold shared dimensions.
        ```python
        @dataclass
        class DimensionsConfig:
            """Centralized dimensions for consistency across stages."""
            # --- Stage B / Pairformer Output ---
            c_s: int = 384  # Single representation dimension (Pairformer -> Merger -> Stage D)
            c_z: int = 128  # Pair representation dimension (Pairformer -> Merger -> Stage D)

            # --- Stage B / TorsionBERT Output ---
            num_angles: int = 7 # Standard torsions (alpha-zeta, chi)
            angle_rep_dim: int = field(init=False) # Calculated: 7 or 14

            # --- Stage D Conditioning / Internal ---
            c_s_inputs: int = 449 # Dimension of initial token features (InputFeatureEmbedder output)
            c_token: int = 768 # Token dimension within Stage D Transformer
            c_atom: int = 128 # Atom embedding dimension within Stage D
            c_atompair: int = 16 # Atom pair embedding dimension within Stage D
            c_noise_embedding: int = 32 # Noise level embedding dimension

            # --- Latent Merger ---
            latent_merger_hidden_dim: int = 256 # Example hidden dim for merger MLP/Transformer
            latent_merger_output_dim: int = 512 # Example output dim for unified latent

            def __post_init__(self):
                 # Example: calculate angle_rep_dim based on TorsionBERT config (needs access to it)
                 # This is tricky with pure dataclasses. Better to handle in RNALightningModule __init__
                 # Or require angle_mode to be passed here. Let's assume 14 for sin/cos default.
                 self.angle_rep_dim = self.num_angles * 2 # Default assumes sin/cos output

        # Add DimensionsConfig registration
        def register_configs() -> None:
             # ... other registrations ...
             cs.store(group="dimensions", name="default", node=DimensionsConfig)

        # In RNAConfig:
        @dataclass
        class RNAConfig:
             # ...
             dimensions: DimensionsConfig = field(default_factory=DimensionsConfig)
             # ...
        ```
    *   **File:** `rna_predict/conf/dimensions/default.yaml` (New file)
        ```yaml
        # Default dimension values
        c_s: 384
        c_z: 128
        num_angles: 7
        c_s_inputs: 449
        c_token: 768
        c_atom: 128
        c_atompair: 16
        c_noise_embedding: 32
        latent_merger_hidden_dim: 256
        latent_merger_output_dim: 512
        ```
    *   **File:** `rna_predict/conf/default.yaml`
        ```yaml
        defaults:
          - _self_
          - data: default
          - dimensions: default # <<< ADD THIS
          # ... model stages ...
          - training: default
          - test_data@test_data
        ```

*   **Action 7.2: Use Interpolation in Stage YAMLs.**
    *   **Rationale:** Link stage-specific dimensions back to the central `dimensions` config group.
    *   **Example (`rna_predict/conf/model/stageB_pairformer.yaml`):**
        ```yaml
        # Core model parameters
        n_blocks: 48 # Example specific value
        n_heads: 8   # Example specific value
        c_z: ${dimensions.c_z}          # Interpolated
        c_s: ${dimensions.c_s}          # Interpolated
        # ... other pairformer params ...
        ```
    *   **Example (`rna_predict/conf/model/stageD_diffusion.yaml`):**
        ```yaml
        # ... other diffusion keys ...
        model_architecture:
          c_s: ${dimensions.c_s}          # Interpolated
          c_z: ${dimensions.c_z}          # Interpolated
          c_s_inputs: ${dimensions.c_s_inputs} # Interpolated
          c_token: ${dimensions.c_token}    # Interpolated
          c_atom: ${dimensions.c_atom}      # Interpolated
          c_atompair: ${dimensions.c_atompair} # Interpolated
          c_noise_embedding: ${dimensions.c_noise_embedding} # Interpolated
          sigma_data: 1.0 # Example specific value
        # ... other model_arch keys ...
        ```
    *   **Example (`rna_predict/conf/model/latent_merger.yaml` - If created):**
        ```yaml
        defaults:
          - _self_

        merge_method: "concat" # Example
        angle_dim: ${dimensions.angle_rep_dim} # Interpolated (Needs calculation logic)
        s_dim: ${dimensions.c_s}             # Interpolated
        z_dim: ${dimensions.c_z}             # Interpolated
        hidden_dim: ${dimensions.latent_merger_hidden_dim} # Interpolated
        output_dim: ${dimensions.latent_merger_output_dim} # Interpolated
        # ... other merger params ...
        ```
    *   **Action:** Apply this interpolation pattern systematically to all relevant dimension parameters in all stage `.yaml` files and corresponding `config_schema.py` dataclasses.

*   **Action 7.3: Verify Bridging Logic Dimensions.**
    *   **File:** `rna_predict/training/rna_lightning_module.py` (`training_step`)
    *   **Action:** When preparing the `conditioning_signal` for Stage D, ensure the expected dimensions match those defined in `cfg.dimensions`.
        *   Check `s_embeddings_atom` has feature dim `cfg.dimensions.c_s`.
        *   Check `z_embeddings_res` (or `z_embeddings_atom`) has feature dim `cfg.dimensions.c_z`.
        *   If using `unified_latent`, ensure its dimension matches `cfg.dimensions.latent_merger_output_dim` and that Stage D conditioning expects this dimension.
    *   **File:** `rna_predict/pipeline/stageD/diffusion/components/diffusion_conditioning.py` (`DiffusionConditioning`)
    *   **Action:** Ensure the dimensions passed during initialization (`c_s`, `c_z`, `c_s_inputs`) are derived from the *interpolated* config values, ultimately linking back to `cfg.dimensions`.

*   **Verification:**
    *   Run `python rna_predict/training/train.py --cfg job`. Examine the output YAML. Verify that dimensions like `c_s`, `c_z` are identical across `model.stageB_pairformer`, `model.stageD.diffusion.model_architecture`, etc.
    *   Instantiate `RNALightningModule` with the config and print the shapes of internal layers (e.g., `self.latent_merger.mlp[0].in_features`) to confirm they match the resolved config dimensions.

---

**8. Testing Strategy**

*   **Action 8.1: Unit Tests.**
    *   **Files:** `tests/unit/test_losses.py`, `tests/unit/test_merger.py`, `tests/unit/test_bridging.py`.
    *   **Actions:**
        *   Write tests for `L_angle` calculation: include sin/cos conversion, MSE logic, masking.
        *   Write tests for `L_diffusion` calculation:
            *   If noise prediction: test MSE with masking.
            *   If coordinate prediction: unit test `weighted_rigid_align`, `SmoothLDDTLoss`, weighted MSE, and the final combination with the `sigma_t` factor.
        *   Write tests for `SimpleLatentMerger` (or chosen merger) checking output shape given input shapes matching `cfg.dimensions`.
        *   Refine tests for `residue_to_atoms` and bridging logic in `training_step` to ensure atom-level outputs have correct shapes based on `cfg.dimensions`.

*   **Action 8.2: Integration Test (`tests/integration/test_lightning_trainer.py`).**
    *   **Setup:**
        *   Use a minimal dataset (1-2 samples with pre-computed angles).
        *   Use a minimal config (`test_config.yaml`) with small dimensions (e.g., `c_s=16, c_z=8`), 1 block per stage, `load_ang=true`, and loss weights `w_diffusion=1.0, w_angle=1.0`.
    *   **Execution:**
        ```python
        import pytest
        import torch
        import lightning as L
        from hydra import compose, initialize
        from rna_predict.training.rna_lightning_module import RNALightningModule
        from rna_predict.dataset.loader import RNADataset
        from rna_predict.dataset.collate import rna_collate_fn
        from torch.utils.data import DataLoader

        @pytest.mark.slow # Mark as slow integration test
        def test_end_to_end_gradient_flow():
            # Assumes test_config.yaml exists and points to minimal data
            with initialize(config_path="../../rna_predict/conf", version_base=None):
                # Compose with minimal overrides for training
                cfg = compose(config_name="test_config.yaml", overrides=[
                    "data.load_ang=true",
                    "training.w_diffusion=1.0",
                    "training.w_angle=1.0",
                    "data.batch_size=1", # Use batch size 1 for easier debugging
                    "data.max_res=30", # Small max length
                    # Minimal model dimensions override example
                    "dimensions.c_s=16",
                    "dimensions.c_z=8",
                    "dimensions.c_atom=4",
                    "dimensions.c_atompair=2",
                    "dimensions.c_token=32",
                    "dimensions.c_s_inputs=10",
                    "dimensions.c_noise_embedding=4",
                    "model.stageB_pairformer.n_blocks=1",
                    "model.stageD.diffusion.transformer.n_blocks=1",
                    # Ensure Stage D runs in train mode for the test
                    "model.stageD.mode=train",
                    "model.stageD.diffusion.mode=train",
                ])

            # Setup Model, Dataset, DataLoader
            model = RNALightningModule(cfg)
            dataset = RNADataset(cfg=cfg, load_adj=False, load_ang=True, verbose=True)
            # Ensure dataset is not empty
            if len(dataset) == 0:
                 pytest.skip("Skipping gradient flow test: Dataset is empty.")
            dataloader = DataLoader(dataset, batch_size=cfg.data.batch_size, collate_fn=rna_collate_fn, num_workers=0)

            # Define parameters to check gradients for
            params_to_check = {
                "stageB_torsion": list(model.stageB_torsion.parameters()),
                "stageB_pairformer": list(model.stageB_pairformer.parameters()), # Check all if not using LoRA
                "latent_merger": list(model.latent_merger.parameters()),
                "stageD": list(model.stageD.parameters()) # Check diffusion manager/module params
            }
            # Filter for trainable params only (requires_grad=True)
            trainable_params = {name: [p for p in params if p.requires_grad] for name, params in params_to_check.items()}

            # Ensure there are trainable parameters before proceeding
            total_trainable = sum(len(p) for p in trainable_params.values())
            if total_trainable == 0:
                 pytest.skip("Skipping gradient flow test: No trainable parameters found in relevant modules.")

            # Simple Training Loop (instead of Trainer.fit for direct control)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            model.train() # Set model to training mode
            batch = next(iter(dataloader))

            # --- Manual Training Step ---
            optimizer.zero_grad()
            loss_dict = model.training_step(batch, 0) # Get loss dict { "loss": total_loss }
            total_loss = loss_dict["loss"]
            assert torch.is_tensor(total_loss) and not torch.isnan(total_loss), "Loss is NaN or not a tensor"
            total_loss.backward()
            # --- End Manual Step ---

            # Check Gradients
            print("\n--- Checking Gradients ---")
            all_grads_present = True
            for name, params in trainable_params.items():
                has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
                print(f"Module: {name}, Trainable Params: {len(params)}, Has Non-Zero Grad: {has_grad}")
                if not params:
                     print(f"  (No trainable params found for {name})")
                elif not has_grad:
                    all_grads_present = False
                    print(f"  WARNING: No gradients flowed to trainable parameters in {name}!")
                    # Optional: Print grad norms for debugging
                    # for i, p in enumerate(params):
                    #     print(f"    Param {i}: grad={p.grad.abs().sum().item() if p.grad is not None else None}")

            assert all_grads_present, "Gradients did not flow back to all expected trainable components (TorsionBERT, Pairformer, Merger, StageD)."
            print("--- Gradient Check Passed ---")
        ```
    *   **Action:** Run this integration test. Debug any assertion failures related to missing gradients or NaN losses.

*   **Action 8.3: Training Monitoring.**
    *   **Setup:**
        *   Use a representative subset of data (e.g., 100-500 structures).
        *   Configure `lightning.pytorch.loggers.TensorBoardLogger`.
        *   Set `Trainer(logger=tb_logger, max_epochs=10)`.
    *   **Execution:** Run `python rna_predict/training/train.py` with appropriate overrides for the dataset subset and logger.
    *   **Analysis:** Open TensorBoard (`tensorboard --logdir ./lightning_logs`). Examine the plots for `train/loss`, `train/loss_diffusion`, `train/loss_angle`. Look for decreasing trends and stability. Experiment with different `w_diffusion` and `w_angle` values (e.g., `[1.0, 0.0]`, `[0.0, 1.0]`, `[1.0, 0.1]`, `[1.0, 1.0]`) in the config to understand their relative impact.

---

**Summary of Phase 3:**

This final phase ensures the components trained in Phases 1 & 2 work together correctly.
1.  **Loss Combination:** Implemented a weighted sum of `L_angle` and `L_diffusion` in `training_step`, controlled by Hydra config. Added robust logging.
2.  **Config Review:** Emphasized using `config_schema.py` and Hydra interpolation (`${dimensions...}`) to enforce dimensional consistency across all YAML files and code. Outlined key dimensions to track.
3.  **Testing:** Defined specific unit tests for loss components and bridging, a critical integration test checking end-to-end gradient flow to all trainable modules (TorsionBERT, Pairformer, Merger, StageD), and a strategy for monitoring training stability and loss convergence using TensorBoard.

Completing this phase successfully validates that the entire trainable pipeline (Stages B, D, and Merger) is correctly connected and receiving appropriate learning signals.
=====
Excellent points! These considerations highlight the practical nuances and potential friction points involved in implementing the proposed training plan. Let's analyze each one:

1.  **External Tool Dependency (DSSR) in Phase 1a:**
    *   **Analysis:** You are correct. Relying on DSSR introduces an external dependency that requires separate installation and licensing (albeit free for academics). This contrasts with a pure Python dependency like MDAnalysis.
    *   **Rationale Recap:** The choice favoured DSSR because of its specialized accuracy and robustness in handling RNA complexities (conventions, modifications) out-of-the-box. The hypothesis is that the effort saved in *not* having to manually implement and rigorously validate angle calculations for all edge cases (especially inter-residue angles like α, ε, ζ) using a general tool like MDAnalysis outweighs the one-time setup cost of DSSR. The risk of introducing subtle errors into the "ground truth" angles via manual implementation with MDAnalysis was deemed higher.
    *   **Plan's Mitigation:** The plan isolates this dependency into a single, explicit pre-computation script (`Phase 1a`).
    *   **Refinement/Action:**
        *   **Documentation:** The pre-computation script's README/documentation *must* clearly state the DSSR version dependency and link to its installation/licensing instructions.
        *   **Error Handling:** The script needs robust error handling to detect DSSR failures (e.g., binary not found, license issue, file parsing error) and provide informative messages.
        *   **Reproducibility:** Store the *exact* DSSR version used alongside the generated `.pt` files (e.g., in a manifest file) to ensure future reproducibility.

2.  **Complexity of AF3 Loss in Phase 2 (Action 5.1):**
    *   **Analysis:** Agreed. Implementing the full AlphaFold 3 coordinate loss (Eq 6, involving Algo 27 `SmoothLDDTLoss` and Algo 28 `weighted_rigid_align`) is considerably more complex than a standard MSE or even a noise-prediction MSE.
    *   **Rationale Recap:** The plan explicitly recommends starting with the simpler **noise prediction MSE loss** (`loss_diffusion = F.mse_loss(predicted_noise, actual_noise, reduction='none'); loss_diffusion = (loss_diffusion * mask_for_loss).sum() / (mask_for_loss.sum() * 3 + 1e-8)`) as the default `L_diffusion`, *unless* the `ProtenixDiffusionManager` is definitively designed to output denoised coordinates (`\hat{x}_0`).
    *   **Plan's Mitigation:** The plan flags the coordinate loss path as complex and requiring specific algorithm implementations.
    *   **Refinement/Action:**
        *   **Prioritize Noise Prediction:** Strongly emphasize implementing and testing the noise prediction loss first. Only switch to the coordinate prediction loss if absolutely necessary and after the rest of the pipeline is stable.
        *   **Sub-tasking:** If coordinate loss *is* required, break down its implementation into distinct sub-tasks: (a) implement `weighted_rigid_align`, (b) implement `SmoothLDDTLoss`, (c) ensure atom-type data is available for weights `wl`, (d) implement the noise-level weighting factor from Eq 6. Treat this as a separate, significant development effort.

3.  **`run_stageD` Implementation Details (Training Mode):**
    *   **Analysis:** Correct. The plan identifies *that* `run_stageD` needs modification but not the precise internal changes. `_run_stageD_impl` currently calls `run_diffusion_and_handle_output`, which in turn calls `ProtenixDiffusionManager.multi_step_inference`. This inference path is unsuitable for training.
    *   **Rationale Recap:** The plan flags this as needing investigation.
    *   **Refinement/Action:**
        *   **Recommended Approach:** Modify `ProtenixDiffusionManager` to have a distinct `train_step` method (or adapt its `forward` method based on `self.training` or a `mode` flag). This method should take noisy coordinates, conditioning signals, and the noise level (`sigma_t`) as input, perform *one* step of the diffusion model's forward pass (predicting noise or denoised coords), and return that prediction.
        *   Modify `run_stageD` / `_run_stageD_impl`: Add conditional logic based on `context.mode`. If `'train'`, it should call the new `diffusion_manager.train_step` method instead of `multi_step_inference`.
        *   **Interface Clarity:** Ensure the *return value* of `run_stageD` in training mode is clearly defined (e.g., just the predicted noise tensor, or a tuple/dict including it) so `RNALightningModule.training_step` knows what to expect for the loss calculation.

4.  **Bridging Function Robustness (`residue_to_atoms`, `derive_residue_atom_map`):**
    *   **Analysis:** Valid concern. The entire conditioning signal's correctness hinges on these utilities mapping residue-level information (like `s_embeddings`) to the correct atom-level representation expected by Stage D.
    *   **Rationale Recap:** The plan uses these functions but implicitly assumes their correctness.
    *   **Refinement/Action:**
        *   **Dedicated Unit Tests:** Implement specific unit tests for `rna_predict.utils.tensor_utils.embedding.residue_to_atoms` and `rna_predict.utils.tensor_utils.residue_mapping.derive_residue_atom_map`.
        *   **Test Cases:** These tests should cover:
            *   Standard RNA sequences.
            *   Sequences with varying lengths (requiring padding/truncation relative to `max_res`).
            *   Cases with `atom_mask` indicating missing atoms.
            *   Batch handling (if the functions are designed to be batched).
            *   Correct output shapes and dtypes.
            *   Correct mapping verification for a small, known example.
        *   **Placement:** Add these tests under `tests/unit/utils/` or similar. Run them as part of the prerequisites before or during Phase 2 implementation.

5.  **Hyperparameter Tuning (Loss Weights):**
    *   **Analysis:** Correct. The optimal `w_diffusion` and `w_angle` are unknown and will impact training dynamics.
    *   **Rationale Recap:** The plan sets up the structure (configurable weights) but defers the tuning.
    *   **Refinement/Action:**
        *   **Acknowledge as Future Work:** Explicitly state in documentation (e.g., a `docs/training_guide.md`) that loss weights are key hyperparameters requiring tuning *after* initial functionality is established.
        *   **Suggest Initial Values:** Recommend starting points (e.g., `w_diffusion=1.0`, `w_angle=0.1` or `w_angle=0.0` initially to isolate diffusion training).
        *   **Tuning Strategy:** Briefly mention potential tuning methods (manual adjustment based on validation loss curves, grid search, random search, Bayesian optimization).

**Revised Hypothesis (Implicit):**

While the implementation plan addresses the core requirements for fixing the training loop, successful execution hinges on careful handling of the identified "Minor Considerations," specifically: managing the DSSR dependency, choosing the appropriate diffusion loss complexity, correctly adapting the `run_stageD` function for training mode, rigorously testing the bridging utilities, and planning for subsequent hyperparameter tuning. These are not blockers but represent key areas requiring developer attention and potentially dedicated effort during or after the main implementation phases.
======
Okay, let's analyze those MDAnalysis timings in the context of using it for on-the-fly angle calculation during training data loading.

**Analysis of Timings:**

*   **Short Sequences (10-24 residues):** 20-53 milliseconds per structure. This is very fast and unlikely to be a bottleneck.
*   **Longer Sequence (240 residues):** 373 milliseconds (0.37 seconds) per structure. This is considerably longer.

**Context: DataLoader Performance**

*   **`__getitem__` Impact:** This calculation happens *every time* `RNADataset.__getitem__` is called for a sample.
*   **`num_workers`:** PyTorch's `DataLoader` uses multiple worker processes (`num_workers` in your config, default is 8) to prepare batches in parallel *ahead* of time. The goal is to have the next batch ready the moment the GPU finishes processing the current one.
*   **Bottleneck Identification:** The *overall throughput* of the DataLoader is limited by the time it takes for workers to prepare samples. If the *average* time per sample (including file I/O, sequence loading, coordinate loading, *and* angle calculation) multiplied by the batch size, divided by the number of workers, is significantly *longer* than the time your model takes for one training step on the GPU, then data loading becomes the bottleneck, and your GPU will sit idle waiting for data.

**Is 0.37s "Fast Enough"?**

*   **For a Single Item:** Maybe.
*   **For Training Throughput:** **Potentially problematic, especially for longer sequences.**
    *   If your dataset has many sequences around the 200-500 residue mark, each taking 0.3-0.5+ seconds just for angle calculation within `__getitem__`.
    *   Even with 8 workers, if a few workers happen to be processing long sequences simultaneously, the time to assemble a full batch could easily exceed the time for a GPU training step (which might be < 0.5s or even < 0.1s depending on model size and hardware).
    *   This leads to the GPU waiting, drastically slowing down your overall training time.

**Comparison with Pre-computation:**

*   **Loading `.pt` file:** Typically takes single-digit milliseconds (1-10ms), dominated by disk I/O.
*   **Difference:** Pre-computation removes the 20ms-370ms+ calculation cost *entirely* from the training loop's data loading path.

**Conclusion & Recommendation:**

While you *can* proceed with on-the-fly MDAnalysis calculation, and it *will* work functionally (especially if your dataset is mostly short sequences or your `num_workers` is high relative to GPU speed), **it is highly likely to become a performance bottleneck during training, potentially slowing it down significantly.**

**Strong Recommendation:**

Leverage the pre-computation script you already built (`compute_ground_truth_angles.py`).

1.  **Run Pre-computation with MDAnalysis:** Execute your script *once* for your entire dataset using the MDAnalysis backend:
    ```bash
    # Example command - adjust paths as needed
    uv run rna_predict/dataset/preprocessing/compute_ground_truth_angles.py \
        --input_dir /path/to/your/pdb_cif_files \
        --output_dir /path/to/store/angle_pt_files \
        --chain_id <your_default_or_logic> \
        --backend mdanalysis
    ```
    *(Ensure the output directory is accessible or adjust paths in `RNADataset` accordingly)*

2.  **Revert `_load_angles` in `loader.py`:** Change the `RNADataset._load_angles` method back to the *simpler version* designed in the refined plan, which just loads the pre-computed `.pt` file:
    ```python
    # Inside RNADataset class
    def _load_angles(self, row) -> Optional[torch.Tensor]:
        """Loads pre-computed ground truth torsion angles from a .pt file."""
        angle_path = None
        try:
            # --- Logic to determine the ANGLE file path ---
            # This MUST match how compute_ground_truth_angles.py saves files.
            # Example: Assuming output_dir from compute script is accessible
            # and uses naming like <pdb_id>_<chain_id>_angles.pt

            # !! ADAPT THIS PATH LOGIC !!
            # 1. Get base name (e.g., from row['id'] or derived from row['filepath'])
            base_name = row.get('id', 'unknown') # Or derive from filepath

            # 2. Construct expected .pt filename
            #    This assumes the compute script saved with _<chain>_angles.pt suffix
            chain_id = row.get('chain_id', 'A') # Get chain if available, else default
            angle_fname = f"{base_name}_{chain_id}_angles.pt"

            # 3. Define the directory where compute_script saved the files
            #    This might need to be configurable via cfg.data.angle_dir
            angle_dir = Path(self.cfg.data.angle_dir) # Add angle_dir to your config!

            angle_path = angle_dir / angle_fname
            # --- End Path Logic ---

            if not angle_path.is_file():
                if self.verbose:
                    logger.warning(f"Pre-computed angle file not found: {angle_path}. Returning None.")
                return None

            # Load the tensor data
            angles_tensor = torch.load(angle_path, map_location='cpu')

            # Basic validation (optional)
            if not isinstance(angles_tensor, torch.Tensor) or angles_tensor.dim() != 2 or angles_tensor.shape[1] != 7:
                 logger.error(f"Invalid data in pre-computed angle file {angle_path}. Shape: {angles_tensor.shape}, Type: {type(angles_tensor)}")
                 return None

            return angles_tensor.to(torch.float32) # Ensure correct dtype

        except Exception as e:
            logger.error(f"Error loading pre-computed angles for {row.get('id', 'N/A')} from {angle_path if angle_path else 'unknown path'}: {e}", exc_info=self.verbose)
            return None
    ```

**Advantages of this approach:**

*   **Best Performance:** Ensures fastest possible data loading during training.
*   **Leverages Existing Work:** Uses the `compute_ground_truth_angles.py` script you already wrote and tested.
*   **Clean Separation:** Keeps calculation (pre-computation) separate from loading (`RNADataset`).
*   **Future-Proof:** When DSSR is ready, you simply re-run the *same* pre-computation script with `--backend dssr` without changing the DataLoader.

While on-the-fly calculation is tempting as a temporary measure, investing the time now to run the pre-computation (even with MDAnalysis) will likely save significant training time later.
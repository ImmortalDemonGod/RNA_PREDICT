Okay, here is a comprehensive document detailing the goals, requirements, and tasks for Milestone 2 (M2), suitable for briefing a senior developer. It incorporates the analysis of your `rna_predict` project and leverages relevant patterns identified from the `ProjectEquiSurv` example.

---

**Project `rna_predict` - Milestone 2 Specification: Initial End-to-End Training & Prediction**

**1. Objective**

The primary goal of Milestone 2 (M2) is to achieve and validate the **functional trainability** of the integrated RNA 3D prediction pipeline developed in M1. This involves demonstrating that the end-to-end system (Sequence -> Stage A Adjacency -> Stage B Torsions/Pairs -> Unified Latent Merger -> Stage D Diffusion -> Final 3D Coords) can execute within a training loop, compute losses, backpropagate gradients correctly to the intended trainable parameters (primarily LoRA adapters), and generate initial 3D coordinate outputs.

**Key Focus:** Functionality and Infrastructure. Achieving high prediction accuracy is **not** the goal of M2; instead, we focus on proving the pipeline works mechanically, gradients flow correctly, and we can generate baseline outputs for subsequent debugging and refinement (M3).

**2. Prerequisites**

*   **M1 Completed:** The integrated pipeline codebase is complete and callable (e.g., via `rna_predict.pipeline.run_pipeline.run_full_pipeline`). This includes functional instances of:
    *   Stage A Adjacency predictor interface (`rfold_predictor.StageARFoldPredictor`).
    *   Stage B combined runner (`run_stage_b.run_stageB_combined`) integrating working TorsionBERT (`torsion_predictor.StageBTorsionBertPredictor`) and Pairformer (`pairwise.pairformer.PairformerWrapper`) modules.
    *   Unified Latent Merger (`run_pipeline.SimpleLatentMerger` or equivalent placeholder).
    *   Stage D Diffusion components (`diffusion.manager.ProtenixDiffusionManager`, `diffusion.model.DiffusionModule`).
    *   LoRA wrapping mechanism applied to TorsionBERT and Pairformer base models, ready for activation.

**3. Core Requirements (Definition of Done for M2)**

The following must be demonstrably working by the end of M2:

1.  **Trainable Pipeline Execution:** The primary training script (`rna_predict/train.py`) successfully executes the full forward pass of the integrated pipeline (A -> B -> Merger -> D) on batches of training data without crashing (resolving any shape, type, or device errors).
2.  **Loss Calculation:** A defined loss function (minimum: 3D coordinate-based loss like L1 or MSE comparing `final_coords` from Stage D to ground truth) is implemented and computes a valid (non-NaN, non-Inf) scalar loss value per batch.
3.  **Data Loading:** A functional PyTorch `DataLoader` (using `rna_predict.data.loader.RNADataset` or similar) must provide batches containing all necessary inputs: sequences, pre-computed adjacency matrices (from Stage A), ground truth 3D coordinates, and any other features required by the pipeline. Must correctly handle padding/batching.
4.  **Optimizer Configuration & Parameter Targeting:** An optimizer (e.g., AdamW) is correctly configured to update *only* the specified trainable parameters (LoRA adapters in TorsionBERT/Pairformer, UnifiedLatentMerger weights, Diffusion head/adapters). A clear mechanism exists to filter and provide only these parameters to the optimizer (e.g., a helper function `get_trainable_parameters(model)`). Base model weights must remain frozen (`requires_grad=False`).
5.  **Backpropagation & Weight Update:** `loss.backward()` and `optimizer.step()` execute successfully, indicating gradients are computed and applied to the trainable parameters without error.
6.  **Training Loop Functionality:** The training script completes at least one full epoch (or a substantial number of training steps) over the training dataset without critical runtime errors.
7.  **Checkpoint Saving:** The training process successfully saves model checkpoints containing the `state_dict` of *trainable* parameters and the optimizer state (e.g., using PyTorch Lightning's `ModelCheckpoint`).
8.  **Inference Script:** A separate inference script (`rna_predict/predict.py` or similar) exists for generating predictions from saved checkpoints.
9.  **Checkpoint Loading (Partial):** The inference script successfully loads the *trainable* parameters from a saved M2 checkpoint into the corresponding pipeline model structure using a partial loading mechanism (handling potentially missing base model weights in the checkpoint).
10. **Initial Prediction Generation:** The inference script, using a loaded M2 checkpoint, successfully generates and saves 3D coordinate outputs (e.g., `.pdb` or `.pt` format) for a representative subset of the validation data without crashing.
11. **LanceDB Logging Stub:** A stub LanceDB logger is present in the codebase (`rna_predict/utils/lance_logger.py`), with no-op logging methods and a config flag (`PipelineConfig.lance_db.enabled`) controlling activation. For M2, this logger is a placeholder; real logging will be implemented in M3.

**4. Key Technical Tasks & Recommended Approaches**

To achieve the M2 requirements, the following technical tasks should be prioritized, leveraging patterns from `ProjectEquiSurv` where applicable:

**4.1. Implement Training Framework (Leverage PyTorch Lightning)**

*   **Task:** Refactor the training logic into a PyTorch Lightning `LightningModule` (e.g., `RNALightningModule`).
*   **Details:**
    *   Encapsulate model instantiation (likely involving the setup logic from `run_full_pipeline`) within the module's `__init__`. The module should hold instances of the stage predictors, merger, and diffusion manager.
    *   Implement `training_step`: Takes a batch, runs the necessary steps of the forward pass (A->B->Merger->D), computes the primary 3D loss (and optional auxiliary losses like angle loss), logs training loss (`self.log`), returns the loss tensor.
    *   Implement `validation_step`: Similar to `training_step` but uses `torch.no_grad()`. Logs validation loss and basic coordinate metrics (e.g., MAE).
    *   Implement `configure_optimizers`: Defines the optimizer (e.g., AdamW) targeting *only* trainable parameters (Task 4.3) and optionally sets up an LR scheduler.
    *   Override `load_state_dict` or use the `on_load_checkpoint` hook to integrate partial checkpoint loading logic (Task 4.4).
    *   Integrate config flags for LanceDB logging (stub for M2, see `PipelineConfig.lance_db`).
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/training/lightning_survival_module.py` (`LightningSurvivalModule`)
    *   `quick_fixes/advanced/training/optim_wrappers.py` (`configure_optimizers`)
*   **Benefit:** Significantly reduces boilerplate for training loops, multi-device handling, logging, and checkpointing.

**4.2. Establish Configuration Management (Leverage Hydra)**

*   **Task:** Manage all hyperparameters and settings (paths, model dimensions, LoRA ranks, loss weights, training settings) using Hydra.
*   **Details:**
    *   Create a `rna_predict/conf/` directory with subdirectories (e.g., `model`, `training`, `data`, `lora`).
    *   Define configuration schemas using dataclasses (e.g., `rna_predict/conf/config_schema.py`) for hierarchical configuration access.
    *   Create corresponding `.yaml` files (e.g., `conf/train.yaml`, `conf/model/default.yaml`, `conf/lora/torsionbert.yaml`).
    *   Use `@hydra.main(config_path="conf", config_name="train")` decorator in `train.py` and `predict.py`. Access config via `cfg: DictConfig`.
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/pipeline/main.py` (`@hydra.main`)
    *   `quick_fixes/conf/config_schema.py`
*   **Benefit:** Provides flexible, reproducible, and command-line-friendly configuration.

**4.3. Implement Trainable Parameter Filtering**

*   **Task:** Create a utility function (e.g., in `rna_predict/core/utils.py`) or a method within `RNALightningModule` to identify and return only the parameters intended for training.
*   **Details:** Iterate through `self.model.named_parameters()` (assuming the full pipeline is `self.model`). Filter based on parameter names (e.g., check if `lora` is in the name, or if the parameter belongs to the `Merger` or `Diffusion` head). Pass only this filtered iterator/list to the optimizer in `configure_optimizers`.
*   **Verification:** Add a log statement at the start of training printing `sum(p.numel() for p in model.parameters() if p.requires_grad)` vs `sum(p.numel() for p in model.parameters())`.

**4.4. Implement Partial Checkpoint Loading**

*   **Task:** Ensure checkpoints saved during M2 (containing only trainable weights) can be loaded correctly for inference or resuming training.
*   **Details:**
    *   Copy or adapt the `partial_load_state_dict` function from `ProjectEquiSurv`.
    *   In `RNALightningModule`, override the `load_state_dict` method. Inside, call `partial_load_state_dict(self, state_dict, strict=False)` instead of the default `super().load_state_dict`.
    *   In `predict.py`, after instantiating the model, load the checkpoint using `torch.load` and then call `partial_load_state_dict(model, checkpoint['state_dict'], strict=False)`.
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/hypercloning/checkpoint_utils.py` (`partial_load_state_dict`)
*   **Benefit:** Critical for LoRA workflow.

**4.5. Build Data Handling Pipeline**

*   **Task:** Implement `RNADataset` and `DataLoader` for training and validation.
*   **Details:**
    *   Create/Refine `RNADataset` in `rna_predict/data/loader.py`. The `__getitem__` should return a dictionary containing:
        *   `sequence`: Raw string.
        *   `adjacency`: Precomputed `[N, N]` tensor.
        *   `coords_gt`: Ground truth `[N_atoms, 3]` tensor.
        *   Any other features needed for `features/initial_embeddings.py`.
    *   Implement data splitting logic (e.g., random split indices) in `train.py` using PyTorch `Subset`.
    *   Instantiate `DataLoader`s with appropriate `batch_size`, `num_workers`, and a custom `collate_fn` if necessary to handle variable length sequences (e.g., padding within a batch).
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/data/graph_survival_dataset.py` (Pattern for Dataset class)
    *   `quick_fixes/advanced/data/split_manager.py` (Pattern for splitting logic)

**4.6. Implement Training Script (`train.py`)**

*   **Task:** Create the main script (`rna_predict/train.py`) using PyTorch Lightning and Hydra.
*   **Details:**
    *   Use `@hydra.main`.
    *   Instantiate `RNALightningModule` using the Hydra config (`cfg`).
    *   Instantiate `DataLoader`s.
    *   Instantiate PyTorch Lightning `Trainer`, configuring callbacks:
        *   `ModelCheckpoint` (monitor validation loss, save top-k checkpoints).
        *   `LearningRateMonitor`.
        *   Optionally `HydraConfigSaverCallback` (from `ProjectEquiSurv` example).
    *   Call `trainer.fit(model, train_dataloader, val_dataloader)`.
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/pipeline/pipeline_train_single.py` (`train_single_run`)

**4.7. Implement Inference Script (`predict.py`)**

*   **Task:** Create the script (`rna_predict/predict.py`) to generate 3D coordinates from a trained checkpoint.
*   **Details:**
    *   Use `@hydra.main`.
    *   Load model configuration (`cfg`).
    *   Instantiate the pipeline model structure (e.g., potentially wrapped in the `RNALightningModule` or directly using the orchestrator logic).
    *   Load *trained* weights from the specified checkpoint path using the `partial_load_state_dict` logic (Task 4.4).
    *   Set `model.eval()`.
    *   Load input sequences (e.g., from a file specified in `cfg`).
    *   Prepare features (including precomputed adjacency) for each sequence.
    *   Iterate through sequences, call the pipeline's inference function (e.g., `run_full_pipeline`) with `torch.no_grad()`.
    *   Save the resulting `final_coords` tensor to a file (e.g., `.pdb` using BioPython or `.pt`).
*   **Reference (`ProjectEquiSurv`):**
    *   `quick_fixes/advanced/kaggle_submission/ensemble_inference.py`
    *   `quick_fixes/advanced/kaggle_submission/inference_utils.py`

**5. Optional Enhancements for M2**

*   **Basic Logging:** Use `self.log("train/loss", loss)` in `training_step` and `self.log("val/loss", loss)` in `validation_step` within the LightningModule.
*   **Gradient Clipping:** Set `gradient_clip_val` in the `Trainer` arguments (e.g., `gradient_clip_val=1.0`).
*   **NaN/Inf Checks:** Add simple checks like `if torch.isnan(loss).any(): raise ValueError("NaN loss detected")` after loss calculation.
*   **Qualitative Output Check:** Manually view 1-2 generated PDBs.

**6. Definition of Done (M2)**

M2 is complete when all 11 "Must-Have" requirements listed in Section 3 are met and verified. This signifies that the integrated pipeline is demonstrably trainable using LoRA and capable of producing initial 3D outputs from saved checkpoints.

**7. Next Steps (Post-M2)**

Upon successful completion of M2, the focus shifts to **M3: Validation & Debugging**. This will involve implementing quantitative validation metrics (TM-score), systematically evaluating the initial M2 predictions, debugging geometric/numerical issues, and performing initial hyperparameter adjustments.

---

Important considerations:
Location and generation process for pre-computed Stage A adjacency matrices.

Location and structure of the LoRA configuration within the Hydra setup.

Exact implementation details for trainable parameter filtering and partial checkpoint loading.

Location and specific type (MSE, L1, FAPE?) of the primary 3D loss function.

Status and input/output handling of the UnifiedLatentMerger.

Confirmation that the necessary dependencies (Lightning, Hydra, PEFT) are installed.

Confirmation that tensor shape patching is active.
=========
V2:
Okay, here is a comprehensive document detailing the goals, requirements, and tasks for Milestone 2 (M2), suitable for briefing a senior developer. It incorporates the analysis of the `rna_predict` project and leverages relevant patterns identified from the `ProjectEquiSurv` example.

---

**Project `rna_predict` - Milestone 2 Specification: Initial End-to-End Training & Prediction**

**Document Version:** 1.0
**Date:** March 30, 2025
**Target Completion for M2:** ~April 7th-10th, 2025

**1. Objective**

The primary goal of Milestone 2 (M2) is to achieve and validate the **functional trainability** of the integrated RNA 3D prediction pipeline developed in M1. This involves demonstrating that the end-to-end system (Sequence -> Stage A Adjacency -> Stage B Torsions/Pairs -> Unified Latent Merger -> Stage D Diffusion -> Final 3D Coords) can execute within a training loop, compute losses, backpropagate gradients correctly to the intended trainable parameters (primarily LoRA adapters), and generate initial 3D coordinate outputs.

**Key Focus:** Functionality and Infrastructure. Achieving high prediction accuracy is **not** the goal of M2; instead, we focus on proving the pipeline works mechanically, gradients flow correctly, and we can generate baseline outputs for subsequent debugging and refinement (M3). The infrastructure established in M2 (training scripts, configuration, checkpointing) will be crucial for future iterations.

**2. Prerequisites**

*   **M1 Completed:** The integrated pipeline codebase is complete and functionally callable via `rna_predict.pipeline.run_pipeline.run_full_pipeline`. This implies:
    *   Functional interfaces/wrappers for Stage A (`StageARFoldPredictor`), Stage B (`run_stageB_combined` using `StageBTorsionBertPredictor` & `PairformerWrapper`), Stage C (`run_stageC`), and Stage D (`ProtenixDiffusionManager`, `run_stageD_diffusion`).
    *   The `UnifiedLatentMerger` (`SimpleLatentMerger` or equivalent) is integrated into the `run_full_pipeline` flow.
    *   LoRA adapters have been applied to the base TorsionBERT and Pairformer models (using `peft` or custom wrappers), and these wrapped models are used in the pipeline. Base weights are configured to be frozen.
    *   The necessary runtime patching (`rna_predict.pipeline.patching.shape_fixes.apply_tensor_fixes`) is integrated and callable.

**3. Core Requirements (Definition of Done for M2)**

The following must be demonstrably working by the end of M2:

1.  **Trainable Pipeline Execution:** The primary training script (`rna_predict/train.py`) successfully executes the full forward pass of the integrated pipeline (A -> B -> Merger -> D) on batches of training data without crashing (resolving any shape, type, or device errors).
2.  **Loss Calculation:** A defined loss function (minimum: 3D coordinate-based loss like L1/MSE between Stage D output `final_coords` and ground truth `coords_gt`) is implemented and computes a valid (non-NaN, non-Inf) scalar loss value per batch.
3.  **Data Loading:** A functional PyTorch `DataLoader` (using `rna_predict.data.loader.RNADataset` or similar) provides batches containing sequences, pre-computed adjacency matrices, ground truth 3D coordinates, and potentially other features. It must correctly handle padding/batching for variable-length sequences.
4.  **Optimizer Configuration & Parameter Targeting:** An optimizer (e.g., AdamW) is correctly configured to update *only* the specified trainable parameters (LoRA adapters in TorsionBERT/Pairformer, UnifiedLatentMerger weights, Diffusion head/adapters). A verified mechanism exists to filter and provide only these parameters to the optimizer.
5.  **Backpropagation & Weight Update:** `loss.backward()` and `optimizer.step()` execute successfully, showing gradients are computed and applied to the trainable parameters without error. Verification should confirm non-zero gradients for trainable parameters and zero gradients for frozen base parameters.
6.  **Training Loop Functionality:** The training script completes at least one full epoch over the training dataset without critical runtime errors, demonstrating stability.
7.  **Checkpoint Saving:** The training process successfully saves model checkpoints containing the `state_dict` of *trainable* parameters and the optimizer state (e.g., via PyTorch Lightning's `ModelCheckpoint`).
8.  **Inference Script:** A separate inference script (`rna_predict/predict.py`) exists for generating predictions from saved checkpoints.
9.  **Checkpoint Loading (Partial):** The inference script successfully loads the *trainable* parameters from a saved M2 checkpoint into the corresponding pipeline model structure using a partial loading mechanism (e.g., `partial_load_state_dict` adapted from `ProjectEquiSurv`), gracefully handling the absence of base model weights in the checkpoint.
10. **Initial Prediction Generation:** The inference script, using a loaded M2 checkpoint, successfully generates and saves 3D coordinate outputs (e.g., `.pdb` format) for a small, representative subset of the validation data without crashing.
11. **LanceDB Logging Stub:** A stub LanceDB logger is present in the codebase (`rna_predict/utils/lance_logger.py`), with no-op logging methods and a config flag (`PipelineConfig.lance_db.enabled`) controlling activation. For M2, this logger is a placeholder; real logging will be implemented in M3.

**4. Key Technical Tasks & Recommended Approaches**

**Confirmations Needed Before Starting:**

*   [ ] **Data Paths:** Confirm locations for raw Kaggle CSVs, precomputed Stage A adjacency files (specify format/naming), and ground truth coordinate data.
*   [ ] **Train/Val Split:** Confirm the method/file defining the train/validation split indices.
*   [ ] **LoRA Configuration:** Confirm the location (`conf/lora/*.yaml`?) and structure of LoRA configs (`r`, `alpha`, `target_modules`).
*   [ ] **LoRA Wrapping:** Confirm the implementation used to wrap base models with LoRA adapters is functional.
*   [ ] **Parameter Freezing:** Confirm the mechanism used to freeze base model parameters is active.
*   [ ] **Loss Function:** Confirm the specific 3D coordinate loss function (L1, MSE, FAPE?) and its location (`rna_predict/core/losses.py`?).
*   [ ] **Merger Status:** Confirm the `UnifiedLatentMerger` implementation is stable and handles expected inputs.
*   [ ] **Dependencies:** Confirm PyTorch Lightning, Hydra, and PEFT (or custom LoRA library) are in `requirements-dev.txt`.
*   [ ] **Patching:** Confirm `apply_tensor_fixes` is being called at the start of `train.py` and `predict.py`.

**Implementation Tasks:**

**4.1. Implement Training Framework (Leverage PyTorch Lightning)**

*   **Task:** Create `rna_predict/training/rna_lightning_module.py` defining `RNALightningModule`.
*   **Details:**
    *   `__init__`: Instantiate the full pipeline model components based on Hydra config (`cfg`). Store components (predictors, merger, manager) as attributes. Store loss weights.
    *   `forward`: Define the core forward pass logic (potentially calling `run_full_pipeline`).
    *   `training_step`: Call `forward`, compute loss(es), log `train/loss`.
    *   `validation_step`: Call `forward` with `torch.no_grad()`, compute loss(es), compute basic coordinate MAE, log `val/loss`, `val/mae`.
    *   `configure_optimizers`: Instantiate optimizer (e.g., AdamW from `cfg.training.optimizer`) targeting *only* trainable parameters (Task 4.3). Optionally build LR scheduler (using `build_scheduler` pattern from `ProjectEquiSurv`).
    *   `load_state_dict`: Override to use `partial_load_state_dict` (Task 4.4).
    *   Integrate config flags for LanceDB logging (stub for M2, see `PipelineConfig.lance_db`).
*   **Reference (`ProjectEquiSurv`):** `lightning_survival_module.py`, `optim_wrappers.py`.

**4.2. Establish Configuration Management (Leverage Hydra)**

*   **Task:** Set up `rna_predict/conf/` directory and schema.
*   **Details:**
    *   Create `rna_predict/conf/config_schema.py` with dataclasses (`DataConfig`, `ModelConfig`, `LoRAConfig`, `TrainingConfig`, `DiffusionConfig`, `PipelineConfig`, `RootConfig`).
    *   Create default `.yaml` files in `rna_predict/conf/` (e.g., `train.yaml`, `model/default.yaml`, `lora/torsionbert.yaml`).
    *   Use `@hydra.main` in `train.py` and `predict.py`.
*   **Reference (`ProjectEquiSurv`):** `conf/config_schema.py`, `pipeline/main.py`.

**4.3. Implement Trainable Parameter Filtering**

*   **Task:** Implement a function `get_trainable_parameters(model: torch.nn.Module)` -> Iterator[torch.nn.Parameter].
*   **Details:** Place this in `rna_predict/core/utils.py` or within `RNALightningModule`. It should iterate `model.named_parameters()` and yield `p` if `lora` is in `name` OR if `p` belongs to the Merger/Diffusion head (identify by module name/path) AND `p.requires_grad` is True.
*   **Verification:** Log the count/percentage of trainable parameters in `RNALightningModule.__init__`.

**4.4. Implement Partial Checkpoint Loading**

*   **Task:** Adapt `partial_load_state_dict` for use in `rna_predict`.
*   **Details:** Copy the function logic into `rna_predict/core/checkpointing.py` or `utils.py`. Ensure it handles loading a `state_dict` containing only LoRA/Merger/Diffusion keys into the full pipeline model structure without raising `strict=True` errors. Call this from `RNALightningModule.load_state_dict` and `predict.py`.
*   **Reference (`ProjectEquiSurv`):** `hypercloning/checkpoint_utils.py`.

**4.5. Build Data Handling Pipeline**

*   **Task:** Finalize `RNADataset` and `DataLoader` setup.
*   **Details:**
    *   Ensure `RNADataset` (`data/loader.py`) correctly loads sequence, adjacency (e.g., from `data/processed/adjacency/{seq_id}.pt`), and ground truth coords (e.g., from `data/processed/coords/{seq_id}.pt`). Return a dictionary.
    *   Implement splitting logic in `train.py` (e.g., using `torch.utils.data.random_split` or loading pre-defined indices).
    *   Create `DataLoader`s. Implement a `collate_fn` if needed for padding batches of variable-length sequences/adjacencies/coords.
*   **Reference (`ProjectEquiSurv`):** `graph_survival_dataset.py`, `split_manager.py`.

**4.6. Implement Training Script (`train.py`)**

*   **Task:** Create the main training script `rna_predict/train.py`.
*   **Details:**
    *   Use `@hydra.main(config_path="conf", config_name="train")`.
    *   Initialize logging (WandB/TensorBoard via Lightning loggers).
    *   Call `apply_tensor_fixes()`.
    *   Instantiate `RNALightningModule(cfg)`.
    *   Setup `DataLoader`s.
    *   Setup `ModelCheckpoint` callback (monitor `val/loss` or `val/mae`).
    *   Instantiate `Trainer` (pass `logger`, `callbacks`, `max_epochs` from `cfg`).
    *   Call `trainer.fit(model, train_dataloader, val_dataloader)`.
*   **Reference (`ProjectEquiSurv`):** `pipeline/pipeline_train_single.py`.

**4.7. Implement Inference Script (`predict.py`)**

*   **Task:** Create the inference script `rna_predict/predict.py`.
*   **Details:**
    *   Use `@hydra.main`.
    *   Call `apply_tensor_fixes()`.
    *   Instantiate the model structure (e.g., `RNALightningModule(cfg)` or directly the pipeline components).
    *   Load checkpoint using `torch.load` and `partial_load_state_dict`.
    *   Set `model.eval()`.
    *   Load list of sequences to predict (e.g., from `cfg.predict.input_file`).
    *   Create a simple `DataLoader` or loop through sequences.
    *   For each sequence: prepare features (adjacency), run inference via `model.forward` or `run_full_pipeline` with `torch.no_grad()`, potentially generating multiple samples (`N_sample` in diffusion).
    *   Save `final_coords` to `.pdb` (using BioPython) or `.pt` files in `cfg.predict.output_dir`.
*   **Reference (`ProjectEquiSurv`):** `kaggle_submission/ensemble_inference.py`, `inference_utils.py`.

**5. Optional Enhancements for M2**

*   **Logging:** Integrate `wandb` logger with PyTorch Lightning `Trainer`.
*   **Gradient Clipping:** Add `gradient_clip_val: 1.0` to `conf/train.yaml` and pass to `Trainer`.
*   **Qualitative Check:** Add a step in `train.py`'s `on_validation_epoch_end` hook (or manually run `predict.py`) to save one prediction PDB and visually inspect it.

**6. Definition of Done (M2)**

M2 is complete when all 11 "Must-Have" requirements listed in Section 3 are met and verified. The integrated pipeline is demonstrably trainable using LoRA, and initial 3D coordinate predictions can be generated from saved checkpoints for validation data.

**7. Next Steps (Post-M2)**

M3 will focus on **Validation & Debugging**: Implementing quantitative metrics (TM-score), evaluating M2 predictions, debugging geometry/numerical issues, and performing initial hyperparameter tuning based on validation results.

---
Okay, let's define specific, actionable code quality and testing goals for Milestone 2 (M2), tailored for a senior developer who has successfully completed M1. These goals align with M2's primary objective: achieving functional end-to-end trainability and initial output generation.

**Context:** M1 established the integrated pipeline structure and module implementations (potentially with stubs or basic functionality). M2 focuses on making this integrated system *runnable* for training and inference, particularly validating LoRA, gradient flow, and the core mechanics.

---

**M2 Code Quality Goals**

The focus is on **functional correctness, clear integration points, and maintainable infrastructure setup**, rather than perfect optimization or deep refactoring at this stage.

1.  **Functional Correctness of New M2 Components:**
    *   **Goal:** Ensure the primary scripts and modules introduced or significantly modified for M2 (`train.py`, `predict.py`, `RNALightningModule`, `RNADataset`, partial checkpoint loader) execute their core functions without runtime errors (crashes, type errors, major shape mismatches not handled by patching).
    *   **Metric:** Successful completion of a minimal training run (e.g., 1 epoch or 100 steps) and a minimal inference run (e.g., predicting 1-2 validation samples).

2.  **Readability and Clarity of Integration Logic:**
    *   **Goal:** The code implementing the training loop (`RNALightningModule`, `train.py`), inference logic (`predict.py`), and data loading (`RNADataset`) should be clearly structured and understandable. Key integration points (e.g., how data flows into `run_full_pipeline`, how loss is computed, how trainable parameters are selected) should be evident.
    *   **Metric:** Peer code review confirms understandability. Docstrings for major classes/functions (`RNALightningModule`, `train_single_run` equivalent, `predict.py` main function) are present and explain purpose/flow.

3.  **Configuration-Driven Execution:**
    *   **Goal:** All critical hyperparameters (learning rate, batch size, LoRA ranks/targets, loss weights, file paths, diffusion steps, etc.) should be managed via the Hydra configuration system (`conf/` directory), not hardcoded.
    *   **Metric:** The `train.py` and `predict.py` scripts primarily rely on the `cfg: DictConfig` object for settings. Manual inspection confirms no major hardcoded parameters remain.

4.  **Correct LoRA/Trainable Parameter Handling:**
    *   **Goal:** The mechanism for identifying and isolating trainable parameters (LoRA adapters, Merger, Diffusion head) for the optimizer must be implemented correctly and demonstrably work. Base model parameters must remain frozen.
    *   **Metric:** Code inspection confirms filtering logic. Logging/debugging output confirms only the intended parameters receive gradients and updates during a test `optimizer.step()`.

5.  **Robust Checkpoint Save/Load (Partial):**
    *   **Goal:** Checkpoints saved during training must contain only the necessary trainable weights and optimizer state. The inference script must successfully load these partial checkpoints into the full model structure.
    *   **Metric:** Successful execution of Requirement #9 and #10 (loading checkpoint and generating inference output). Manual inspection of a saved `.pt` file can confirm it doesn't contain the full base model weights (should be significantly smaller).

6.  **Explicit Patching:**
    *   **Goal:** Runtime tensor shape patches should be applied explicitly and controllably, ideally via the central `patching/shape_fixes.py` module.
    *   **Metric:** Patches are applied via a clear function call (e.g., `apply_tensor_fixes()`) at the start of `train.py` and `predict.py`.

**Out of Scope for M2 Code Quality:**

*   Deep refactoring of M1 modules (unless required to fix critical M2 blockers).
*   Extensive performance optimization (micro-optimizations, kernel fusion).
*   Full PEP 8 compliance across the entire codebase (focus on *new* M2 code).
*   Elimination of the *need* for runtime patching (this is a later refactoring goal).

---

**M2 Testing Goals**

The focus is on **integration testing** to verify the core mechanics of the end-to-end training and inference workflows. Unit testing is secondary and targeted at critical new components.

1.  **Core Training Loop Integration Test:**
    *   **Goal:** Verify that the training script (`train.py` using `RNALightningModule`) can successfully execute a small number of steps (e.g., 2-5 batches) on realistic (or simplified real) data without crashing.
    *   **Implementation:** A pytest test (e.g., `tests/pipeline/test_train_integration.py`) that:
        *   Sets up a minimal Hydra config pointing to small data subsets/adjacencies.
        *   Instantiates the `RNALightningModule` and a Lightning `Trainer` with `fast_dev_run=True` or `max_steps=5`.
        *   Calls `trainer.fit()`.
    *   **Assertions:** The test passes if `trainer.fit()` completes without exceptions. Check that the logged training loss is not NaN/Inf.

2.  **Gradient Flow and Parameter Update Test:**
    *   **Goal:** Confirm that backpropagation works and *only* updates the intended trainable parameters (LoRA, etc.).
    *   **Implementation:** Extend the training loop integration test (or create a separate one):
        *   Before the first `optimizer.step()`, store the initial values (or norms) of a sample of trainable parameters and a sample of frozen parameters.
        *   After `optimizer.step()`, assert that the trainable parameters *have changed* and the frozen parameters *have not changed*.
        *   Assert that `param.grad` is non-None for trainable params and None (or all zeros) for frozen params after `loss.backward()`.
    *   **Metric:** Test passes assertions.

3.  **Checkpoint Save/Load/Inference Cycle Test:**
    *   **Goal:** Verify the critical cycle of saving trainable weights during training and loading them for inference works correctly.
    *   **Implementation:** A pytest test (e.g., `tests/pipeline/test_checkpoint_inference_cycle.py`) that:
        1.  Runs the training loop test (Task 1) for a few steps, ensuring a checkpoint is saved by `ModelCheckpoint`.
        2.  Instantiates the inference pipeline model structure.
        3.  Loads the saved checkpoint using the `partial_load_state_dict` logic.
        4.  Runs the inference logic (`predict.py`'s core steps) on a single test sequence using the loaded model.
    *   **Assertions:** Loading the partial checkpoint succeeds. Inference runs without crashing. The output coordinate tensor has the expected shape and contains numeric values (not all zeros, no NaNs/Infs).

4.  **Unit Test: `partial_load_state_dict`:**
    *   **Goal:** Ensure the partial checkpoint loading utility functions correctly in isolation.
    *   **Implementation:** Unit tests in `tests/core/test_checkpointing.py` using dummy `nn.Module`s and `state_dict`s (one full, one partial) to verify correct loading behavior and error handling.

5.  **Unit Test: `RNADataset` / `collate_fn`:**
    *   **Goal:** Verify the dataset loading and batch collation.
    *   **Implementation:** Unit tests in `tests/data/test_loader.py` that:
        *   Instantiate `RNADataset` with mock file paths.
        *   Check that `__getitem__` returns a dictionary with the expected keys and tensor shapes/types for a single sample.
        *   Test the `collate_fn` (if custom) to ensure it correctly batches samples, potentially including padding.

**Out of Scope for M2 Testing:**

*   Unit tests for *all* individual modules implemented in M1 (assume basic tests exist or focus is on integration).
*   `hypothesis`-based fuzzing for most components (can be added in M3+).
*   Quantitative accuracy tests (TM-score, precise coordinate RMSD).
*   Testing edge cases in data loading (corrupted files, very long sequences beyond padding limits).
*   Mocking external dependencies like Hugging Face Hub downloads (assume models are pre-downloaded or rely on caching).

---

**Summary for Senior Developer:**

M2 requires building and validating the core training and inference infrastructure around the integrated pipeline from M1. Code quality focuses on functional correctness and clear configuration. Testing priorities are **integration tests** verifying the train loop, gradient flow to LoRA parameters, and the checkpoint save/load/predict cycle. Unit tests should cover critical new M2 components like partial checkpoint loading and data collation. Achieving these goals ensures we have a mechanically sound system ready for quantitative validation and iterative refinement in M3. Please confirm the prerequisite checks listed in the previous document section before proceeding with these tasks.

# Full-Pipeline Partial Checkpoint Test for RNA_PREDICT

To fully verify partial checkpointing in RNA_PREDICT, we need an end-to-end test using the real model, real configuration (via Hydra), and the LightningModule interface. This will ensure robust, production-like coverage beyond dummy/unit tests.

---

## Enhanced Change Plan Summary (Additions + Clarifications)

### RNALightningModule Upgrade (Step 1)

1. **Modular construction**:
   - Add a helper function (`build_pipeline(cfg)`) that creates the full model based on Hydra config and returns the initialized object.
   - This function should live in `rna_predict/pipeline/build_pipeline.py`.

2. **Trainable params filter**:
   - Add `get_trainable_params(model: nn.Module)` in `rna_predict/core/utils.py` (or within the LightningModule).
   - Filters by `requires_grad=True` and optionally by substring match like `'lora'`, `'merger'`, `'diffusion'`.

3. **Checkpoint-friendly naming**:
   - Ensure all submodules (TorsionBERT, Pairformer, Merger, DiffusionManager) are attributes of the top-level module and their names are stable.
   - This helps `state_dict` consistency and partial load granularity.

---

### Full-Pipeline Integration Test (Step 2)

**Filename:** `tests/integration/test_partial_checkpoint_full_pipeline.py`

**Expanded Steps**:

- **Step 3.5:** Add validation that the saved partial state dict has no unexpected base model keys (e.g., assert no `bert.encoder.layer.0` if LoRA-only checkpoint).
- **Step 6.5:** Assert that all `requires_grad=True` parameters changed after `optimizer.step()`; others did not.
- **Step 8 (Comparison):**
   ```python
   partial_ckpt_size = os.path.getsize(partial_ckpt_path)
   full_ckpt_size = os.path.getsize(full_ckpt_path)
   assert partial_ckpt_size < full_ckpt_size
   ```

---

### Utility Enhancements (Step 3)

- `save_trainable_checkpoint(model: nn.Module, path: str)`:
   - Filters `requires_grad=True` parameters and saves them.
   - Lives in `rna_predict/utils/checkpointing.py`.

- `partial_load_state_dict(model: nn.Module, state_dict: Dict[str, Any], strict=False)`:
   - Already exists in your earlier planning—just ensure it's recursively safe for all submodules in `LightningModule`.

---

### Documentation (Step 4)

**Suggested Locations**:
- Add inline docstrings in:
   - `RNALightningModule.training_step`, `forward`, `configure_optimizers`
   - `test_partial_checkpoint_full_pipeline.py` — explain each test phase
- Add Markdown explanation to `docs/guides/testing/partial_checkpoint.md`:
   - Include diagram: Full Model → Save LoRA-Only → Reload → Inference Pass

---

## Add-On: Safety Checks (Optional but Strongly Recommended)

- Run `model.eval()` before checkpoint comparison (ensure dropout doesn’t introduce noise).
- Add sanity check:
   ```python
   assert not torch.isnan(output).any()
   assert not torch.isinf(output).any()
   ```

---

## Systematic Change Plan (Original)

### 1. Upgrade RNALightningModule
- **Replace dummy parameter** with the actual pipeline/model as constructed in the main pipeline.
- **Support Hydra config:** Accept and correctly use Hydra config for model construction and device handling.
- **Implement real forward:** Forward should run the actual pipeline logic, using realistic input shapes/types.
- **Ensure compatibility:** Confirm all methods (training_step, configure_optimizers, etc.) are compatible with the real model.

### 2. Create Full-Pipeline Integration Test
- **Location:** `tests/integration/test_partial_checkpoint_full_pipeline.py`
- **Steps:**
  1. **Hydra Initialization:** Use the correct config path (`/Users/tomriddle1/RNA_PREDICT/rna_predict/conf`).
  2. **Instantiate real model:** Use Hydra config to instantiate the upgraded LightningModule wrapping the real pipeline.
  3. **Dummy data:** Use a small batch of dummy input matching the real pipeline’s expected input.
  4. **Train for a few steps:** Use PyTorch Lightning’s Trainer for a minimal number of steps.
  5. **Save partial checkpoint:** Save only LoRA/new module parameters or those with `requires_grad=True`.
  6. **Reload checkpoint:** Instantiate a fresh model, load the partial checkpoint with `partial_load_state_dict`.
  7. **Inference:** Run a forward pass, verify correct output shape/type and no errors.
  8. **Compare checkpoint sizes:** Assert the partial checkpoint is smaller than a full checkpoint.
  9. **Assertions:**
     - No key/shape errors on load.
     - Forward pass works.
     - Output shapes/types are as expected.
     - Partial checkpoint is smaller than full checkpoint.

### 3. Refactor Utilities as Needed
- **partial_load_state_dict:** Ensure it works for the real model and LightningModule, including nested modules.
- **Saving logic:** Add or refactor utility to extract only parameters of interest (LoRA, adapters, etc.).

### 4. Documentation and Acceptance Criteria
- **Document test logic and rationale** in the new test file and in this plan.
- **Acceptance Criteria:**
  - No key/shape errors on load.
  - Forward pass works after partial load.
  - Partial checkpoint is smaller than full checkpoint.

---

## Review Checklist
- [X] RNALightningModule upgraded to wrap real pipeline
- [X] Full-pipeline integration test created
- [ ] Utilities verified/refactored as needed
- [ ] All acceptance criteria covered by assertions
- [ ] Documentation updated

---

*Prepared for systematic review before implementation.*

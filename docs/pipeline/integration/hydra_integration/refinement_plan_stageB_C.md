# Hydra Integration Refinement Plan (Stages B & C) - 2025-04-13

This document outlines the approved plan to refine the Hydra integration for Stages B and C, ensuring alignment between documentation, configuration, and code.

## Analysis Summary

* **Stage B:** Implementation uses two config files (`rna_predict/conf/model/stageB_torsion.yaml`, `rna_predict/conf/model/stageB_pairformer.yaml`), while documentation (`StageB_Torsion_Pairwise.md`) showed one nested file. Code integration is functionally correct. User prefers the implemented two-file structure.
* **Stage C:** Implemented `rna_predict/conf/model/stageC.yaml` included a `defaults:` list referring to a nested model config (`mp_nerf_model`) with an incorrect path or missing target file, causing a loading error. The overall structure also deviated slightly from the example in `StageC_3D_Reconstruction.md`. Code integration is functionally correct.

## Approved Refinement Plan

1.  **Update Stage B Documentation:**
    * **File:** `docs/pipeline/integration/hydra_integration/components/stageB/StageB_Torsion_Pairwise.md`
    * **Action:** Modify YAML examples and explanatory text to accurately describe the implemented two-file structure (`stageB_torsion.yaml`, `stageB_pairformer.yaml`) and their inclusion in `default.yaml`. *(This requires only documentation edits)*.
    * **Reason:** Align documentation with the approved and working implementation for clarity.

2.  **Refine Stage C Configuration:**
    * **File:** `rna_predict/conf/model/stageC.yaml`
    * **Action:** Overwrite the file content to:
        * Remove the `defaults:` list section entirely.
        * Ensure parameters (`method`, `device`, `do_ring_closure`, `place_bases`, `sugar_pucker`, memory flags like `use_memory_efficient_kernel`) are directly under the top-level `stageC:` key, matching the structure in `StageC_3D_Reconstruction.md`.
        * Set the default for `device` to `"auto"`.
    * **Reason:** Resolve the config loading error and align `stageC.yaml` structure with its specific design documentation. Selection of underlying `mp_nerf_model` variants (if needed) will depend on the main `default.yaml` or CLI overrides.

## Implementation Notes

* No Python code changes are planned for Stages B or C as part of *this* refinement plan.
* The existence and structure of `rna_predict/conf/mp_nerf_model/` and its files (e.g., `default_rna.yaml`) are currently outside the scope of this specific refinement but may need addressing separately for full Stage C functionality with different model variants.
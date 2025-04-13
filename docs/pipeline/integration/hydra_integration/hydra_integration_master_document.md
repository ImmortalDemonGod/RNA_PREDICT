# **Hydra Integration Master Document: A Comprehensive Implementation Plan for RNA_PREDICT**

This document merges the **best features** from all previous versions (V1–V5) to create a **definitive, in-depth guide** for integrating [**Hydra**](https://github.com/facebookresearch/hydra) into the RNA_PREDICT pipeline. It balances clarity, structure, code examples, acceptance criteria, and references to synergy across all pipeline stages (A–D), plus LoRA, optional energy minimization, and memory optimization—ultimately **better than the sum of its parts**.

---

## **1. Introduction & Rationale**

### **1.1 Why Use Hydra?**

1. **Centralized Configuration**  
   - Move from scattered constants/hardcoded defaults to YAML config files, ensuring a single source of truth for hyperparameters, paths, LoRA adapter settings, toggles, and more.
2. **Ease of Experimentation**  
   - Override any parameter from the command line (e.g., `stageB.torsion_bert.lora.enabled=true`) without rewriting code. This is crucial for advanced pipelines with many parameters.
3. **Reproducibility & Modularity**  
   - YAML files can be committed to version control, letting you revisit exactly how each run was configured. Hydra also supports composition, so each stage's config remains modular.
4. **Backward Compatibility**  
   - By setting YAML defaults to match your current inline constants, the pipeline's behavior remains unchanged unless you override parameters—preventing disruption of existing workflows.

### **1.2 Pipeline Context**

The RNA_PREDICT pipeline is composed of multiple stages:
- **Stage A**: 2D adjacency prediction (e.g., via RFold).
- **Stage B**: TorsionBERT (with optional LoRA) & Pairformer (also optional LoRA) for angles and pairwise embeddings.
- **Stage C**: 3D reconstruction (MP-NeRF or fallback).
- **Unified Latent Merger**: Combines adjacency, angles, partial coords, embeddings into a single representation.
- **Stage D**: Diffusion-based refinement (with optional LoRA) and optional energy minimization.

Hydra must manage a range of parameters—model hyperparams, file paths, devices, synergy between the shape/dimensions of different stages, etc.

---

## **2. Scope & Acceptance Criteria**

### **2.1 Scope**

This plan addresses:

1. **Installing & Pinning Hydra Dependencies**  
   - Ensuring both `hydra-core` and `omegaconf` are part of the environment.
2. **Creating a `conf/` Directory Structure**  
   - Organizing YAML files by stage/component.
3. **Defining a Configuration Schema**  
   - Using Python `@dataclass` or similar for typed configs (optional but recommended).
4. **Refactoring Each Stage (A–D)**  
   - Replacing inline defaults with Hydra-based config references.
5. **Ensuring Backward Compatibility**  
   - Matching current defaults so the pipeline's existing behavior is preserved.
6. **Optional Features**  
   - Incorporating LoRA toggles, memory optimization, and energy minimization config sections.
7. **Documentation & Team Onboarding**  
   - Educating developers on Hydra usage, command-line overrides, and custom YAML composition.

### **2.2 Out of Scope**

- **Deep architectural refactoring** beyond reading Hydra configs.
- **Extensive hyperparameter tuning** (the plan focuses on exposing parameters, not optimizing them).
- **Replacing or rewriting the entire code logic**; we only adapt it to use Hydra as a centralized config manager.

### **2.3 Acceptance Criteria**

The integration is complete when:

1. **Hydra & OmegaConf** are installed and pinned in `requirements.txt/pyproject.toml`.
2. The **directory `rna_predict/conf/`** exists with default YAMLs for each major pipeline stage (A–D) and any optional modules (LoRA, data, etc.).
3. **Python dataclasses** (or an equivalent schema) in `config_schema.py` reflect each stage's parameters, providing typed defaults matching existing code.
4. **All major code** (like `run_stageA.py`, TorsionBertPredictor, `stage_c_reconstruction.py`, `run_stageD.py`) reads configuration values from Hydra `cfg` instead of inline constants.
5. **Backward-compatible results**: Running the pipeline with default YAMLs produces outputs consistent with the previous (pre-Hydra) pipeline.
6. **CLI Override**: A developer can override parameters (e.g., `python main.py stageA.num_hidden=256`) and see changes reflected in the pipeline.
7. **Documentation**: A short "Hydra Usage Guide" is added, ensuring any new or existing developer can adopt the new config system.

---

## **3. Step-by-Step Implementation Plan**

### **3.1 Dependencies & Environment**

1. **Add Hydra to Project**  
   - In `requirements.txt` or `pyproject.toml`:
     ```bash
     hydra-core==1.3.2
     omegaconf==2.3.0
     ```
   - Pin these versions to avoid unexpected compatibility issues.
2. **Update Containers**  
   - If using Docker or any container system, ensure you `RUN pip install hydra-core==1.3.2 omegaconf==2.3.0`.
3. **Verify Installation**  
   - Create a minimal test script:
     ```python
     import hydra
     from omegaconf import DictConfig

     @hydra.main(version_base=None, config_path=None)
     def demo(cfg: DictConfig):
         print("Hydra test:", cfg)

     if __name__ == "__main__":
         demo()
     ```
   - Run `python demo.py` to confirm Hydra loads without errors.

---

### **3.2 Directory & File Structure**

Create a `conf/` directory inside `rna_predict/`:

```
rna_predict/
  ├── conf/
  │   ├── config_schema.py        # Python dataclasses for typed config (optional but recommended)
  │   ├── default.yaml            # The top-level config referencing sub-configs
  │   ├── model/
  │   │   ├── stageA.yaml
  │   │   ├── stageB_torsion.yaml
  │   │   ├── stageB_pairformer.yaml
  │   │   ├── stageC.yaml
  │   │   └── stageD_diffusion.yaml
  │   ├── memory_optimization.yaml (optional)
  │   ├── energy_minimization.yaml (optional)
  │   ├── train.yaml (optional)
  │   └── inference.yaml (optional)
  └── ...
```

- **Why subdirectories?**: Hydra supports "composable configs," letting you pick `model/stageA.yaml` or override it with `model/stageA_alt.yaml`.  
- Keep related parameters together (e.g., all TorsionBERT settings in one file).

---

### **3.3 Defining the Configuration Schema**

In `config_schema.py`, define Python dataclasses that match each stage's config. This ensures:

- **Typed defaults**: Minimizes confusion about parameter types.  
- **Validation**: If the YAML is missing or has the wrong type, Hydra can warn or fail early.

**Example**:

```python
# rna_predict/conf/config_schema.py
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class StageAConfig:
    num_hidden: int = 128
    dropout: float = 0.3
    min_length: int = 80
    checkpoint_path: str = "RFold/checkpoints/RNAStralign_trainset_pretrained.pth"
    device: str = "cuda"
    # Possibly additional parameters like binary_threshold, argmax_mode, etc.

@dataclass
class LoraConfig:
    enabled: bool = False
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: Optional[List[str]] = None

@dataclass
class TorsionBertConfig:
    model_name_or_path: str = "sayby/rna_torsionbert"
    device: str = "cuda"
    angle_mode: str = "degrees"
    num_angles: int = 7
    max_length: int = 512
    checkpoint_path: Optional[str] = None
    lora: LoraConfig = LoraConfig()

@dataclass
class PairformerConfig:
    n_blocks: int = 2
    c_z: int = 32
    c_s: int = 64
    dropout: float = 0.1
    use_checkpoint: bool = False
    init_z_from_adjacency: bool = True
    lora: LoraConfig = LoraConfig()

@dataclass
class StageCConfig:
    method: str = "mp_nerf"
    do_ring_closure: bool = False
    place_bases: bool = True
    sugar_pucker: str = "C3'-endo"
    device: str = "auto"

@dataclass
class LatentMergerConfig:
    method: str = "MLP"
    dim_angles: int = 7
    dim_s: int = 64
    dim_z: int = 32
    dim_out: int = 128
    hidden_sizes: List[int] = field(default_factory=lambda: [128])
    dropout: float = 0.1
    activation: str = "ReLU"
    freeze: bool = False

@dataclass
class MemoryOptimizationConfig:
    enable: bool = True

@dataclass
class EnergyMinimizationConfig:
    enabled: bool = False
    steps: int = 1000
    method: str = "OpenMM"

@dataclass
class StageDConfig:
    mode: str = "inference"  # "inference" or "training"
    device: str = "cuda"
    sigma_data: float = 16.0
    gamma0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5
    n_steps: int = 50
    c_atom: int = 128
    c_atompair: int = 16
    c_token: int = 768
    lora: LoraConfig = LoraConfig()

@dataclass
class RNAConfig:
    stageA: StageAConfig = StageAConfig()
    torsion_bert: TorsionBertConfig = TorsionBertConfig()
    pairformer: PairformerConfig = PairformerConfig()
    stageC: StageCConfig = StageCConfig()
    latent_merger: LatentMergerConfig = LatentMergerConfig()
    stageD: StageDConfig = StageDConfig()
    memory_optimization: MemoryOptimizationConfig = MemoryOptimizationConfig()
    energy_minimization: EnergyMinimizationConfig = EnergyMinimizationConfig()
```

---

### **3.4 Writing YAML Files per Stage**

Example: `stageA.yaml`

```yaml
stageA:
  num_hidden: 128
  dropout: 0.3
  min_length: 80
  checkpoint_path: "RFold/checkpoints/RNAStralign_trainset_pretrained.pth"
  device: "cuda"
```

Example: `stageB_torsion.yaml` for TorsionBERT:

```yaml
torsion_bert:
  model_name_or_path: "sayby/rna_torsionbert"
  device: "cuda"
  angle_mode: "degrees"
  num_angles: 7
  max_length: 512
  checkpoint_path: null
  lora:
    enabled: false
    r: 8
    alpha: 16
    dropout: 0.1
    target_modules: ["attention.query", "attention.key"]
```

Create similar for Pairformer (`stageB_pairformer.yaml`), Stage C (`stageC.yaml`), and Stage D (which can also hold memory & minimization config if you like).

Finally, reference them in your `default.yaml`:

```yaml
defaults:
  - model/stageA
  - model/stageB_torsion
  - model/stageB_pairformer
  - model/stageC
  - model/stageD_diffusion
```

---

### **3.5 Code Integration**

**Core Changes**:  
1. **Add Hydra Decorator** to a main script (e.g., `rna_predict/main.py`):

```python
# rna_predict/main.py
import hydra
from omegaconf import DictConfig, OmegaConf
# If you want typed configs:
# from rna_predict.conf.config_schema import RNAConfig

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig):
    print("HYDRA CONFIG:\n", OmegaConf.to_yaml(cfg))

    # Stage A usage
    # from rna_predict.pipeline.stageA.adjacency.rfold_predictor import StageARFoldPredictor
    # predictor = StageARFoldPredictor(
    #     config={
    #       "num_hidden": cfg.stageA.num_hidden,
    #       "dropout": cfg.stageA.dropout,
    #       ...
    #     },
    #     checkpoint_path=cfg.stageA.checkpoint_path,
    #     device=cfg.stageA.device
    # )
    # ...
    # Similarly for TorsionBERT, Pairformer, etc.

if __name__ == "__main__":
    main()
```

2. **Replace Hardcoded Defaults** in each stage with config references:
   - In `StageBTorsionBertPredictor.__init__`, remove e.g. `self.num_angles = 7`, replace with `self.num_angles = cfg.num_angles`.
   - In `stage_c_reconstruction.py`, remove e.g. `method="mp_nerf"` defaults, read from `cfg.stageC.method`, `cfg.stageC.do_ring_closure`, etc.
   - In your diffusion code (`run_stageD.py`), reference `cfg.stageD.sigma_data`, `cfg.stageD.noise_scale`, etc.

3. **LoRA**:
   - If LoRA is relevant for TorsionBERT or Pairformer, pass `cfg.torsion_bert.lora` or `cfg.pairformer.lora` to your LoRA injection logic.

4. **Memory Optimization**:
   - If your code calls something like `apply_memory_fixes(diffusion_config)`, replace inline logic with reading from `cfg.memory_optimization.enable`.

5. **Energy Minimization**:
   - The final pipeline step can check `if cfg.energy_minimization.enabled:` and run the relevant routine (OpenMM, GROMACS, etc.) for a given `cfg.energy_minimization.method`.

---

### **3.6 Ensuring Backward Compatibility**

1. **Match All Defaults**:
   - For each stage's YAML, copy over the same numeric defaults from your old code (e.g., `num_hidden=128`).
2. **Gradual Migration**:
   - Start with Stage A or B, confirm minimal breakage, then proceed to Stage C, D, etc.
3. **Testing**:
   - Validate that running "vanilla pipeline" (no CLI overrides) yields the same adjacency outputs, angles, or final 3D coords as before.

---

### **3.7 Documentation & Team Onboarding**

1. **README or Wiki**:
   - Document how to run the pipeline with Hydra:
     ```bash
     python -m rna_predict.main stageA.num_hidden=256 stageB_torsion.lora.enabled=true
     ```
2. **Override YAML**:
   - Teach advanced usage, e.g. adding a second YAML file `model/stageB_torsion_experiment.yaml` for a bigger TorsionBERT model or a special LoRA rank.
3. **Team Communication**:
   - Announce the Hydra shift in Slack/Teams so everyone knows to rely on `conf/` rather than inline code constants.

---

### **3.8 Testing & Validation**

1. **Smoke Tests**:
   - Quick end-to-end tests on short synthetic sequences (e.g., "ACGUACGU").
   - Confirm no runtime errors and that each stage sees the right shapes/dimensions.
2. **Unit Tests**:
   - If you have `pytest`, add tests that load `default.yaml`, instantiate each stage's predictor, and confirm correct parameter usage.
3. **Integration Tests**:
   - Possibly adapt your existing `test_full_pipeline.py` or `test_main_integration.py` to ensure that the pipeline completes with Hydra config.
4. **Edge Cases**:
   - Toggling LoRA on/off, enabling memory optimization, different sugar puckers for Stage C, or flipping Stage D from "inference" to "training."

---

## **4. Conclusion & Next Steps**

### **4.1 Summary of Implementation**

By following this plan:
1. **Install Hydra** in your environment (pin the version).
2. **Create** a structured `conf/` folder with top-level `default.yaml` referencing stage-specific YAML files.
3. **Define** typed dataclasses (`config_schema.py`) for clarity and type safety (optional but strongly recommended).
4. **Refactor** each pipeline stage (A–D) to read from Hydra's `cfg` object, removing scattered hard-coded defaults.
5. **Maintain** backward-compatible defaults so existing runs remain stable.
6. **Document** usage in README or docs/hydra_integration.md, providing examples of CLI overrides for your teammates.

### **4.2 Potential Benefits**

- **Centralized Parameters**: 
  - No more searching through multiple .py files for relevant defaults.
- **Experimentation Made Simple**: 
  - Single command-line overrides for different LoRA ranks, diffusion steps, or ring closure toggles.
- **Scalable Approach**: 
  - As you add or refine pipeline stages (e.g., advanced multi-structure embeddings or new diffusion schedules), simply extend the YAML schema, avoiding code churn.

### **4.3 Future Enhancements**

- **Multi-run Sweeps**: 
  - Hydra supports `-m` for launching multiple runs (e.g., sweeping over different `num_hidden` or dropout rates).
- **Config-Driven Minimization**: 
  - More fine-grained control over energy minimization (like force fields, temperature).
- **Advanced Validation**: 
  - If you need to ensure certain relationships (e.g., `c_s` must match `dim_s` in the latent merger), you can write custom checks or use pydantic for advanced constraints.
  
---

> **With these steps, you'll achieve a robust, well-organized Hydra integration for RNA_PREDICT, combining**:
> - **Formality & acceptance criteria** (from earlier plans),
> - **Conciseness & stepwise clarity** (so developers know exactly what to do),
> - **Stage-based synergy** (ensuring adjacency from Stage A seamlessly links to Pairformer or TorsionBERT in Stage B, partial coords in Stage C, etc.),
> - **Clear references to LoRA, memory optimization, and optional energy minimization** (like in your current pipeline code).

Once you complete this final integrated plan, you'll have a flexible, maintainable, and easily extendable configuration system—**truly better than the sum of its parts** from previous iterations. Good luck with the Hydra integration, and happy coding! 
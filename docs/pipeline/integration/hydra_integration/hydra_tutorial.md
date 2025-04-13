🧙🏾‍♂️: Aligning to your goal: You need a self-contained reference document that guides a junior developer to incorporate Hydra into an ML project, using patterns drawn from the "ProjectEquiSurv" code base.

[emoji]: Below is a comprehensive tutorial-style guide. A junior dev can follow it step-by-step to set up Hydra for configuration management, referencing real code snippets from ProjectEquiSurv.

---

# Hydra Usage Guide with ML Project Patterns (Based on Project EquiSurv)

This document introduces the Hydra framework, a powerful Python library for composable configurations. We will walk through:

1. Why Hydra?
2. Project Structure & Key Files
3. Basic Hydra Concepts
4. Setting Up Hydra
5. Config Directory & YAML Organization
6. Composing Configs
7. Runtime Overrides & CLI Examples
8. Using Hydra for HPC vs. Local
9. (Optional) Hydra Sweeping vs. Custom Search
10. FAQ & Common Pitfalls

Throughout, we'll reference patterns from the ProjectEquiSurv code—particularly the quick_fixes/conf/ directory and usage in pipeline/main.py.

---

## 1. Why Hydra?

Machine learning projects often require multiple environment-dependent configurations:

- Different hyperparameters for training
- Different dataset paths, logging settings, HPC vs. local
- Variation in experiment modes (e.g. "train", "eval", "hyperparam sweep")

Hydra standardizes how these configurations are defined, merged, and overridden via command-line. Instead of large ad-hoc config Python scripts or a single giant YAML, Hydra encourages modular sub-configs that Hydra merges at runtime, letting you override any field on the command-line if desired.

---

## 2. Project Structure & Key Files

Below is an example structure (mirroring how ProjectEquiSurv organizes quick_fixes/conf), focusing on the Hydra config pieces:

```
my_ml_project/
├── conf
│   ├── config.yaml               # The main or "top-level" config
│   ├── model
│   │   └── default.yaml          # Model hyperparams
│   ├── dataset
│   │   └── default.yaml          # Dataset references, data paths
│   ├── experiment
│   │   ├── local.yaml            # Local dev environment overrides
│   │   └── hpc.yaml              # HPC environment overrides
│   └── search_space
│       └── default.yaml          # Hyperparam search ranges
├── pipeline
│   ├── main.py                   # Hydra entrypoint (@hydra.main)
│   ├── pipeline_train_single.py  # Train logic referencing cfg
│   └── ... (other scripts)
└── ...
```

### Quick-Glance at Key Files

- conf/config.yaml: Declares top-level defaults and merges sub-configs:

```yaml
defaults:
  - model: default
  - dataset: default
  - experiment: local
  - search_space: default
  - _self_

mode: "train"
some_global_setting: true

# Logging dir or run settings, e.g.:
hydra:
  run:
    dir: ./outputs  # or "outputs/${now:%Y-%m-%d_%H-%M-%S}"
```

- pipeline/main.py: The Python script that calls:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Hydra merges everything into cfg
    mode = cfg.mode
    if mode == "train":
        ...
    elif mode == "hyperparam_sweep":
        ...
    else:
        print(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
```

---

## 3. Basic Hydra Concepts

- Structured YAML: Hydra merges multiple YAMLs declared in the defaults list (or via overrides)
- DictConfig: The resulting object inside main.py. Access config fields with cfg.something
- Compositional configs: Each sub-config file can hold a portion of the overall settings (e.g. model vs. dataset)
- CLI Overrides: You can override any setting from the command-line, e.g. python main.py mode=train model.hidden_dim=512

---

## 4. Setting Up Hydra

1. Install Hydra:
   ```bash
   pip install hydra-core>=1.2  # Or match your environment
   ```

2. Create a conf/ directory
   Inside your project, place a root YAML, e.g. config.yaml.

3. In your main script:
   ```python
   import hydra
   from omegaconf import DictConfig

   @hydra.main(version_base=None, config_path="conf", config_name="config")
   def main(cfg: DictConfig):
       print(cfg)   # For debugging
       # rest of your code

   if __name__ == "__main__":
       main()
   ```

With that minimal approach, Hydra loads conf/config.yaml. Next, you add sub-configs to define specific settings.

---

## 5. Config Directory & YAML Organization

Following ProjectEquiSurv best practices:

- conf/config.yaml (top-level):
  ```yaml
  defaults:
    - model: default
    - dataset: default
    - experiment: local
    - search_space: default
    - _self_

  # Hydra or top-level fields
  mode: "train"
  device: "cpu"
  # (You can store pipeline settings, e.g. logs, output_dir, etc.)
  ```

- conf/model/default.yaml:
  ```yaml
  # model/default.yaml
  hidden_dim: 256
  num_bins: 10
  survival_mode: "cox"  # or "discrete"

  dropout: 0.1
  # Additional model hyperparams
  ```

- conf/dataset/default.yaml:
  ```yaml
  # dataset/default.yaml
  train_csv_path: "data/raw/train.csv"
  test_csv_path: "data/raw/test.csv"

  val_fraction: 0.2
  split_strategy: "random"
  ```

- conf/experiment/local.yaml:
  ```yaml
  # local.yaml
  epochs: 5
  train_batch_size: 8
  some_local_override: true
  ```

- conf/experiment/hpc.yaml (for HPC):
  ```yaml
  # hpc.yaml
  epochs: 50
  train_batch_size: 64
  some_local_override: false
  # HPC cluster settings
  ```

- conf/search_space/default.yaml (for hyperparam search definitions):
  ```yaml
  # search_space/default.yaml
  hyperparam_search:
    # e.g. define param ranges, or store them for a custom search routine
  ```

Key: Hydra sees the defaults list in conf/config.yaml and merges each sub-config. The final result is accessible as cfg.model.hidden_dim, cfg.dataset.train_csv_path, etc.

---

## 6. Composing Configs

Inside conf/config.yaml:

defaults:
  - model: default
  - dataset: default
  - experiment: local
  - search_space: default
  - _self_

- The order matters: Hydra merges them top to bottom. _self_ means "finally merge the contents of this file (config.yaml) last."

If you want to switch from experiment: local to experiment: hpc, you can do one of two approaches:
  1. Edit config.yaml:

defaults:
  - model: default
  - dataset: default
  - experiment: hpc  # swapped local => hpc
  ...

2. Command-line Override:

python main.py experiment=hpc

Hydra sees experiment=hpc and merges hpc.yaml instead of local.yaml.

---

## 7. Runtime Overrides & CLI Examples

A big advantage of Hydra is on-the-fly overrides. For example:

# 1) Switch environment
python main.py experiment=hpc

# 2) Override a single field
python main.py model.hidden_dim=512

# 3) Combine multiple overrides
python main.py mode=train dataset.split_strategy=race_time_stratified model.survival_mode=discrete

# 4) Overriding path for train_csv
python main.py dataset.train_csv_path="/mnt/large_train.csv"

All those changes apply without rewriting config YAMLs. Hydra merges them at runtime.

---

## 8. Using Hydra for HPC vs. Local

ProjectEquiSurv uses separate YAMLs for HPC and local:

- experiment/local.yaml:

epochs: 5
train_batch_size: 8
# ...

- experiment/hpc.yaml:

epochs: 50
train_batch_size: 64
# HPC-specific fields

When you specify experiment:hpc, Hydra merges the HPC settings. If your code references cfg.epochs or cfg.train_batch_size, it automatically picks the HPC values.

Additionally, in HPC contexts, you might define some_local_override: false, or cluster-based resource strings (like slurm scripts). This keeps your code environment-agnostic, because it always reads from cfg....

---

## 9. (Optional) Hydra Sweeping vs. Custom Search

In ProjectEquiSurv, the hyperparameter search is done with a custom Optuna approach. This is completely valid. Alternatively, Hydra has built-in sweepers that can loop through sets of config overrides in multiple runs. For instance:

python main.py -m model.hidden_dim=128,256,512 optimizer.lr=0.001,0.01

Hydra would produce 6 runs (3 × 2). If you prefer your custom approach, just keep it. You still store search ranges in, e.g., conf/search_space/default.yaml.

---

## 10. FAQ & Common Pitfalls

1. "My directory structure is different"
   - Hydra doesn't require an exact naming convention or "conf/" location. You can define @hydra.main(config_path="path/to/my_configs", config_name="base"). Just keep a subdirectory of YAMLs and be consistent.

2. "ValueError: Could not merge config"
   - Usually you have conflicting keys or typed fields. Double check that your sub-configs have consistent naming or mark items as optional.

3. Logging & Output Directory
   - By default, Hydra can create per-run subdirectories like ./outputs/2023-07-10_15-03-22 to isolate logs. If you prefer your own directory creation logic (like ProjectEquiSurv does in path_utils.py), that's also fine.

4. Common Patterns
   - Use cfg.mode to route to different pipelines (train, eval, etc.).
   - Keep environment overrides (like HPC, local, dev) in separate YAMLs.
   - Keep large code blocks out of the config: the config is for hyperparams, paths, toggles, not big scripts.

---

Example: Putting It All Together

Below is a minimal working snippet that echoes the style of ProjectEquiSurv:

1. Directory:

my_project/
├── conf
│   ├── config.yaml
│   ├── model
│   │   └── default.yaml
│   └── experiment
│       ├── local.yaml
│       └── hpc.yaml
└── main.py

2. conf/config.yaml:

defaults:
  - model: default
  - experiment: local
  - _self_

mode: "train"

3. conf/model/default.yaml:

hidden_dim: 256
survival_mode: "cox"

4. conf/experiment/local.yaml:

epochs: 5
train_batch_size: 8

5. conf/experiment/hpc.yaml:

epochs: 50
train_batch_size: 64

6. main.py:

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print("MODE:", cfg.mode)
    print("MODEL:", cfg.model)
    print("EPOCHS:", cfg.epochs)
    if cfg.mode == "train":
        run_training(cfg)
    else:
        print("No valid mode selected.")

def run_training(cfg):
    # Example usage:
    print(f"Training for {cfg.epochs} epochs with batch_size={cfg.train_batch_size}")
    # Access model config
    print(f"Hidden dim: {cfg.model.hidden_dim}, survival_mode: {cfg.model.survival_mode}")
    # ... do your training ...

if __name__ == "__main__":
    main()

Running it:

# Default local config
python main.py

# HPC override
python main.py experiment=hpc

# Additional override
python main.py experiment=hpc model.hidden_dim=512

---

Final Remarks

By adopting Hydra:
  1. You keep your Python code simpler—just read cfg.
  2. You scale easily to multiple environments or hyperparam sets.
  3. You can embed advanced Hydra features (e.g. plugin-based sweepers, custom search space merges, dynamic imports of partial configs, etc.).

For deeper reference, see official Hydra docs or the more advanced usage in ProjectEquiSurv's quick_fixes/advanced/pipeline/*.py scripts, where HPC, local, meta-splits, and more specialized sub-configs are managed elegantly.

---

🧙🏾‍♂️: Next Steps,
[emoji]: Now that you have a comprehensive Hydra setup guide referencing ProjectEquiSurv, try implementing your own conf/ structure, define sub-configs for your model and dataset, and integrate them in a main.py Hydra entry script. If you run into any questions, feel free to ask—what specific aspect of Hydra integration can we clarify next?
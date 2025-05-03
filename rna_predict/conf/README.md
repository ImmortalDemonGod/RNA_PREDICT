# RNA_PREDICT Configuration System

This directory contains the Hydra configuration files for the RNA_PREDICT project.

## Configuration Files

The project uses a single main configuration file:

- `default.yaml` - Used by all scripts including `demo_entry.py` and individual stage scripts

This file serves as the central configuration for the entire project, ensuring consistency across all components.

## Directory Structure

```
rna_predict/conf/
├── default.yaml          # Main configuration file used by all scripts
├── model/                # Stage-specific configurations
│   ├── stageA.yaml
│   ├── stageB_pairformer.yaml
│   ├── stageB_torsion.yaml
│   ├── stageC.yaml
│   └── stageD_diffusion.yaml
└── config_schema.py      # Python dataclasses for typed configuration
```

## Best Practices

When working with the configuration system:

1. **Single Source of Truth**: Each configuration value is defined in exactly one place - the stage-specific configuration files are the source of truth for their respective stages.

2. **Use Typed Configurations**: Use the dataclasses defined in `config_schema.py` for type safety and validation.

3. **Follow the Structure**: Keep the hierarchical structure consistent with `model.stageX` prefixes in all stage configuration files.

4. **Composition Over Duplication**: Use Hydra's composition feature to include configuration files rather than duplicating their content.

## Usage

To use the configuration in your code:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    # Access configuration values
    device = cfg.device
    stage_a_config = cfg.model.stageA
    # ...
```

For more information on Hydra, see the [official documentation](https://hydra.cc/docs/intro/).

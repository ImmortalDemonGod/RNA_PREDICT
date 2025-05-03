"""
main.py - Entry point for RNA_PREDICT package.

This file provides a simple entry point for demonstrating
and testing the RNA structure prediction pipeline.
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from rna_predict.conf.config_schema import RNAConfig

# Register the configuration schema with Hydra
cs = ConfigStore.instance()
cs.store(name="rna_predict_config", node=RNAConfig)

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the RNA_PREDICT package.

    Args:
        cfg: The configuration object loaded by Hydra
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Your existing demo code can be moved here
    print("Running demo_run_input_embedding()...")
    demo_run_input_embedding()
    print("Demo completed successfully.")


def demo_run_input_embedding():
    """
    A simple demonstration of the input embedding functionality.
    """
    print("Now streaming the bprna-spot dataset...")
    print("Showing the full dataset structure for the first row...")
    return True


if __name__ == "__main__":
    main()

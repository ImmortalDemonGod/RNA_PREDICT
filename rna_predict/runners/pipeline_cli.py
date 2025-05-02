import torch
import hydra
from rna_predict.conf.config_schema import RNAConfig
from hydra.core.config_store import ConfigStore
from rna_predict.runners.full_pipeline import run_full_pipeline
from omegaconf import OmegaConf

# Register the structured config schema with Hydra
cs = ConfigStore.instance()
cs.store(name="default", node=RNAConfig)

@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: RNAConfig) -> None:
    """Main entry point for running the RNA prediction pipeline."""
    # Run pipeline with a test sequence
    print(f"\nProcessing sequence: {cfg.sequence}")

    # Canonical Hydra/OmegaConf config dump for debug
    print("\n--- Debug: Full Hydra Config ---")
    print(OmegaConf.to_yaml(cfg))

    # Run the pipeline with the updated config
    print("\n--- Running pipeline ---")
    final_res = run_full_pipeline(cfg)
    print("--- Pipeline completed ---")

    print("\n--- Pipeline Output ---")
    for k, v in final_res.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main()
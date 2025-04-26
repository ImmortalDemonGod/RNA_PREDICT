import torch
import hydra
from omegaconf import DictConfig
from rna_predict.runners.full_pipeline import run_full_pipeline

@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for running the RNA prediction pipeline."""
    # Run pipeline with a test sequence
    seq_input = "AUGCAUGG"
    print(f"\nProcessing sequence: {seq_input}")

    # Set the sequence in the config
    cfg.sequence = seq_input

    # Add debug output
    print("\n--- Debug: Config ---")
    print(f"Config type: {type(cfg)}")
    print(f"Config keys: {list(cfg.keys()) if hasattr(cfg, 'keys') else 'No keys method'}")
    print(f"Config has 'device': {'device' in cfg}")
    print(f"Config has 'sequence': {'sequence' in cfg}")

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
import torch
import hydra
from omegaconf import DictConfig
from rna_predict.run_full_pipeline import run_full_pipeline

@hydra.main(config_path="rna_predict/conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point for running the RNA prediction pipeline."""
    # Run pipeline with a test sequence
    seq_input = "AUGCAUGG"
    print(f"\nProcessing sequence: {seq_input}")

    final_res = run_full_pipeline(seq_input, cfg)

    print("\n--- Pipeline Output ---")
    for k, v in final_res.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
    print("Done.")

if __name__ == "__main__":
    main() 
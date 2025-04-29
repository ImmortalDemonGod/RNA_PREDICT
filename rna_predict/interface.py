"""High-level interface for the RNA_PREDICT pipeline."""

import hydra
from omegaconf import DictConfig, OmegaConf
from rna_predict.conf.config_schema import register_configs

# Register all configurations with Hydra
register_configs()

# Import and re-export RNAPredictor for backward compatibility
from rna_predict.predict import RNAPredictor

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function to demonstrate RNAPredictor using Hydra configuration."""
    print("Configuration loaded by Hydra:")
    print(OmegaConf.to_yaml(cfg))
    print("-" * 30)

    # Get sequence from config, with fallbacks
    sequence = None

    # First try direct sequence attribute
    if hasattr(cfg, "sequence"):
        sequence = cfg.sequence
    # Then try test_data.sequence
    elif hasattr(cfg, "test_data"):
        sequence = cfg.test_data.sequence

    if sequence is None:
        # Use default sequence if none found
        sequence = "ACGUACGU"
        print("Warning: No sequence found in config, using default sequence:", sequence)

    # Add sequence to top level config for consistency
    cfg.sequence = sequence

    # Instantiate the predictor with the loaded configuration
    try:
        from rna_predict.predict import RNAPredictor
        predictor = RNAPredictor(cfg)
    except Exception as e:
        print(f"Error initializing RNAPredictor: {e}")
        # Print relevant config sections for debugging
        if hasattr(cfg, "stageB_torsion"):
             print("Relevant config (stageB_torsion):")
             print(OmegaConf.to_yaml(cfg.stageB_torsion))
        if hasattr(cfg, "stageB_pairformer"):
             print("Relevant config (stageB_pairformer):")
             print(OmegaConf.to_yaml(cfg.stageB_pairformer))
        raise

    # Use test sequence from config
    test_sequence = cfg.sequence
    print(f"Running prediction for sequence: {test_sequence}")

    try:
        submission_df = predictor.predict_submission(test_sequence)
        print("\nSubmission DataFrame Head:")
        print(submission_df.head())
        output_path = "submission.csv"
        submission_df.to_csv(output_path, index=False)
        print(f"\nSubmission saved to {output_path}")
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise


if __name__ == "__main__":
    main()

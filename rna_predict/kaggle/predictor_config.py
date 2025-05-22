import logging
import torch
from omegaconf import OmegaConf, DictConfig # Ensure DictConfig is imported

from rna_predict.predict import RNAPredictor
from rna_predict.kaggle.kaggle_env import is_kaggle

# Configure logging for this module
logger = logging.getLogger(__name__)

def create_predictor(cfg: DictConfig) -> RNAPredictor:
    """
    Creates and returns an RNAPredictor instance, configured by the provided
    Hydra DictConfig (cfg) and adapted for the Kaggle environment if necessary.

    Args:
        cfg (DictConfig): The main Hydra configuration object, expected to have
                          sections like 'model' and 'prediction'.

    Returns:
        RNAPredictor: An instance of the RNA predictor.
    """
    logger.info("Creating RNAPredictor instance...")

    # Create a mutable copy of the config for predictor-specific adjustments
    # This ensures that the original cfg object passed to the main script is not modified.
    final_predictor_conf = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Resolve TorsionBERT model path and device, adapting for Kaggle environment
    if is_kaggle():
        torsion_bert_model_path_resolved = "/kaggle/input/rna-predict-dependencies/TorsionBERT_ckpt"
        logger.info(f"Kaggle environment detected. Using TorsionBERT path: {torsion_bert_model_path_resolved}")
        # In Kaggle, this resolved path is a specific checkpoint
        final_predictor_conf.model.stageB.torsion_bert.checkpoint_path = torsion_bert_model_path_resolved
        final_predictor_conf.model.stageB.torsion_bert.model_name_or_path = torsion_bert_model_path_resolved
    else:
        # Local environment logic
        custom_override_path = cfg.model.get("torsion_bert_model_path", None)
        if custom_override_path:
            logger.info(f"Local environment. Using custom TorsionBERT path from cfg.model.torsion_bert_model_path: {custom_override_path}")
            final_predictor_conf.model.stageB.torsion_bert.checkpoint_path = custom_override_path
            final_predictor_conf.model.stageB.torsion_bert.model_name_or_path = custom_override_path
        else:
            # No custom override, use defaults already in final_predictor_conf.model.stageB.torsion_bert
            # (model_name_or_path="sayby/rna_torsionbert", checkpoint_path=None)
            logger.info(f"Local environment. Using default TorsionBERT configuration: "
                        f"model_name_or_path='{final_predictor_conf.model.stageB.torsion_bert.model_name_or_path}', "
                        f"checkpoint_path='{final_predictor_conf.model.stageB.torsion_bert.checkpoint_path}'")
            # No changes needed to final_predictor_conf.model.stageB.torsion_bert paths here as defaults are already set.

    # Resolve device
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    device_resolved = cfg.model.get("device", default_device)
    logger.info(f"Using device: {device_resolved}")
    final_predictor_conf.model.device = device_resolved

    # Ensure prediction settings for diverse outputs are explicitly set for Kaggle.
    # This overrides any incoming config and restores the previous behavior shown in the git diff.
    final_predictor_conf.prediction = OmegaConf.create({
        "repeats": 5,
        "residue_atom_choice": 0, # Default from previous configuration
        "enable_stochastic_inference_for_submission": True,
        "submission_seeds": [42, 101, 2024, 7, 1991], # Optional: for reproducible stochastic runs
    })

    # Log the final configuration being passed to RNAPredictor
    logger.debug(f"RNAPredictor configuration: {OmegaConf.to_yaml(final_predictor_conf)}")

    try:
        predictor = RNAPredictor(final_predictor_conf) # Pass as positional argument
        logger.info("RNAPredictor instance created successfully.")
        return predictor
    except Exception as e:
        logger.error(f"Error creating RNAPredictor: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller

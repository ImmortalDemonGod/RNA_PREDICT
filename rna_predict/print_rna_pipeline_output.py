"""Print detailed output from RNA prediction pipeline with analysis."""
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import logging
from rna_predict.run_full_pipeline import run_full_pipeline
from rna_predict.conf.config_schema import validate_config
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Flag to indicate if Stage D is available
STAGE_D_AVAILABLE = False
try:
    from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
    STAGE_D_AVAILABLE = True
except ImportError:
    pass

def print_tensor_example(name: str, tensor, max_items: int = 5):
    """Print a tensor with shape information and example values.

    Args:
        name: Name of the tensor
        tensor: Tensor to print (can be numpy array, torch tensor, or None)
        max_items: Maximum number of items to print in each dimension
    """
    if tensor is None:
        print(f"{name}: None")
        return

    # Convert torch tensor to numpy if needed
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    # Print shape information
    print(f"{name}: shape={tensor.shape}")

    # Print example values based on dimensionality
    print("Example values:")

    if tensor.ndim == 1:
        # 1D tensor: print first max_items elements
        if tensor.shape[0] <= max_items:
            print(tensor)
        else:
            print(tensor[:max_items], "...")

    elif tensor.ndim == 2:
        # 2D tensor: print first max_items rows and columns
        rows = min(tensor.shape[0], max_items)
        cols = min(tensor.shape[1], max_items)

        for i in range(rows):
            if cols < tensor.shape[1]:
                # Format to match the expected output in the test
                # Replace the closing bracket with ", ...]"
                tensor_str = str(tensor[i, :cols])
                formatted_str = tensor_str[:-1] + ", ...]"
                print(formatted_str)
            else:
                print(tensor[i])

        if rows < tensor.shape[0]:
            print(" ...]")

    elif tensor.ndim >= 3:
        # Higher dimensional tensor: print first slice
        print("Data:")
        if tensor.shape[1] <= max_items:
            print(tensor[0])
        else:
            print("First slice:")
            for i in range(min(tensor.shape[1], max_items)):
                print(tensor[0, i])
            if tensor.shape[1] > max_items:
                print("...")

def analyze_coordinates(stage_name: str, coords: torch.Tensor) -> None:
    """Analyze coordinate tensors with detailed statistics."""
    logger.info(f"\n=== {stage_name} Coordinates Analysis ===")
    logger.info(f"Shape: {coords.shape}")

    # Basic statistics
    logger.info("Statistics:")
    logger.info(f"  Range: min={coords.min().item():.3f}, max={coords.max().item():.3f}")
    logger.info(f"  Mean: {coords.mean().item():.3f}")
    logger.info(f"  Std: {coords.std().item():.3f}")

    # Per-dimension analysis
    for dim in range(coords.shape[-1]):
        dim_data = coords[..., dim]
        logger.info(f"Dimension {dim} (xyz):")
        logger.info(f"  Range: min={dim_data.min().item():.3f}, max={dim_data.max().item():.3f}")
        logger.info(f"  Mean: {dim_data.mean().item():.3f}")
        logger.info(f"  Std: {dim_data.std().item():.3f}")

    # Distance analysis if we have multiple points
    if coords.shape[-2] > 1:
        dists = torch.cdist(coords.view(-1, 3), coords.view(-1, 3))
        logger.info("Pairwise Distances:")
        logger.info(f"  Min (non-zero): {dists[dists > 0].min().item():.3f}")
        logger.info(f"  Max: {dists.max().item():.3f}")
        logger.info(f"  Mean: {dists[dists > 0].mean().item():.3f}")

def analyze_adjacency(adj: torch.Tensor, sequence: str) -> None:
    """Analyze adjacency matrix with detailed statistics."""
    logger.info("\n=== Adjacency Matrix Analysis ===")
    logger.info(f"Shape: {adj.shape}")

    # Basic statistics
    logger.info("Statistics:")
    logger.info(f"  Range: min={adj.min().item():.3f}, max={adj.max().item():.3f}")
    logger.info(f"  Mean: {adj.mean().item():.3f}")
    logger.info(f"  Std: {adj.std().item():.3f}")

    # Connection analysis
    connections = adj > 0.5  # Threshold for considering a connection
    num_connections = connections.sum().item()
    logger.info("Connections:")
    logger.info(f"  Total connections: {num_connections}")
    logger.info(f"  Connection density: {num_connections / (adj.shape[0] * adj.shape[1]):.3f}")

    # Base pair analysis
    for i in range(len(sequence)):
        for j in range(i+1, len(sequence)):
            if connections[i,j]:
                logger.info(f"  Base pair: {sequence[i]}-{sequence[j]} at positions {i+1}-{j+1}")

def analyze_torsion_angles(angles: torch.Tensor) -> None:
    """Analyze torsion angles with detailed statistics."""
    logger.info("\n=== Torsion Angles Analysis ===")
    logger.info(f"Shape: {angles.shape}")

    # Convert to degrees for easier interpretation
    angles_deg = angles * 180 / np.pi

    # Overall statistics
    logger.info("Statistics (degrees):")
    logger.info(f"  Range: min={angles_deg.min().item():.1f}°, max={angles_deg.max().item():.1f}°")
    logger.info(f"  Mean: {angles_deg.mean().item():.1f}°")
    logger.info(f"  Std: {angles_deg.std().item():.1f}°")

    # Per-angle analysis
    angle_names = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']
    for i in range(min(angles.shape[1], len(angle_names))):
        angle_data = angles_deg[:, i]
        logger.info(f"{angle_names[i]}:")
        logger.info(f"  Range: min={angle_data.min().item():.1f}°, max={angle_data.max().item():.1f}°")
        logger.info(f"  Mean: {angle_data.mean().item():.1f}°")
        logger.info(f"  Std: {angle_data.std().item():.1f}°")

def analyze_embeddings(name: str, emb: torch.Tensor) -> None:
    """Analyze embedding tensors with detailed statistics."""
    logger.info(f"\n=== {name} Analysis ===")
    logger.info(f"Shape: {emb.shape}")

    # Basic statistics
    logger.info("Statistics:")
    logger.info(f"  Range: min={emb.min().item():.3f}, max={emb.max().item():.3f}")
    logger.info(f"  Mean: {emb.mean().item():.3f}")
    logger.info(f"  Std: {emb.std().item():.3f}")

    # Norm analysis
    norms = torch.norm(emb.float(), dim=-1)
    logger.info("Vector Norms:")
    logger.info(f"  Min: {norms.min().item():.3f}")
    logger.info(f"  Max: {norms.max().item():.3f}")
    logger.info(f"  Mean: {norms.mean().item():.3f}")

#@snoop
def setup_pipeline(cfg: DictConfig):
    """Set up the RNA prediction pipeline using Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        tuple: (config, device) where config is a dictionary of pipeline components
              and device is the computation device ("cpu" or "cuda")
    """
    # Import here to avoid circular imports
    from rna_predict.pipeline.stageB.torsion.torsion_bert_predictor import StageBTorsionBertPredictor
    from rna_predict.pipeline.stageB.pairwise.pairformer_wrapper import PairformerWrapper

    # Check if Stage D is available
    if not STAGE_D_AVAILABLE:
        print("[Warning] Stage D not available. Skipping diffusion refinement.")

    # Get device from config
    device = cfg.device
    logger.info(f"Using device: {device}")

    # Create a dummy Stage A predictor for testing
    class DummyStageAPredictor:
        def predict_adjacency(self, sequence):
            if not sequence:
                return torch.zeros((0, 0))

            n = len(sequence)
            adj = torch.zeros((n, n))

            # Set diagonal to 1.0
            for i in range(n):
                adj[i, i] = 1.0

            # Set adjacent positions to 0.8
            for i in range(n-1):
                adj[i, i+1] = 0.8
                adj[i+1, i] = 0.8

            # Set some non-local interactions
            if n > 4:
                adj[0, n-1] = 0.5
                adj[n-1, 0] = 0.5

            return adj

    # Create configuration dictionary with safe access to config values
    config = {
        "stageA_predictor": DummyStageAPredictor(),
        "enable_stageC": True,  # Default value
        "merge_latent": True,
        "init_z_from_adjacency": True,  # Default value
    }

    # Safely access configuration values if they exist
    if "stageC" in cfg and hasattr(cfg.stageC, "enabled"):
        config["enable_stageC"] = cfg.stageC.enabled

    if "stageB_pairformer" in cfg and hasattr(cfg.stageB_pairformer, "init_z_from_adjacency"):
        config["init_z_from_adjacency"] = cfg.stageB_pairformer.init_z_from_adjacency

    # Try to load Stage B models
    try:
        # Use the Hydra config for TorsionBertPredictor
        config["torsion_bert_model"] = StageBTorsionBertPredictor(cfg)
        logger.info("Loaded TorsionBertPredictor with Hydra config")
    except Exception as e:
        logger.warning(f"Could not load torsion BERT model: {e}")
        print(f"[Warning] Could not load torsion BERT model: {e}")
        # Create a dummy model
        class DummyTorsionModel:
            def __init__(self):
                self.output_dim = 14
            def predict(self, sequence):
                # Return random angles for testing
                n = len(sequence)
                return torch.randn(n, 14)  # 14 angles per residue to match output_dim

        config["torsion_bert_model"] = DummyTorsionModel()
        logger.info("Using dummy torsion model")

    try:
        # Use the Hydra config for PairformerWrapper
        config["pairformer_model"] = PairformerWrapper(cfg)
        logger.info("Loaded PairformerWrapper with Hydra config")
    except Exception as e:
        logger.warning(f"Could not load Pairformer model: {e}")
        # Create a dummy model
        class DummyPairformerModel:
            def predict(self, sequence, adjacency=None):  # adjacency param needed for interface compatibility
                # Return random embeddings for testing
                n = len(sequence)
                return torch.randn(n, 384), torch.randn(n, n, 128)  # Match expected dimensions

        config["pairformer_model"] = DummyPairformerModel()
        logger.info("Using dummy Pairformer model")

    # Create a dummy merger for testing
    class DummyMerger:
        def merge(self, torsion_embeddings, pairwise_embeddings):
            # Simple concatenation for testing
            return torch.cat([torsion_embeddings, pairwise_embeddings], dim=-1)

    config["merger"] = DummyMerger()

    # Add Stage D components if available and requested in config
    if STAGE_D_AVAILABLE and hasattr(cfg, "model") and hasattr(cfg.model, "stageD"):
        logger.info("Attempting to initialize Stage D components...")
        try:
            # Remove unused import as per F401
            # from rna_predict.pipeline.stageD.diffusion.protenix_diffusion_manager import ProtenixDiffusionManager
            logger.info("Stage D configuration present in model config")
            diffusion_manager = ProtenixDiffusionManager(cfg=cfg.model.stageD)  # Pass the Stage D config
            config["diffusion_manager"] = diffusion_manager
            config["stageD_config"] = cfg.model.stageD
            config["run_stageD"] = True
            logger.info("Stage D components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stage D components: {str(e)}")
            logger.error("Stack trace:", exc_info=True)
            config["diffusion_manager"] = None
            config["run_stageD"] = False
    else:
        if not STAGE_D_AVAILABLE:
            logger.warning("Stage D module not available")
        elif not hasattr(cfg, "model"):
            logger.warning("No model configuration found")
        elif not hasattr(cfg.model, "stageD"):
            logger.warning("No Stage D configuration found in model config")
        config["run_stageD"] = False

    return config, device

#@snoop
def _main_impl(cfg: DictConfig) -> None:
    """Main implementation function.

    Args:
        cfg: Hydra configuration object
    """
    print("Starting main function with Hydra config")
    print(f"Config keys: {list(cfg.keys())}")
    print()

    # Get sequence from config or use default
    sequence = cfg.sequence if hasattr(cfg, "sequence") else "AUGCAUGG"
    print(f"Running RNA prediction pipeline on sequence: {sequence}")
    print("Using Hydra configuration from default.yaml")

    # Check Stage D availability
    print(f"Stage D available: {STAGE_D_AVAILABLE}")

    # Run the pipeline
    try:
        # Run pipeline with the full config
        results = run_full_pipeline(
            sequence=sequence,
            cfg=cfg,  # Pass the entire config object
            device=cfg.device if hasattr(cfg, "device") else "cpu"
        )
        
        print("\nPipeline execution successful!")

        # Print and analyze results
        print("\nPipeline Output with Examples:")
        for key, value in results.items():
            print(f"Key: {key}")
            # Handle dicts (like atom_metadata) separately
            if isinstance(value, dict):
                print(f"{key}: dict with keys {list(value.keys())}")
                for subkey, subval in value.items():
                    print(f"  {subkey}: type={type(subval)}, len={len(subval) if hasattr(subval, 'len') else 'N/A'}")
                    # Optionally print a small example
                    if hasattr(subval, '__getitem__') and len(subval) > 0:
                        print(f"    First 5: {subval[:5]}")
            else:
                print_tensor_example(key, value)

        print("\n=== Detailed Analysis ===")
        
        # Analyze adjacency matrix if present
        if "adjacency" in results:
            analyze_adjacency(results["adjacency"], sequence)

        # Analyze torsion angles if present
        if "torsion_angles" in results:
            analyze_torsion_angles(results["torsion_angles"])

        # Analyze embeddings if present
        if "s_embeddings" in results:
            analyze_embeddings("Single Residue Embeddings", results["s_embeddings"])
        if "z_embeddings" in results:
            analyze_embeddings("Pairwise Embeddings", results["z_embeddings"])

        # Analyze coordinates if present
        if "partial_coords" in results and results["partial_coords"] is not None:
            analyze_coordinates("Partial Coordinates", results["partial_coords"])
        if "final_coords" in results and results["final_coords"] is not None:
            analyze_coordinates("Final Coordinates", results["final_coords"])

        print("Done.")

    except Exception as e:
        print(f"\nError running pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Done.")
        return

@hydra.main(version_base=None, config_path="conf", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Args:
        cfg: Hydra configuration object
    """
    # Ensure proper configuration structure
    if not OmegaConf.is_dict(cfg):
        cfg = OmegaConf.create(cfg)
    
    # Ensure model configuration exists
    if not hasattr(cfg, "model"):
        cfg.model = OmegaConf.create({})
    
    # Ensure required model stages exist
    required_stages = ["stageA", "stageB", "stageC", "stageD"]
    for stage in required_stages:
        if not hasattr(cfg.model, stage):
            cfg.model[stage] = OmegaConf.create({})
            
    # Ensure stageB has required subconfigs
    if not hasattr(cfg.model.stageB, "torsion_bert"):
        cfg.model.stageB.torsion_bert = OmegaConf.create({})
    if not hasattr(cfg.model.stageB, "pairformer"):
        cfg.model.stageB.pairformer = OmegaConf.create({})
        cfg.model.stageB.pairformer.init_z_from_adjacency = False
    
    # Validate the config strictly before any pipeline logic
    validate_config(cfg)
    
    # Run the main implementation
    result = _main_impl(cfg)
    return result

# Example usage (if run directly, using default config)
if __name__ == "__main__":
    # This will call main() through Hydra, which in turn calls _main_impl()
    main()

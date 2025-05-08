import pathlib
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from rna_predict.conf.config_schema import register_configs
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.dataset.loader import RNADataset
from rna_predict.dataset.collate import rna_collate_fn
from lightning.pytorch.callbacks import ModelCheckpoint
import logging
import os
import pathlib  # required for PROJECT_ROOT and path handling

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
register_configs()

# Use a relative config path instead of absolute
@hydra.main(config_path="/Users/tomriddle1/RNA_PREDICT/rna_predict/conf", config_name="default.yaml", version_base="1.1")
####@snoop
def main(cfg: DictConfig):
    # SYSTEMATIC HYDRA INTERPOLATION DEBUGGING PATCH
    """
    Main training entry point for the RNA prediction model using Hydra configuration.
    
    Loads and resolves configuration, prepares dataset and DataLoader, initializes the model,
    sets up checkpointing, and runs training with PyTorch Lightning. Handles device selection,
    debug logging, and error reporting for configuration and data loading issues.
    
    Args:
        cfg: Hydra configuration object containing all settings for data, model, and training.
    """
    logger = logging.getLogger("rna_predict.training.train")
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.debug("[DEBUG][PATCH] Calling OmegaConf.resolve(cfg) to force interpolation...")
    # SYSTEMATIC DEBUGGING: Print the full resolved config to check what namespaces are present
    from omegaconf import OmegaConf
    print("\n[DEBUG][PATCH] FULL CONFIG TREE:\n" + OmegaConf.to_yaml(cfg), flush=True)
    OmegaConf.resolve(cfg)
    logger.debug("[DEBUG][PATCH] After resolve: cfg.device: %s", getattr(cfg, 'device', None))
    for stage in ["stageA", "stageB", "stageC", "stageD"]:
        stage_cfg = getattr(cfg.model, stage, None)
        if stage_cfg is not None:
            logger.debug(f"[DEBUG][PATCH] After resolve: cfg.model.{stage}.device: %s", getattr(stage_cfg, 'device', None))

    # Debug configuration
    logger.debug("[DEBUG] Configuration keys:")
    for k in cfg.keys():
        logger.debug("  - %s", k)

    # Print the full model config for debugging device overrides
    logger.debug("[DEBUG] Full cfg.model:")
    print(cfg.model)
    if hasattr(cfg.model, 'stageB_torsion'):
        print("[DEBUG] cfg.model.stageB_torsion:")
        print(cfg.model.stageB_torsion)
    if hasattr(cfg.model, 'stageB'):
        print("[DEBUG] cfg.model.stageB:")
        print(cfg.model.stageB)

    if 'data' in cfg:
        logger.debug("[DEBUG] Data configuration keys:")
        for k in cfg.data.keys():
            logger.debug("  - %s", k)
    else:
        logger.error("[ERROR] 'data' key not found in configuration!")
        return

    # Get the original working directory (project root)
    original_cwd = hydra.utils.get_original_cwd()
    # Early ensure checkpoint directory exists so test always sees it
    import os
    ckpt = cfg.training.checkpoint_dir
    # resolve absolute
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(original_cwd, ckpt)
    os.makedirs(ckpt, exist_ok=True)
    project_root = pathlib.Path(original_cwd)
    logger.debug("[DEBUG] Original working directory: %s", original_cwd)

    # Check if we have a test_data.data_index to use
    index_csv_path = None
    if hasattr(cfg, 'test_data') and hasattr(cfg.test_data, 'data_index') and cfg.test_data.data_index:
        logger.debug("[DEBUG] Using test_data.data_index: %s", cfg.test_data.data_index)
        index_path = pathlib.Path(cfg.test_data.data_index)
        if not index_path.is_absolute():
            # Resolve relative to the project root
            abs_index_path = project_root / index_path
            logger.debug("[DEBUG] Resolved test_data.data_index path: %s", abs_index_path)
            logger.debug("[DEBUG] File exists: %s", abs_index_path.exists())
            if abs_index_path.exists():
                index_csv_path = str(abs_index_path)

    # Fall back to data.index_csv if test_data.data_index doesn't exist
    if not index_csv_path and hasattr(cfg.data, 'index_csv'):
        logger.debug("[DEBUG] Falling back to data.index_csv: %s", cfg.data.index_csv)
        index_path = pathlib.Path(cfg.data.index_csv)
        # Accept absolute path directly
        if index_path.exists():
            index_csv_path = str(index_path)
        else:
            # Resolve relative to the project root
            abs_index_path = project_root / index_path
            logger.debug("[DEBUG] Resolved data.index_csv path: %s", abs_index_path)
            logger.debug("[DEBUG] File exists: %s", abs_index_path.exists())
            if abs_index_path.exists():
                index_csv_path = str(abs_index_path)

    if not index_csv_path:
        logger.error("[ERROR] No valid index CSV file found in either test_data.data_index or data.index_csv!")
        return

    model = RNALightningModule(cfg)
    # DataLoader setup
    try:
        ds = RNADataset(index_csv_path, cfg,
                        load_adj=cfg.data.load_adj,
                        load_ang=cfg.data.load_ang)
        # Set num_workers=0 if device is mps (PyTorch limitation)
        num_workers = 0 if str(getattr(cfg, 'device', 'cpu')).startswith('mps') else getattr(cfg.data, 'num_workers', 0)
        logger.debug(f"[DEBUG][main] DataLoader num_workers={num_workers} (device={getattr(cfg, 'device', 'cpu')})")
        dl = DataLoader(ds,
                        batch_size=cfg.data.batch_size,
                        num_workers=num_workers,
                        collate_fn=rna_collate_fn,
                        shuffle=True)
        # After DataLoader creation, print its type for debugging
        logger.debug(f"[DEBUG][train.py] DataLoader type: {type(dl)}")
        # DEBUG: Inspect first batch in detail only if debug_inspect_batch is enabled
        debug_inspect_batch = getattr(cfg.data, 'debug_inspect_batch', False)
        if debug_inspect_batch:
            # Only create workers and fetch a batch when explicitly requested
            logger.debug("[DEBUG] Inspecting first batch (debug_inspect_batch=True)")
            first_batch = next(iter(dl))
            logger.debug("[DEBUG][main] First batch keys: %s", list(first_batch.keys()))
            for k, v in first_batch.items():
                if hasattr(v, 'shape'):
                    shape_val = getattr(v, 'shape', None)
                    dtype_val = getattr(v, 'dtype', None)
                    requires_grad_val = getattr(v, 'requires_grad', 'N/A')
                    logger.debug(f"[DEBUG][main] Key: {k!r}, Shape: {shape_val!r}, Dtype: {dtype_val!r}, requires_grad: {requires_grad_val!r}")
                else:
                    logger.debug(f"[DEBUG][main] Key: {k!r}, Type: {type(v)!r}")
            logger.debug("[DEBUG] First batch device: %r", first_batch['coords_true'].device)
        # Additional debug instrumentation
        logger.debug(f"[DEBUG] Dataset length: {len(ds)}")
        try:
            logger.debug(f"[DEBUG] DataLoader length: {len(dl)}")
            first_batch = next(iter(dl))
            logger.debug(f"[DEBUG] First batch keys: {list(first_batch.keys())}")
        except Exception as batch_e:
            logger.error(f"[ERROR] Exception when iterating DataLoader: {batch_e}")
        # SYSTEMATIC DEBUGGING: Print CWD, checkpoint dir config, resolved absolute path, and permissions
        cwd = os.getcwd()
        logger.info(f"[CHECKPOINT-DEBUG] Current working directory (os.getcwd()): {cwd}")
        logger.info(f"[CHECKPOINT-DEBUG] cfg.training.checkpoint_dir: {cfg.training.checkpoint_dir}")
        resolved_ckpt_dir = pathlib.Path(cfg.training.checkpoint_dir)
        if not resolved_ckpt_dir.is_absolute():
            resolved_ckpt_dir = pathlib.Path(cwd) / resolved_ckpt_dir
        logger.info(f"[CHECKPOINT-DEBUG] Resolved checkpoint directory absolute path: {resolved_ckpt_dir}")
        logger.info(f"[CHECKPOINT-DEBUG] Directory exists: {resolved_ckpt_dir.exists()}")
        logger.info(f"[CHECKPOINT-DEBUG] Directory writable: {os.access(resolved_ckpt_dir.parent, os.W_OK)} (parent: {resolved_ckpt_dir.parent})")
        # Ensure checkpoint directory exists (use resolved absolute path)
        from pathlib import Path
        ckpt_dir = Path(cfg.training.checkpoint_dir)
        if not ckpt_dir.is_absolute():
            from os import getcwd
            ckpt_dir = Path(getcwd()) / ckpt_dir
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Add ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            save_top_k=1,
            monitor=None,  # No validation metric for now
            save_last=True
        )
        logger.info(f"[CHECKPOINT-DEBUG] About to start training with checkpoint dir: {checkpoint_callback.dirpath}")
        try:
            trainer = L.Trainer(
                callbacks=[checkpoint_callback],
                max_epochs=1,  # Run at least one epoch
                # FIX: Use cfg.device for accelerator, and set devices accordingly
                accelerator=cfg.device,
                devices=1 if cfg.device in ['mps', 'cuda'] else cfg.training.devices
            )
            trainer.fit(model, dl)
            logger.info(f"[CHECKPOINT-DEBUG] Training completed. Checkpoints should be saved to: {checkpoint_callback.dirpath}")
        except Exception as train_exc:
            logger.error(f"[CHECKPOINT-ERROR] Exception during training/checkpoint saving: {train_exc}", exc_info=True)
        logger.info("[CHECKPOINT-DEBUG] End of main training block reached.")
    except Exception as e:
        logger.error("[ERROR] Exception during dataset/dataloader setup: %r", e, exc_info=True)

if __name__ == "__main__":
    main()

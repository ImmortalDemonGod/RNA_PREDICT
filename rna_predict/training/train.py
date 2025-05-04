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

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
register_configs()

# Use a relative config path instead of absolute
@hydra.main(config_path="../conf", config_name="default.yaml", version_base="1.1")
####@snoop
def main(cfg: DictConfig):
    # SYSTEMATIC HYDRA INTERPOLATION DEBUGGING PATCH
    print("[DEBUG][PATCH] Calling OmegaConf.resolve(cfg) to force interpolation...")
    OmegaConf.resolve(cfg)
    print("[DEBUG][PATCH] After resolve: cfg.device:", getattr(cfg, 'device', None))
    for stage in ["stageA", "stageB", "stageC", "stageD"]:
        stage_cfg = getattr(cfg.model, stage, None)
        if stage_cfg is not None:
            print(f"[DEBUG][PATCH] After resolve: cfg.model.{stage}.device:", getattr(stage_cfg, 'device', None))

    # Debug configuration
    print("[DEBUG] Configuration keys:")
    for k in cfg.keys():
        print("  - {!s}".format(k))

    # Print the full model config for debugging device overrides
    print("[DEBUG] Full cfg.model:")
    print(cfg.model)
    if hasattr(cfg.model, 'stageB_torsion'):
        print("[DEBUG] cfg.model.stageB_torsion:")
        print(cfg.model.stageB_torsion)
    if hasattr(cfg.model, 'stageB'):
        print("[DEBUG] cfg.model.stageB:")
        print(cfg.model.stageB)

    if 'data' in cfg:
        print("[DEBUG] Data configuration keys:")
        for k in cfg.data.keys():
            print("  - {!s}".format(k))
    else:
        print("[ERROR] 'data' key not found in configuration!")
        return

    # Get the original working directory (project root)
    original_cwd = hydra.utils.get_original_cwd()
    project_root = pathlib.Path(original_cwd)
    print("[DEBUG] Original working directory: {}".format(original_cwd))

    # Check if we have a test_data.data_index to use
    index_csv_path = None
    if hasattr(cfg, 'test_data') and hasattr(cfg.test_data, 'data_index') and cfg.test_data.data_index:
        print("[DEBUG] Using test_data.data_index: {}".format(cfg.test_data.data_index))
        index_path = pathlib.Path(cfg.test_data.data_index)
        if not index_path.is_absolute():
            # Resolve relative to the project root
            abs_index_path = project_root / index_path
            print("[DEBUG] Resolved test_data.data_index path: {}".format(abs_index_path))
            print("[DEBUG] File exists: {}".format(abs_index_path.exists()))
            if abs_index_path.exists():
                index_csv_path = str(abs_index_path)

    # Fall back to data.index_csv if test_data.data_index doesn't exist
    if not index_csv_path and hasattr(cfg.data, 'index_csv'):
        print("[DEBUG] Falling back to data.index_csv: {}".format(cfg.data.index_csv))
        index_path = pathlib.Path(cfg.data.index_csv)
        if not index_path.is_absolute():
            # Resolve relative to the project root
            abs_index_path = project_root / index_path
            print("[DEBUG] Resolved data.index_csv path: {}".format(abs_index_path))
            print("[DEBUG] File exists: {}".format(abs_index_path.exists()))
            if abs_index_path.exists():
                index_csv_path = str(abs_index_path)

    if not index_csv_path:
        print("[ERROR] No valid index CSV file found in either test_data.data_index or data.index_csv!")
        return

    model = RNALightningModule(cfg)
    # DataLoader setup
    try:
        ds = RNADataset(index_csv_path, cfg,
                        load_adj=cfg.data.load_adj,
                        load_ang=cfg.data.load_ang)
        # Set num_workers=0 if device is mps (PyTorch limitation)
        num_workers = 0 if str(getattr(cfg, 'device', 'cpu')).startswith('mps') else getattr(cfg.data, 'num_workers', 0)
        print(f"[DEBUG][main] DataLoader num_workers={num_workers} (device={getattr(cfg, 'device', 'cpu')})")
        dl = DataLoader(ds,
                        batch_size=cfg.data.batch_size,
                        num_workers=num_workers,
                        collate_fn=lambda batch: rna_collate_fn(batch, cfg=cfg),
                        shuffle=True)
        # DEBUG: Inspect first batch in detail only if debug_inspect_batch is enabled
        debug_inspect_batch = getattr(cfg.data, 'debug_inspect_batch', False)
        if debug_inspect_batch:
            # Only create workers and fetch a batch when explicitly requested
            print("[DEBUG] Inspecting first batch (debug_inspect_batch=True)")
            first_batch = next(iter(dl))
            print("[DEBUG][main] First batch keys:", list(first_batch.keys()))
            for k, v in first_batch.items():
                if hasattr(v, 'shape'):
                    shape_val = getattr(v, 'shape', None)
                    dtype_val = getattr(v, 'dtype', None)
                    requires_grad_val = getattr(v, 'requires_grad', 'N/A')
                    print(f"[DEBUG][main] Key: {k!r}, Shape: {shape_val!r}, Dtype: {dtype_val!r}, requires_grad: {requires_grad_val!r}")
                else:
                    print(f"[DEBUG][main] Key: {k!r}, Type: {type(v)!r}")
            print("[DEBUG] First batch device: {!r}".format(first_batch['coords_true'].device))
        # Add ModelCheckpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.training.checkpoint_dir,
            save_top_k=1,
            monitor=None,  # No validation metric for now
            save_last=True
        )
        trainer = L.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=1,  # Run at least one epoch
            # FIX: Use cfg.device for accelerator, and set devices accordingly
            accelerator=cfg.device,
            devices=1 if cfg.device in ['mps', 'cuda'] else cfg.training.devices
        )
        trainer.fit(model, dl)
        print(f"[DEBUG] Checkpoints saved to: {checkpoint_callback.dirpath}")
    except Exception as e:
        print("[ERROR] Exception during dataset/dataloader setup: {!r}".format(e))

if __name__ == "__main__":
    main()

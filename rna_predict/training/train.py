import pathlib
import lightning as L
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from rna_predict.conf.config_schema import register_configs
from rna_predict.training.rna_lightning_module import RNALightningModule
from rna_predict.dataset.loader import RNADataset
from rna_predict.dataset.collate import rna_collate_fn

# Get the project root directory
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
register_configs()

# Use a relative config path instead of absolute
@hydra.main(config_path="../conf", config_name="default.yaml", version_base="1.1")
##@snoop
def main(cfg: DictConfig):
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

    # Check if index_csv exists and resolve it relative to the project root
    if hasattr(cfg.data, 'index_csv'):
        print("[DEBUG] Original index_csv: {}".format(cfg.data.index_csv))
        index_path = pathlib.Path(cfg.data.index_csv)
        if not index_path.is_absolute():
            # Resolve relative to the project root
            abs_index_path = project_root / index_path
            print("[DEBUG] Resolved index_csv path: {}".format(abs_index_path))
            print("[DEBUG] File exists: {}".format(abs_index_path.exists()))
            # Update the path in the config
            cfg.data.index_csv = str(abs_index_path)
    else:
        print("[ERROR] 'index_csv' key not found in data configuration!")
        return

    model = RNALightningModule(cfg)
    # DataLoader setup
    try:
        ds = RNADataset(cfg.data.index_csv, cfg,
                        load_adj=cfg.data.load_adj,
                        load_ang=cfg.data.load_ang)
        dl = DataLoader(ds,
                        batch_size=cfg.data.batch_size,
                        num_workers=cfg.data.num_workers,
                        collate_fn=rna_collate_fn,
                        shuffle=True)
        # DEBUG: Inspect first batch in detail
        first_batch = next(iter(dl))
        print("[DEBUG][main] First batch keys:", list(first_batch.keys()))
        for k, v in first_batch.items():
            if hasattr(v, 'shape'):
                print(f"[DEBUG][main] Key: {k}, Shape: {getattr(v, 'shape', None)}, Dtype: {getattr(v, 'dtype', None)}, requires_grad: {getattr(v, 'requires_grad', 'N/A')}")
            else:
                print(f"[DEBUG][main] Key: {k}, Type: {type(v)}")
        print("[DEBUG] First batch device: {}".format(first_batch['coords_true'].device))
        trainer = L.Trainer(fast_dev_run=True)
        trainer.fit(model, dl)
    except Exception as e:
        print("[ERROR] Exception during dataset/dataloader setup: {}".format(e))

if __name__ == "__main__":
    main()

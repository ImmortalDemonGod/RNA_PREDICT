import lightning as L
from rna_predict.training.rna_lightning_module import RNALightningModule

def main():
    model = RNALightningModule()
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model)

if __name__ == "__main__":
    main()

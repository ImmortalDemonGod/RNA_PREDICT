import lightning as L
from rna_predict.training.rna_lightning_module import RNALightningModule

def test_trainer_fast_dev_run():
    model = RNALightningModule()
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model)

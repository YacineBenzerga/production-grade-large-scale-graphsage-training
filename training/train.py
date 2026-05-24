import os
import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from core.data_module import AmazonProductsDataModule
from core.system_module import GNNProductionSystem
import torch


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    pl.seed_everything(cfg.seed)

    torch.set_float32_matmul_precision('high')

    dm = AmazonProductsDataModule(cfg)
    dm.setup()

    system = GNNProductionSystem(cfg,
                                 in_channels=dm.in_channels,
                                 out_channels=dm.out_channels)

    checkpoint_callback = ModelCheckpoint(
        irpath=os.path.join(cfg.artifacts_dir, "models"),
        filename="best_model",
        monitor="val_macro_f1",
        mode="max",
        save_top_k=1
        )
    
    early_stop_callback = EarlyStopping(monitor=cfg.early_stopping.monitor, 
                                        patience=cfg.early_stopping.patience,
                                        mode=cfg.early_stopping.mode)
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        precision="16-mixed"
    )
    
    trainer.fit(system, datamodule=dm)

if __name__ == "__main__":
    main()
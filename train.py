import hydra
import mlflow
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from src.data_module import AmazonProductsDataModule
from src.system_module import GraphSAGELightningEngine

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    import torch
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg.seed)

    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    custom_run_name = f"{cfg.mlflow.run_name_prefix}_batch{cfg.data.batch_size}_lr{cfg.model.lr}_{timestamp}"

    
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=custom_run_name
    )

    datamodule = AmazonProductsDataModule(
        cfg
    )

    engine = GraphSAGELightningEngine(model_cfg=cfg.model)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/f1",
        mode="max",
        save_top_k=1,
        filename="best-graphsage-{epoch:02d}-{val_f1:.3f}"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val/f1",
        patience=3,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=mlflow_logger
    )

    
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(engine, datamodule=datamodule)

if __name__ == "__main__":
    main()
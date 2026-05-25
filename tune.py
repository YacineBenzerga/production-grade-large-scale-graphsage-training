import hydra
import optuna
import mlflow
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from src.datamodule import AmazonProductsDataModule
from src.systemodule import GraphSAGELightningEngine

def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    cfg.model.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.model.hidden_channels = trial.suggest_categorical("hidden_channels", [128, 256, 512])
    cfg.model.dropout = trial.suggest_float("dropout", 0.2, 0.6)
    
    
    trial_run_name = f"Trial_{trial.number}_hdim{cfg.model.hidden_channels}_lr{cfg.model.lr:.4f}"
    
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=trial_run_name
    )
    
    datamodule = AmazonProductsDataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        sizes=list(cfg.data.sizes),
        num_workers=cfg.data.num_workers
    )
    
    engine = GraphSAGELightningEngine(model_cfg=cfg.model)
    
    trainer = pl.Trainer(
        max_epochs=5,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        enable_checkpointing=False,
        logger=mlflow_logger
    )
    
    trainer.fit(engine, datamodule=datamodule)
    return trainer.callback_metrics["val/f1"].item()

@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    hpo_cfg = cfg.get("optuna", {"study_name": "graphsage_tuning", "n_trials": 10})
    
    study = optuna.create_study(
        study_name=hpo_cfg.study_name,
        direction="maximize"
    )
    study.optimize(lambda trial: objective(trial, cfg), n_trials=hpo_cfg.n_trials)
    
    print("Best Trial parameters:")
    print(study.best_trial.params)

if __name__ == "__main__":
    main()
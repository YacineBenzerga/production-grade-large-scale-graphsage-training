import pytorch_lightning as pl
import torch
from torchmetrics.classification import MultilabelF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR
from core.arch import GraphSAGEWithLabelCorrelation
from core.loss import AssymetricLoss



class GNNProductionSystem(pl.LightningModule):
    def __init__(self, cfg, in_channels, out_channels):
        super().__init__()

        self.save_hyperparameters(cfg)
        self.cfg = cfg
        

        self.model = GraphSAGEWithLabelCorrelation(
            in_channels=in_channels,
            hidden_dim=cfg.model.hidden_dim,
            out_channels=out_channels,
            num_layers=cfg.model.num_layers,
            dropout_rate=cfg.model.dropout_rate
        )

        self.loss_fn = AssymetricLoss(gamma_neg=cfg.model.asl_gamma_neg, 
                                      gamma_pos=cfg.model.asl_gamma_pos,
                                      clip=cfg.model.asl_clip)
        
        self.val_f1 = MultilabelF1Score(num_labels=out_channels, average="macro")


    def forward(self, x, edge_index):
        return self.model(x, edge_index)


    def training_step(self, batch, _batch_idx):

        out = self(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].float()

        loss = self.loss_fn(out, y)

        self.val_f1.update(out, y)
        self.log("val_loss", loss, batch_size=batch.batch_size)
        return True
    

    def on_validation_epoch_end(self):
        val_f1_score = self.val_f1.compute()
        self.log("val_macro_f1", val_f1_score, prog_bar=True)
        self.val_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.model.lr,
            weight_decay=self.cfg.model.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": { 
                "scheduler":scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }



            
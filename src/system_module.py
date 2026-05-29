import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import MultilabelF1Score
from src.model import GraphSAGEWithLabelCorrelation

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        p = torch.sigmoid(x)
        targets = y.float()
        loss_pos = targets * torch.log(p.clamp(min=self.eps)) * ((1 - p) ** self.gamma_pos)
        
        if self.clip and self.clip > 0:
            p_neg = (p + self.clip).clamp(max=1.0)
        else:
            p_neg = p
            
        loss_neg = (1 - targets) * torch.log((1 - p_neg).clamp(min=self.eps)) * (p_neg ** self.gamma_neg)
        loss = - (loss_pos + loss_neg)
        return loss.mean()


class GraphSAGELightningEngine(pl.LightningModule):
    def __init__(self, model_cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = model_cfg
        
        self.model = GraphSAGEWithLabelCorrelation(
            in_feats=self.cfg.in_channels,          
            hidden_feats=self.cfg.hidden_channels,  
            embedding_dim=self.cfg.get("embedding_dim", 128),  
            num_classes=self.cfg.out_channels       
        )
        
        self.loss_fn = AsymmetricLoss(
            gamma_neg=self.cfg.get("gamma_neg", 4),
            gamma_pos=self.cfg.get("gamma_pos", 1),
            clip=self.cfg.get("clip", 0.05)
        )
        
        self.train_f1 = MultilabelF1Score(num_labels=self.cfg.out_channels, average='micro')
        self.val_f1 = MultilabelF1Score(num_labels=self.cfg.out_channels, average='micro')

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def common_step(self, batch, stage: str):
        out = self(batch.x, batch.edge_index)[:batch.batch_size]
        y = batch.y[:batch.batch_size].float()
        loss = self.loss_fn(out, y)
        preds = (torch.sigmoid(out) > 0.5).long()
        self.log(f"{stage}/loss", loss, batch_size=batch.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self.common_step(batch, "train")
        self.train_f1(preds, y.long())
        self.log("train/f1", self.train_f1, batch_size=batch.batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self.common_step(batch, "val")
        self.val_f1(preds, y.long())
        self.log("val/f1", self.val_f1, batch_size=batch.batch_size, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
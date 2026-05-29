import pytorch_lightning as pl
from torch_geometric.datasets import AmazonProducts
from torch_geometric.loader import NeighborLoader
import torch
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

class AmazonProductsDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.batch_size = cfg.data.batch_size
        self.num_neighbors = list(cfg.data.num_neighbors)
        self.num_workers = cfg.data.num_workers
        self.sample_ratio = getattr(cfg.data, "sample_ratio", None)

    def prepare_data(self):
        AmazonProducts(root=self.root_dir)

    def setup(self, stage=None):
        dataset = AmazonProducts(root=self.root_dir)
        self.data = dataset[0]
        
        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes
        self.train_nodes_mask = self.data.train_mask

        if self.sample_ratio is not None and self.sample_ratio < 1.0:
            self.train_nodes_mask = self._generate_stratified_mask()

    def _generate_stratified_mask(self):
        y_train = self.data.y[self.data.train_mask]
        
        label_frequencies = y_train.sum(dim=0)
        label_frequencies = torch.where(label_frequencies == 0, torch.ones_like(label_frequencies), label_frequencies)
        inverse_frequencies = 1.0 / label_frequencies

        node_label_weights = y_train * inverse_frequencies.unsqueeze(0)
        has_labels = (y_train.sum(dim=1) > 0)
        
        rarest_label_per_node = torch.where(
            has_labels, 
            node_label_weights.argmax(dim=1), 
            torch.tensor(self.out_channels, device=y_train.device)
        ).cpu().numpy()

        
        stratifier = StratifiedShuffleSplit(
            n_splits=1, 
            train_size=self.sample_ratio, 
            random_state=self.cfg.seed
        )
        
        train_indices = np.where(self.data.train_mask.cpu().numpy())[0]
        
        
        for target_sample_idx, _ in stratifier.split(np.zeros(len(rarest_label_per_node)), rarest_label_per_node):
            absolute_stratified_indices = train_indices[target_sample_idx]

        stratified_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        stratified_mask[absolute_stratified_indices] = True
        return stratified_mask

    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors, 
            batch_size=self.batch_size, 
            shuffle=True, 
            input_nodes=self.train_nodes_mask, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            input_nodes=self.data.val_mask,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            input_nodes=self.data.test_mask,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
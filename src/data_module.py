import os
import gc
import torch
from torch_geometric.datasets import AmazonProducts
from torch_geometric.loader import NeighborLoader
import pytorch_lightning as pl

class AmazonProductsDataModule(pl.LightningDataModule):
    def __init__(self, root_dir: str, batch_size: int, sizes: list, num_workers: int):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.sizes = sizes
        self.num_workers = num_workers
        self.data = None

    def prepare_data(self):
        AmazonProducts(root=self.root_dir)

    def setup(self, stage: str = None):
        dataset = AmazonProducts(root=self.root_dir)
        self.data = dataset[0]
        
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.sizes,
            batch_size=self.batch_size,
            input_nodes=self.data.train_mask,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.sizes,
            batch_size=self.batch_size,
            input_nodes=self.data.val_mask,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.sizes,
            batch_size=self.batch_size,
            input_nodes=self.data.test_mask,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
import pytorch_lightning as pl
from torch_geometric.datasets import AmazonProducts
from torch_geometric.loader import NeighborLoader



class AmazonProducsDataModule(pl.LightningDataModule):
    def __ini__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.num_neighbors = list(cfg.data.num_neighbors)

    def prepare_data(self):
        AmazonProducts(root=self.root_dir)

    def setup(self, stage=None):
        dataset = AmazonProducts(root=self.root_dir)
        self.data = dataset[0]
        self.in_channels = dataset.num_features
        self.out_channels = dataset.num_classes


    def train_dataloader(self):
        return NeighborLoader(
            self.data,
            num_neighbors=self.num_neighbors, 
            batch_size=self.batch_size, 
            shuffle=True, 
            input_nodes=self.data.train_mask, 
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
            persistent_workers=True) 
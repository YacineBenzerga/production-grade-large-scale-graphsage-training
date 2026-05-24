import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEWithLabelCorrelation(nn.Module):
    def __init__(self, in_feats, hidden_feats, embedding_dim, num_classes):
        super().__init__()
        
        #backbone
        self.conv1 = SAGEConv(in_feats, hidden_feats, root_weight=True)
        self.conv2 = SAGEConv(hidden_feats, embedding_dim, root_weight=True)
        
        #skip connections
        self.input_skip_projector = nn.Linear(in_feats, embedding_dim)
        self.layer1_skip_projector = nn.Linear(hidden_feats, embedding_dim)
        
        #classifer and regulariztaion
        self.mlp1 = nn.Linear(embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128) 
        self.dropout = nn.Dropout(0.3) 
        self.mlp2 = nn.Linear(128, num_classes)
        
        #maybe we can learn through correlation of labels
        self.label_correlation = nn.Linear(num_classes, num_classes, bias=False)
        self.label_dropout = nn.Dropout(0.15)  
        
        nn.init.eye_(self.label_correlation.weight)

    def forward(self, x, edge_index):
        h1 = F.relu(self.conv1(x, edge_index))
        
        # Hop 2: Deeper aggregation mixed with inter-layer hidden representation
        h2 = F.relu(self.conv2(h1, edge_index)) + self.layer1_skip_projector(h1)
        
        embeddings = F.relu(h2 + self.input_skip_projector(x))
        
        
        out = self.mlp1(embeddings)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        marginal_logits = self.mlp2(out)
        
        regularized_marginal_logits = self.label_dropout(marginal_logits)
        
        conditioned_logits = self.label_correlation(regularized_marginal_logits)
        
        return conditioned_logits
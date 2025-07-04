import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GATConv
from torch_geometric.nn import global_mean_pool as gmp, global_add_pool as gap
from torch_geometric.nn import Sequential
from torch.nn import Linear

from torch.nn import Linear, TransformerEncoder, TransformerEncoderLayer

class GINNet(torch.nn.Module):
    def __init__(self, eps=0., train_eps=True, num_heads=8, num_layers=2):
        super(GINNet, self).__init__()

        # GIN layers (GINEConv)
        self.gin1 = GINEConv(nn=Linear(93, 512).cuda(), eps=eps, train_eps=train_eps).cuda()
        self.gin2 = GINEConv(nn=Linear(512, 1024).cuda(), eps=eps, train_eps=train_eps).cuda()
        self.gin3 = GINEConv(nn=Linear(1024, 512).cuda(), eps=eps, train_eps=train_eps).cuda()

        # FC layers to process edge attributes
        self.fc1 = Linear(11, 93).cuda()
        self.fc2 = Linear(11, 512).cuda()
        self.fc3 = Linear(11, 1024).cuda()

        # Transformer Encoder (used to learn global interactions)
        self.transformer_layer = TransformerEncoderLayer(d_model=512, nhead=num_heads)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    def forward(self, x1, edge_index, edge_attr, batch):
        # Process edge attributes


        edge_attr1 = self.fc1(edge_attr)
        edge_attr2 = self.fc2(edge_attr)
        edge_attr3 = self.fc3(edge_attr)

        # GIN layers for local graph feature extraction
        x1 = self.gin1(x1, edge_index, edge_attr1)
       # print(f'x1shape = {x1.shape}')
        x1 = F.elu(x1)

        x1 = self.gin2(x1, edge_index, edge_attr2)
        x1 = F.elu(x1)

        x1 = self.gin3(x1, edge_index, edge_attr3)
        x1 = F.elu(x1)

        # Apply Transformer to capture global relationships
        x1 = x1.unsqueeze(0)  # Add batch dimension for Transformer
        x1 = self.transformer(x1)

        # Global mean pooling to aggregate node features
        x_mean = gmp(x1.squeeze(0), batch)  # Remove batch dimension after transformer

        return x_mean



class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=93, output_dim=512, heads=10, edge_dim=11, dropout=0.2, num_heads=8, num_layers=2):
        super(GATNet, self).__init__()

        # GAT layers for local feature extraction
        self.gat1 = GATConv(num_features_xd, output_dim, heads=heads, edge_dim=edge_dim, dropout=dropout)
        self.gat2 = GATConv(output_dim * heads, output_dim, heads=1, edge_dim=edge_dim, dropout=dropout)

        # FC layers to process edge attributes
        self.fc1 = Linear(11, 93).cuda()
        self.fc2 = Linear(11, 512).cuda()

        # Transformer Encoder (captures global dependencies)
        self.transformer_layer = TransformerEncoderLayer(d_model=output_dim, nhead=num_heads)
        self.transformer = TransformerEncoder(self.transformer_layer, num_layers=num_layers)

    def forward(self, x1, edge_index, edge_attr, batch):
        # Process edge attributes
        x1, weight = self.gat1(x1, edge_index, edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x1, weight2 = self.gat2(x1, edge_index, edge_attr, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        # Apply Transformer to capture global dependencies
        x1 = x1.unsqueeze(0)  # Add batch dimension for Transformer
        x1 = self.transformer(x1)

        # Global mean pooling to aggregate node features
        x_mean = gmp(x1.squeeze(0), batch)  # Remove batch dimension after transformer

        return x_mean, weight2
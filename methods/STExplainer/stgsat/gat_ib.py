import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

from .conv_layers import GATConv
from Params import args, logger


class gat_ib(nn.Module):
    def __init__(self, in_channels, out_channels, heads ):
        super().__init__()        
        self.use_edge_attr = args.use_edge_attr
        self.dropout_p = args.gib_drop
        
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        self.convs.append(GATConv(in_channels, out_channels, heads[0], dropout = self.dropout_p, edge_dim = 1 ))
        self.convs.append(GATConv(out_channels*heads[0], out_channels, heads[1], dropout = self.dropout_p, edge_dim = 1 ))  

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        h = x
        weights = []
        for i, layer in enumerate(self.convs):
            h, weight = layer(h, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            weights.append(weight)
            # if i == 0:
            #     print(weight)
        return h, weights

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        h = x
        weights = []
        for i, layer in enumerate(self.convs):
            # print('h: {}'.format(h.shape))
            h, weight = layer(h, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            weights.append(weight)

            # if i == 0:
            #     print(weight)
        return h, weights

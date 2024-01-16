import dgl 
from torch.nn import functional as F
import dgl.nn as dglnn
import torch
import torch.nn as nn
import dgl.nn.functional as fn

class GNN(nn.Module):
    def __init__(self, n_layer, in_dim, hid_dim, out_dim) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layer = n_layer
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.device = torch.device('cpu')

    def infer(self, block, x, layer):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        if isinstance(args[0], torch.device):
            self.device = args[0]
        return super().to(*args, **kwargs)

class SAGE(GNN):
    def __init__(self, n_layer, in_dim, hid_dim, out_dim) -> None:
        super().__init__(n_layer, in_dim, hid_dim, out_dim)
        for _ in range(n_layer - 1):
            self.layers.append(dglnn.SAGEConv(in_dim, hid_dim, 'mean', activation=F.relu))
            in_dim = hid_dim
        self.layers.append(dglnn.SAGEConv(in_dim, out_dim, 'mean'))
    
    def infer(self, block, x, layer):
        return self.layers[layer](block, x)


class HGT(GNN):
    def __init__(self, n_layer, in_dim, hid_dim, out_dim) -> None:
        super().__init__(n_layer, in_dim, hid_dim, out_dim)
        self.layers.append(dglnn.HGTConv(in_dim, hid_dim, out_dim, num_heads=4, feat_drop=0.2, attn_drop=0.2))



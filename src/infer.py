import dgl
import torch
torch.set_grad_enabled(False)

def infer(graph: dgl.DGLGraph, model):
    model.eval()
    num_local_nodes = graph.ndata['inner_node'].sum()
    for l in range(model.n_layer):
        y = torch.empty((num_local_nodes, model.hid_dim), dtype=torch.float32)

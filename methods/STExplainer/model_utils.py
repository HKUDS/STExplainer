"""Utility functions."""
import torch
from torch_geometric.utils import degree, softmax, subgraph
from torch_sparse import SparseTensor
from torch_geometric.utils import remove_self_loops, add_self_loops
import sys, os
import os.path as osp
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from methods.STExplainer.pytorch_net.util import to_np_array
import scipy.sparse as sp
import numpy as np
import pdb
import pickle



def get_reparam_num_neurons(out_channels, reparam_mode):
    if reparam_mode is None or reparam_mode == "None":
        return out_channels
    elif reparam_mode == "diag":
        return out_channels * 2
    elif reparam_mode == "full":
        return int((out_channels + 3) * out_channels / 2)
    else:
        raise "reparam_mode {} is not valid!".format(reparam_mode)


def sample_lognormal(mean, sigma=None, sigma0=1.):
    """
    Samples from a log-normal distribution using the reparametrization
    trick so that we can backprogpagate the gradients through the sampling.
    By setting sigma0=0 we make the operation deterministic (useful at testing time)
    """
    e = torch.randn(mean.shape).to(sigma.device)
    return torch.exp(mean + sigma * sigma0 * e)

def scatter_sample(src, index, temperature, num_nodes=None):
    gumbel = torch.distributions.Gumbel(torch.tensor([0.]).to(src.device), 
            torch.tensor([1.0]).to(src.device)).sample(src.size()).squeeze(-1)
    log_prob = torch.log(src+1e-16)
    logit = (log_prob + gumbel) / temperature
    return softmax(logit, index, num_nodes)

def uniform_prior(index):
    deg = degree(index)
    deg = deg[index]
    return 1./deg.unsqueeze(1)

def add_distant_neighbors(data, hops):
    """Add multi_edge_index attribute to data which includes the edges of 2,3,... hops neighbors."""
    assert hops > 1
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index, _ = add_self_loops(edge_index,
                                   num_nodes=data.x.size(0))
    one_hop_set = set([tuple(x) for x in edge_index.transpose(0, 1).tolist()])
    row, col = edge_index
    adj = SparseTensor(row=row, col=col)
    multi_adj = adj
    for _ in range(hops - 1):
        multi_adj = multi_adj @ adj
    row, col, _ = multi_adj.coo()
    multi_edge_index = torch.stack([row, col], dim=0)
    multi_hop_set = set([tuple(x) for x in multi_edge_index.transpose(0, 1).tolist()])
    multi_hop_set = multi_hop_set - one_hop_set
    multi_edge_index = torch.LongTensor(list(multi_hop_set)).transpose(0, 1)
    data.multi_edge_index = multi_edge_index
    return


def compose_log(metrics, key, spaces=0, tabs=0, newline=False):
    string = "\n" if newline else ""
    return string + "\t" * tabs + " " * spaces + "{}: ({:.4f}, {:.4f}, {:.4f})".format(key, metrics["train_{}".format(key)], metrics["val_{}".format(key)], metrics["test_{}".format(key)])


def edge_index_2_csr(edge_index, size):
    """Edge index (PyG COO format) transformed to csr format."""
    csr_matrix = sp.csr_matrix(
        (np.ones(edge_index.shape[1]), to_np_array(edge_index)),
        shape=(size, size))
    return csr_matrix


def process_data_for_nettack(data):
    data.features = sp.csr_matrix(to_np_array(data.x))
    data.adj = edge_index_2_csr(data.edge_index, size=data.x.shape[0])
    data.labels = to_np_array(data.y)
    data.idx_train = np.where(to_np_array(data.train_mask))[0]
    data.idx_val = np.where(to_np_array(data.val_mask))[0]
    data.idx_test = np.where(to_np_array(data.test_mask))[0]
    return data


def to_tuple_list(edge_index):
    """Transform a coo-format edge_index to a list of edge tuples."""
    return [tuple(item) for item in edge_index.T.cpu().numpy()]


def classification_margin(output, true_label):
    '''probs_true_label - probs_best_second_class'''
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()




def to_inductive(data):
    mask = data.train_mask | data.val_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.val_mask = data.val_mask[mask]
    data.test_mask = None
    data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data









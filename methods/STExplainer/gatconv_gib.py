from typing import Optional
from torch import Tensor
from torch_scatter import scatter, segment_csr

from torch_geometric.nn.conv.utils.helpers import expand_left

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.distributions.normal import Normal

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import MessagePassing
from .pytorch_net.net import reparameterize, Mixture_Gaussian_reparam
# from .pytorch_net.util import sample, to_np_array
from .model_utils import get_reparam_num_neurons, sample_lognormal, scatter_sample, uniform_prior
from numbers import Number
import numpy as np
from torch.autograd import Variable
def sample(dist, n=None):
    if n is None:
        return dist.rsample()
    else:
        return dist.rsample((n,))
def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if array is None:
            array_list.append(array)
            continue
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        if not ("keep_list" in kwargs and kwargs["keep_list"]):
            array_list = array_list[0]
    return array_list


class GATConv(MessagePassing):
    
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, reparam_mode=None, prior_mode=None,
                 struct_dropout_mode=None, sample_size=1,
                 val_use_mean=True,
                 bias=True,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add',node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.reparam_mode = reparam_mode if reparam_mode != "None" else None
        self.prior_mode = prior_mode
        self.out_neurons = get_reparam_num_neurons(out_channels, self.reparam_mode)
        self.struct_dropout_mode = struct_dropout_mode
        self.sample_size = sample_size
        self.val_use_mean = val_use_mean

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * self.out_neurons))
        self.att = Parameter(torch.Tensor(1, heads, 2 * self.out_neurons))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * self.out_neurons))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.out_neurons))
        else:
            self.register_parameter('bias', None)
            
        if self.reparam_mode is not None:
            if self.prior_mode.startswith("mixGau"):
                n_components = eval(self.prior_mode.split("-")[1])
                self.feature_prior = Mixture_Gaussian_reparam(is_reparam=False, Z_size=self.out_channels, n_components=n_components)

        self.skip_editing_edge_index = struct_dropout_mode[0] == 'DNsampling'
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        
        if ptr is not None:
            ptr = expand_left(ptr, dim=self.node_dim, dims=inputs.dim())
            return segment_csr(inputs, ptr, reduce=self.aggr)
        else:
            # print('inputs:{}, index:{}'.format(inputs.shape, index.shape))
            out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size,
                           reduce=self.aggr)
            # print('out :{}'.format(out.shape))
            return out

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x) and not self.skip_editing_edge_index:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=x.size(self.node_dim))
        # x = (x, x)
        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        # print('edge_index :{}, size :{}, x :{}'.format(edge_index.shape, size, x.shape))
        # print('edge_index max: {}, edge_index min: {}'.format(torch.max(edge_index), torch.min(edge_index)))
        # print(edge_index.shape)
        
        # TODO: pass
        out = self.propagate(edge_index, size=size, x=x)
        # TODO: pass
        # print('out propagate:', out.shape)
        # print('reparam_mode: {}'.format(self.reparam_mode))
        if self.reparam_mode is not None:
            # Reparameterize:
            
            out = out.view(-1, self.out_neurons)
            self.dist, _ = reparameterize(model=None, input=out,
                                          mode=self.reparam_mode,
                                          size=self.out_channels,
                                         )  # dist: [B * head, Z]
            Z_core = sample(self.dist, self.sample_size)  # [S, B * head, Z]
            Z = Z_core.view(self.sample_size, -1, self.heads * self.out_channels)  # [S, B, head * Z]

            if self.prior_mode == "Gaussian":
                self.feature_prior = Normal(loc=torch.zeros(out.size(0), self.out_channels).to(x.device),
                                            scale=torch.ones(out.size(0), self.out_channels).to(x.device),
                                           )  # feature_prior: [B * head, Z]

            if self.reparam_mode == "diag" and self.prior_mode == "Gaussian":
                ixz = torch.distributions.kl.kl_divergence(self.dist, self.feature_prior).sum(-1).view(-1, self.heads).mean(-1)
            else:
                Z_logit = self.dist.log_prob(Z_core).sum(-1) if self.reparam_mode.startswith("diag") else self.dist.log_prob(Z_core)  # [S, B * head]
                prior_logit = self.feature_prior.log_prob(Z_core).sum(-1)  # [S, B * head]
                # upper bound of I(X; Z):
                ixz = (Z_logit - prior_logit).mean(0).view(-1, self.heads).mean(-1)  # [B]
            self.Z_std = to_np_array(Z.std((0, 1)).mean())
            if self.val_use_mean is False or self.training:
                out = Z.mean(0)
            else:
                out = out[:, :self.out_channels].contiguous().view(-1, self.heads * self.out_channels)
        else:
            ixz = torch.zeros(x.size(0)).to(x.device)

        if "Nsampling" in self.struct_dropout_mode[0]:
            if 'categorical' in self.struct_dropout_mode[1]:
                structure_kl_loss = torch.sum(self.alpha*torch.log((self.alpha+1e-16)/self.prior))
            elif 'Bernoulli' in self.struct_dropout_mode[1]:
                posterior = torch.distributions.bernoulli.Bernoulli(self.alpha)
                prior = torch.distributions.bernoulli.Bernoulli(self.prior) 
                structure_kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).sum(-1).mean()
            else:
                raise Exception("I think this belongs to the diff subset sampling that is not implemented")
        else:
            structure_kl_loss = torch.zeros([]).to(x.device)
        # if self.reparam_mode
        
        # TODO: out pass; ixz pass; structure_kl_loss pass
        # if self.reparam_mode is None:
        #     print(out)

        alpha_norm = self._alpha_norm
        assert alpha_norm is not None
        self._alpha_norm = None
        return out, ixz, structure_kl_loss, alpha_norm

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_neurons)  # [N_edge, heads, out_channels]
        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_neurons:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_neurons)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [N_edge, heads]

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Sample attention coefficients stochastically.
        if self.struct_dropout_mode[0] == "None":
            alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
        else:
            if self.struct_dropout_mode[0] == "standard":
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                prob_dropout = self.struct_dropout_mode[1]
                alpha = F.dropout(alpha, p=prob_dropout, training=self.training)
            elif self.struct_dropout_mode[0] == "identity":
                alpha = torch.ones_like(alpha)
                alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
            elif self.struct_dropout_mode[0] == "info":
                mode = self.struct_dropout_mode[1]
                if mode == "lognormal":
                    max_alpha = self.struct_dropout_mode[2] if len(self.struct_dropout_mode) > 2 else 0.7
                    alpha = 0.001 + max_alpha * alpha
                    self.kl = -torch.log(alpha/(max_alpha + 0.001))
                    sigma0 = 1. if self.training else 0.
                    alpha = sample_lognormal(mean=torch.zeros_like(alpha), sigma=alpha, sigma0=sigma0)
                else:
                    raise Exception("Mode {} for the InfoDropout is invalid!".format(mode))
            elif "Nsampling" in self.struct_dropout_mode[0]:
                neighbor_sampling_mode = self.struct_dropout_mode[1]
                # print('Nsampling mode: {}'.format(neighbor_sampling_mode))
                if 'categorical' in neighbor_sampling_mode:
                    alpha = softmax(alpha, edge_index_i, num_nodes=size_i)
                    self.alpha = alpha
                    self.prior = uniform_prior(edge_index_i)
                    if self.val_use_mean is False or self.training:
                        temperature = self.struct_dropout_mode[2]
                        sample_neighbor_size = self.struct_dropout_mode[3]
                        if neighbor_sampling_mode == 'categorical':
                            alpha = scatter_sample(alpha, edge_index_i, temperature, size_i)
                        elif 'multi-categorical' in neighbor_sampling_mode:
                            alphas = []
                            for _ in range(sample_neighbor_size): #! this can be improved by parallel sampling
                                alphas.append(scatter_sample(alpha, edge_index_i, temperature, size_i))
                            alphas = torch.stack(alphas, dim=0)
                            if 'sum' in neighbor_sampling_mode:
                                alpha = alphas.sum(dim=0)
                            elif 'max' in neighbor_sampling_mode:
                                alpha, _ = torch.max(alphas, dim=0)
                            else:
                                raise
                        else:
                            raise
                elif neighbor_sampling_mode == 'Bernoulli':
                    if self.struct_dropout_mode[4] == 'norm':
                        alpha_normalization = torch.ones_like(alpha)
                        alpha_normalization = softmax(alpha_normalization, edge_index_i, num_nodes=size_i)
                    alpha = torch.clamp(torch.sigmoid(alpha), 0.01, 0.99)
                    self.alpha = alpha
                    self.prior = (torch.ones_like(self.alpha)*self.struct_dropout_mode[3]).to(alpha.device)
                    if not self.val_use_mean or self.training:
                        temperature = self.struct_dropout_mode[2]
                        alpha = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.Tensor([temperature]).to(alpha.device),
                            probs=alpha).rsample()
                    if self.struct_dropout_mode[4] == 'norm':
                        self._alpha_norm = alpha
                        alpha = alpha*alpha_normalization
                        
                else:
                    raise
            else:
                raise

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_neurons)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


    def to_device(self, device):
        self.to(device)
        return self

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
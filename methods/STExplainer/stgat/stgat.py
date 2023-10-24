import torch
import torch.nn as nn
from .MLP import MLP_res
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import coo_matrix
from torch_geometric.utils import dense_to_sparse
from .gat import GAT


from Params import args, logger

class STGAT(nn.Module):
    def __init__(self, sp_adj, sp_adj_w, temp_adj):
        super().__init__()
        # attributes
        self.num_nodes  = args.num_nodes
        self.node_dim   = args.hidden_size
        self.input_len  = args.lag
        self.input_dim  = 3
        self.embed_dim  = args.hidden_size
        self.output_len = args.horizon
        self.num_layer  = 3
        self.temp_dim_tid   = args.hidden_size
        self.temp_dim_diw   = args.hidden_size

        self.if_T_i_D = True
        self.if_D_i_W = True
        self.if_node  = True

        # spatial embeddings
        if self.if_node:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_T_i_D:
            self.T_i_D_emb  = nn.Parameter(torch.empty(288, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.T_i_D_emb)
        if self.if_D_i_W:
            self.D_i_W_emb  = nn.Parameter(torch.empty(7, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.D_i_W_emb)

        # embedding layer 
        self.time_series_emb_layer = nn.Conv2d(in_channels=args.d_model * self.input_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)

        # encoding
        if args.only_spat:
            num_embed = 1
        else:
            num_embed = 2
        self.hidden_dim = self.embed_dim*num_embed+self.node_dim*int(self.if_node)+self.temp_dim_tid*int(self.if_D_i_W) + self.temp_dim_diw*int(self.if_T_i_D)
        self.encoder = nn.Sequential(*[MLP_res(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1,1), bias=True)
        self.init_st_graph(sp_adj, sp_adj_w, temp_adj)
        self.start_fc = nn.Linear(in_features=self.input_dim, out_features=args.d_model)
        
        self.spat_gsat = GAT(args.d_model_spat, args.d_model_spat, args.head)
        
        
        self.temp_gsat= GAT(args.d_model_temp, args.d_model_temp, args.head)
        self.TransSpat = nn.Linear(in_features=args.d_model*args.lag, out_features=args.d_model_spat)
        self.InverseTransSpat = nn.Linear(in_features=args.d_model_spat, out_features=args.d_model*args.lag)
        self.TransTemp = nn.Linear(in_features=args.d_model*args.num_nodes, out_features=args.d_model_temp)
        self.InverseTransTemp = nn.Linear(in_features=args.d_model_temp, out_features=args.d_model*args.num_nodes)
        # self.temporal_gib = GATIB(num_layers=2)

    def forward(self, history_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """feed forward.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        self.reg_info = {}
        X = history_data[..., range(self.input_dim)] # 
        t_i_d_data   = history_data[..., 1] # B, L, N
        d_i_w_data   = history_data[..., 2] # B, L, N
        epoch = kwargs.get('epoch')

        if self.if_T_i_D:
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data[:, -1, :]).type(torch.LongTensor)]    # [B, N, D]
        else:
            T_i_D_emb = None
        if self.if_D_i_W:
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data[:, -1, :]).type(torch.LongTensor)]          # [B, N, D]
        else:
            D_i_W_emb = None

        # time series embedding
        B, L, N, _ = X.shape # B, L, N, 3
        X = self.start_fc(X) # B, L, N, D
        time_series_emb = []
        res_x = X
        res_x = res_x.transpose(2, 1)
        res_x = res_x.reshape(B, N, -1).transpose(1, 2).unsqueeze(-1)      # B, L*d_model, N, 1
        time_series_emb_res = self.time_series_emb_layer(res_x)         # B, D, N, 1
        time_series_emb.append(time_series_emb_res)

        temp_out = self.batch_st_gat(X, epoch)
        
        # if args.only_spat is False:
        temp_out = temp_out.transpose(2, 1)
        temp_out = temp_out.reshape(B, N, -1).transpose(1, 2).unsqueeze(-1)      # B, L*d_model, N, 1
        time_series_emb_temp = self.time_series_emb_layer(temp_out)
        time_series_emb.append(time_series_emb_temp) 
        node_emb = []
        if self.if_node:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1))  # B, D, N, 1
        # temporal embeddings
        tem_emb  = []
        if T_i_D_emb is not None:
            tem_emb.append(T_i_D_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        if D_i_W_emb is not None:
            tem_emb.append(D_i_W_emb.transpose(1, 2).unsqueeze(-1))                     # B, D, N, 1
        
        # concate all embeddings
        hidden = torch.cat(time_series_emb + node_emb + tem_emb, dim=1)

        # encoding
        hidden = self.encoder(hidden)

        # regression
        prediction = self.regression_layer(hidden)

        return prediction
    def init_st_graph(self, sp_adj, sp_adj_w, tem_adj):
        self.edge_idx_spa, self.edge_wg_spa = dense_to_sparse(torch.from_numpy(sp_adj_w))
        tem_adj = np.ones((args.lag, args.lag))
        self.edge_idx_temp, self.edge_wg_temp = dense_to_sparse(torch.from_numpy(tem_adj))
        self.edge_idx_spa = self.edge_idx_spa.to(args.device)
        self.edge_wg_spa = self.edge_wg_spa.to(args.device)
        self.edge_idx_temp = self.edge_idx_temp.to(args.device)
        self.edge_wg_temp = self.edge_wg_temp.to(args.device)
    
    def batch_st_gat(self, inputs, epoch):
        # inputs: B, L, N, D
        # return: B, L, N, D
        batch_size = inputs.shape[0]
        
        spa_inp = inputs.transpose(2, 1).reshape(batch_size, args.num_nodes, -1) 
        # spat gib
        spa_inp = self.TransSpat(spa_inp)# B, N, SpatD
        # edge_index, inputs, epoch, training, batch, edge_attr = None
        spa_out_bch = []
        # batch = torch.zeros([args.num_nodes], dtype=torch.int64).to(args.device)
        batch = None
        for bch_idx in range(batch_size):
            spa_out = self.spat_gsat(spa_inp[bch_idx], self.edge_idx_spa, batch = batch)
            spa_out_bch.append(spa_out)
        spa_out_bch = torch.stack(spa_out_bch, dim = 0)
        spa_out_bch = self.InverseTransSpat(spa_out_bch).reshape(batch_size, args.num_nodes, args.lag, -1) 
        spa_out_bch = spa_out_bch.transpose(2, 1)
        # temp gib
        temp_in = spa_out_bch.reshape(batch_size, args.lag, -1)
        temp_in = self.TransTemp(temp_in)
        temp_out_bch = []
        # batch = torch.zeros([args.lag], dtype=torch.int64).to(args.device)
        batch = None
        for bch_idx in range(batch_size):
            temp_out = self.temp_gsat(temp_in[bch_idx], self.edge_idx_temp, batch = batch)
            temp_out_bch.append(temp_out)
        temp_out_bch = torch.stack(temp_out_bch, dim = 0)
        temp_out_bch = self.InverseTransTemp(temp_out_bch).reshape(batch_size, args.lag, args.num_nodes, -1)
        return temp_out_bch
        
        


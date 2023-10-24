import torch.nn as nn
import torch.nn.functional as F
import torch
from .gatconv_gib import GATConv
from Params import args
from .pytorch_net.util import record_data
class GATIB(nn.Module):
    def __init__(self, num_layers, d_model):
        super().__init__()
        self.num_layers = num_layers
        self.GATlayers = nn.ModuleList()
        if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
            self.GATlayers_multi = nn.ModuleList()
        self.reparam_layers = []

        for idx in range(self.num_layers):
            if idx == 0:
                input_size = d_model # 16
            else:
                if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                    input_size = d_model * 8 * 2
                else:
                    input_size = d_model * 8
            if args.reparam_all_layers is True:
                is_reparam = True
            elif isinstance(args.reparam_all_layers, tuple):
                reparam_all_layers = tuple([kk + self.num_layers if kk < 0 else kk for kk in args.reparam_all_layers])
                is_reparam = idx in reparam_all_layers
            else:
                raise
            if is_reparam:
                self.reparam_layers.append(idx)
            self.GATlayers.append(GATConv(
                input_size,
                d_model ,
                heads=8 if idx != self.num_layers - 1 else 1, concat=True,
                reparam_mode=args.reparam_mode if is_reparam else None,
                prior_mode=args.prior_mode if is_reparam else None,
                val_use_mean=args.val_use_mean,
                struct_dropout_mode=args.struct_dropout_mode,
                sample_size=args.sample_size,
            ))
            if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                self.GATlayers_multi.append(GATConv(
                    input_size,
                    d_model ,
                    heads=8 if idx != self.num_layers - 1 else 1, concat=True,
                    reparam_mode=args.reparam_mode if is_reparam else None,
                    prior_mode=args.prior_mode if is_reparam  else None,
                    val_use_mean=args.val_use_mean,
                    struct_dropout_mode=args.struct_dropout_mode,
                    sample_size=args.sample_size,
                ))

    def forward(self, reg_info, inputs, edge_index, multi_edge_index = None):
        # inputs: B, L*N, D
        # 2, L*N
        # graph: L*N, D
        batch_size = inputs.shape[0]
        x = inputs
        for i in range(self.num_layers - 1):   
            if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                x_lm_bch, ixz_lm_bch, structure_kl_loss_lm_bch = [], [], []
                for bch in range(batch_size):
                    x_bch = x[bch] # L*N, D
                    x_lm, ixz_lm, structure_kl_loss_lm = self.GATlayers_multi[i](x_bch, multi_edge_index)
                    x_lm_bch.append(x_lm)
                    ixz_lm_bch.append(ixz_lm)
                    structure_kl_loss_lm_bch.append(structure_kl_loss_lm)
                x_lm_bch = torch.stack(x_lm_bch, dim=0)
                ixz_lm_bch = torch.stack(ixz_lm_bch, dim=0)
                structure_kl_loss_lm_bch = torch.stack(structure_kl_loss_lm_bch, dim=0)
            
            layer = self.GATlayers[i]
            x_l_bch, ixz_l_bch, structure_kl_loss_l_bch, alpha_norm_bch = [], [], [], []
            for bch in range(batch_size):
                x_bch = x[bch] # L*N, D
                x_l, ixz_l, structure_kl_loss_l, alpha_norm = layer(x_bch, edge_index)
                x_l_bch.append(x_l)
                ixz_l_bch.append(ixz_l)
                structure_kl_loss_l_bch.append(structure_kl_loss_l)
                if i == 0:
                    # print(alpha_norm.shape)
                    alpha_norm_bch.append(alpha_norm)
            x_l_bch = torch.stack(x_l_bch, dim=0)
            ixz_l_bch = torch.stack(ixz_l_bch, dim=0)
            structure_kl_loss_l_bch = torch.stack(structure_kl_loss_l_bch, dim=0)
            alpha_norm_bch = torch.stack(alpha_norm_bch, dim=0).mean(-1).mean(0)
            # Record:
            record_data(reg_info, [ixz_l_bch, structure_kl_loss_l_bch], ["ixz_list", "structure_kl_list"])

            # Multi-hop:
            x = x_l_bch
            if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
                x = torch.cat([x, x_lm_bch], dim=-1)
                record_data(reg_info, [ixz_lm_bch, structure_kl_loss_lm_bch], ["ixz_DN_list", "structure_kl_DN_list"])
            x = F.elu(x)
            # x = F.dropout(x, p=0.0, training=self.training)
        ## last layer, input: x
        # multi-hop
        if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
            x_lm_bch, ixz_lm_bch, structure_kl_loss_lm_bch = [], [], []
            for bch in range(batch_size):
                x_bch = x[bch] # L*N, D
                x_lm, ixz_lm, structure_kl_loss_lm = self.GATlayers_multi[self.num_layers-1](x_bch, multi_edge_index)
                x_lm_bch.append(x_lm)
                ixz_lm_bch.append(ixz_lm)
                structure_kl_loss_lm_bch.append(structure_kl_loss_lm)
            x_lm_bch = torch.stack(x_lm_bch, dim=0)
            ixz_lm_bch = torch.stack(ixz_lm_bch, dim=0)
            structure_kl_loss_lm_bch = torch.stack(structure_kl_loss_lm_bch, dim=0)
        # one-hop:
        layer = self.GATlayers[self.num_layers-1]
        x_l_bch, ixz_l_bch, structure_kl_loss_l_bch = [], [], []
        for bch in range(batch_size):
            x_bch = x[bch] # L*N, D
            x_l, ixz_l, structure_kl_loss_l, alpha_norm = layer(x_bch, edge_index)
            x_l_bch.append(x_l)
            ixz_l_bch.append(ixz_l)
            structure_kl_loss_l_bch.append(structure_kl_loss_l)
        x_l_bch = torch.stack(x_l_bch, dim=0)
        ixz_l_bch = torch.stack(ixz_l_bch, dim=0)
        structure_kl_loss_l_bch = torch.stack(structure_kl_loss_l_bch, dim=0)
        x = x_l_bch
        # Record:
        record_data(reg_info, [ixz_l_bch, structure_kl_loss_l_bch], ["ixz_list", "structure_kl_list"])
        
        # Multi-hop:
        if args.struct_dropout_mode[0] == 'DNsampling' or (args.struct_dropout_mode[0] == 'standard' and len(args.struct_dropout_mode) == 3):
            x = x + x_lm_bch
            record_data(reg_info, [ixz_lm_bch, structure_kl_loss_lm_bch], ["ixz_DN_list", "structure_kl_DN_list"])
        # x: B, L*N, D
        outputs = x
        return outputs, alpha_norm_bch
        


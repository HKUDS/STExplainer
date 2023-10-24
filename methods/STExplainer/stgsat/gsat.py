import torch
import torch.nn as nn
from torch_geometric.utils import is_undirected
from torch_sparse import transpose
from Params import args, logger
class GSAT(nn.Module):
    def __init__(self, conv_k, extractor):
        super().__init__()
        self.conv_k = conv_k
        self.extractor = extractor
        self.learn_edge_att = args.learn_edge_att
        self.fix_r = args.fix_r
        self.decay_interval = args.decay_interval
        self.decay_r = args.decay_r
        self.final_r = args.final_r
        self.init_r = args.init_r

    def __loss__(self, att, epoch):
        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()        
        return info_loss

    def forward(self, edge_index, inputs, epoch, training, batch, edge_attr = None):
        r"""
        edge_index: 2, num_node
        inputs: num_node, feats
        """        
        emb, edge_weights = self.conv_k.get_emb(inputs, edge_index, batch=batch, edge_attr=edge_attr)
        att_log_logits = self.extractor(emb, edge_index, batch)
        att = self.sampling(att_log_logits, epoch, training)
        if self.learn_edge_att:
            if is_undirected(edge_index):
                nodesize = inputs.shape[0]
                edge_att = (att + transpose(edge_index, att, nodesize, nodesize, coalesced=False)[1]) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, edge_index)

        sp_emb, feat_weights = self.conv_k(inputs, edge_index, batch, edge_attr=edge_attr, edge_atten=edge_att)
        loss = self.__loss__(att, epoch)
        return edge_att, loss, sp_emb, edge_weights, feat_weights
    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
import torch.nn as nn
import torch
from torch_geometric.nn import InstanceNorm
from Params import args

class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                # print('inputs: {}, batch: {}'.format(inputs.shape, batch.shape))
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs

class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                # m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.learn_edge_att = args.learn_edge_att
        dropout_p = args.extractor_drop

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            # print(batch[col])

            # att_log_logits = self.feature_extractor(f12, batch[col])
            att_log_logits = self.feature_extractor(f12, batch)
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits
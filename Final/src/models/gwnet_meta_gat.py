import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionWeightGenerator(nn.Module):
    def __init__(self, meta_dim, hidden_dim, out_dim):
        super(AttentionWeightGenerator, self).__init__()
        self.fc1 = nn.Linear(meta_dim, hidden_dim).double()
        self.fc2 = nn.Linear(hidden_dim, 2 * out_dim).double()

    def forward(self, pair_metadata):
        # Generate attention weights from metadata
        attn_weights = self.fc1(pair_metadata)
        attn_weights = self.fc2(attn_weights)
        return attn_weights


class MetaGAT(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(MetaGAT, self).__init__()
        self.weight = AttentionWeightGenerator(in_features, hidden_size, out_features)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, src_node, dst_node, pair_metadata):
        meta_knowledge = self.weight(pair_metadata)
        meta_knowledge = meta_knowledge.repeat(64, 1, 1)
        state = torch.cat((src_node, dst_node), dim=1).squeeze(2).double()

        raw_attention = self.leaky_relu(torch.bmm(meta_knowledge, state))
        return raw_attention

class GWNET(BaseModel):
    '''
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    '''
    def __init__(self, supports, adp_adj, adp_adj_meta, dst_indices, dropout, residual_channels, dilation_channels, \
                 skip_channels, end_channels, kernel_size=2, blocks=4, layers=2, **args):
        super(GWNET, self).__init__(**args)

        self.relu = nn.ReLU(inplace=True)
        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj
        self.adp_adj_meta = adp_adj_meta

        self.dst_indices = dst_indices

        if adp_adj:
            self.nodevec1 = nn.Parameter(torch.randn(self.node_num, 10), requires_grad=True)
            self.nodevec2 = nn.Parameter(torch.randn(10, self.node_num), requires_grad=True)
            self.supports_len += 1

        print('check supports length', len(supports), self.supports_len)
        
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        self.metagat = MetaGAT(10, residual_channels, residual_channels)
        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1,kernel_size), dilation=new_dilation))

                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1,1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(GCN(dilation_channels, residual_channels, self.dropout, support_len=self.supports_len))


        self.receptive_field = receptive_field
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.output_dim * self.horizon,
                                    kernel_size=(1,1),
                                    bias=True)


    def forward(self, input, label=None):  # (b, t, n, f)
        input = input.transpose(1,3)
        in_len = input.size(3)

        if in_len < self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        if self.adp_adj:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]
        else:
            new_supports = self.supports

        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)

            sum_final_prime = []
            for node in range(self.node_num):
                softmax_attention = []
                src_node = x[:, :, node, :].unsqueeze(2)
                index = self.dst_indices[node]
                for number in index:
                    dst_node = x[:, :, number, :].unsqueeze(2)
                    pair_metadata = torch.cat((torch.tensor(self.adp_adj_meta[node]), torch.tensor(self.adp_adj_meta[number])), dim=-1).double().cuda()
                    softmax_attention.append(self.metagat(src_node, dst_node, pair_metadata))
                softmax_attention = torch.cat(softmax_attention, dim=1)
                normalized_attention = F.softmax(softmax_attention, dim=1)
                scores = []
                for j, number in enumerate(index):
                    dst_node = x[:, :, number, :]
                    score = normalized_attention[:, j, :].unsqueeze(1) * dst_node
                    scores.append(score)

                stacked_scores = torch.stack(scores)
                summed_scores = torch.sum(stacked_scores, dim=0)
                final_prime = 0.5 * src_node.squeeze(2) + self.relu(0.5 * summed_scores)
                sum_final_prime.append(final_prime)

            x = torch.stack(sum_final_prime, dim=0).float()
            x = x.permute(1, 2, 0, 3)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()


    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)


    def forward(self,x):
        return self.mlp(x)

    
class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order


    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
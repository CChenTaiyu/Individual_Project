import torch
import torch.nn as nn
import torch.nn.functional as F
from src.base.model import BaseModel

class NMKLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NMKLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightGenerator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)



class GWNET(BaseModel):
    '''
    Reference code: https://github.com/nnzhan/Graph-WaveNet
    '''
    def __init__(self, supports, adp_adj, adp_adj_meta, dst_indices, dropout, residual_channels, dilation_channels, \
                 skip_channels, end_channels, kernel_size=2, blocks=4, layers=2, **args):
        super(GWNET, self).__init__(**args)

        self.supports = supports
        self.supports_len = len(supports)
        self.adp_adj = adp_adj
        self.adp_adj_meta = adp_adj_meta

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

        # Meta learning components
        self.nmk_learner = NMKLearner(5, 32, 32)
        filter_weight_shape = dilation_channels * residual_channels * kernel_size
        gate_weight_shape = dilation_channels * residual_channels * kernel_size
        self.filter_weight_generator = WeightGenerator(32, filter_weight_shape)
        self.gate_weight_generator = WeightGenerator(32, gate_weight_shape)

        # Process metadata to generate meta knowledge
        meta_knowledge = self.nmk_learner(torch.tensor(self.adp_adj_meta).float())

        # Generate weights using meta knowledge
        self.filter_weights, self.gate_weights = self.generate_weights(meta_knowledge, self.filter_convs[0])

    def generate_weights(self, meta_knowledge, conv_layer):
        # Generate weights for the convolutional layers
        filter_weights = self.filter_weight_generator(meta_knowledge).view(self.node_num, conv_layer.out_channels, conv_layer.in_channels, 1, conv_layer.kernel_size[1])
        gate_weights = self.gate_weight_generator(meta_knowledge).view(self.node_num, conv_layer.out_channels, conv_layer.in_channels, 1, conv_layer.kernel_size[1])
        return filter_weights, gate_weights

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
            output = []
            for node in range(self.node_num):
                node_filter_weights = self.filter_weights[node].cuda()
                node_gate_weights = self.gate_weights[node].cuda()
                self.filter_convs[i].weight.data = node_filter_weights.clone()
                self.gate_convs[i].weight.data = node_gate_weights.clone()
                node_residual = residual[:, :, node, :].unsqueeze(2)
                filter = self.filter_convs[i](node_residual)
                gate = self.gate_convs[i](node_residual)

                filter = torch.tanh(filter)
                gate = torch.sigmoid(gate)
                node_output = filter * gate

                output.append(node_output)
            x = torch.cat(output, dim=2)

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, new_supports)
            
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
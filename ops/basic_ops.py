# -*- coding:utf-8 -*-

import torch
import math
from torch import nn
import torch.nn.functional as f


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            # print("input_tensor: ", input_tensor.shape)
            output = input_tensor.mean(dim=self.dim, keepdim=True)
            # print("output: ", output.shape)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        # print("consensus_type: ", self.consensus_type)
        # print("dim: ", self.dim)
        # print("input: ", input.shape)
        output = SegmentConsensus(self.consensus_type, self.dim).forward(input)
        # print("output: ", output.shape)

        return output
        # return SegmentConsensus(self.consensus_type, self.dim).forward(input)


# 
# use multi-head attention as feature fusion module
# 
class FeatureFusionModule(torch.nn.Module):
    # TODO rewrite a more efficient module
    # since we can reuse some hidden states during iterations
    # 
    # 
    # self.feat_fusion = FeatureFusionModule(dim+3, dropout=hparams.dropout)
    # self.feat_fusion(x, x, x, need_weights=False)
    # 
    def __init__(self, dim, dropout=0):
        super(FeatureFusionModule, self).__init__()
        # 
        # torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, 
        #                             bias=True, add_bias_kv=False, add_zero_attn=False, 
        #                             kdim=None, vdim=None, batch_first=False, 
        #                             device=None, dtype=None)
        # 
        self.mha = torch.nn.MultiheadAttention(dim, 1, dropout)
        self.relu = torch.nn.ReLU(inplace=True)
        # 
        # 如果传入整数，比如4，则被看做只有一个整数的list
        # 此时LayerNorm会对输入的最后一维进行归一化，这个int值需要和输入的最后一维一样大
        # 
        self.norm = torch.nn.LayerNorm(dim)
        
        for name, weight in self.mha.named_parameters():
            if weight.dim() == 2:
                # 
                # if 'proj' not in name 
                # return name
                # 
                assert 'proj' in name, name
                torch.nn.init.kaiming_normal_(weight.data)
            elif weight.dim() == 1:
                assert 'bias' in name, name
                weight.data.zero_()

    def forward(self, *args, **kwargs):
        x = self.mha(*args, **kwargs)[0]
        x = self.norm(x)[-1]
        x = self.relu(x)
        return x


class ConvLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(ConvLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size  # mem_size
        # calculate 4 states in a single convolution operation
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size,
        kernel_size=kernel_size, stride=stride, padding=padding)
        # convolution parameters initialization
        torch.nn.init.xavier_normal_(self.Gates.weight)
        torch.nn.init.constant_(self.Gates.bias, 0)
    
    def forward(self, input_, prev_state):
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]  # height and width

        # initialize previous hidden_state and cell_state
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (torch.zeros(state_size).cuda(),
                          torch.zeros(state_size).cuda())

        prev_hidden, prev_cell = prev_state
        # stack input with previous hidden state
        stacked_inputs = torch.cat((input_, prev_hidden), 1)

        # convolution gates
        gates = self.Gates(stacked_inputs)
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)  # forget gate
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        # hidden state and cell state
        return hidden, cell

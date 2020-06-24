import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import numpy as np
from  sampling import sampling_schedule


class BahdanauAttentionDecoder(nn.Module):
    def __init__(self, cfgs):
        super(BahdanauAttentionDecoder, self).__init__()
        self.hidden_size = cfgs.HIDDEN_SIZE
        self.output_size = cfgs.OUTPUT_SIZE
        self.max_length = cfgs.MAX_LENGTH
        self.bidirectional = cfgs.BIDIRECTIONAL
        self.num_layers = cfgs.NUM_LAYERS
        self.cell_type = cfgs.CELL_TYPE
        self. attentionunit = AttentionUnit(cfgs)
        self.sampling_method = cfgs.SAMPLING_METHOD
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.iter =cfgs.ITER
        self.sampling_iter=cfgs.SAMPLING_ITER
        self.sampling_threshold=cfgs.SAMPLING_THRESHOLD
        self.sampler=sampling_schedule( final_iter=self.sampling_iter,sampling_method=self.sampling_method,threshold=self.sampling_threshold)
        if cfgs.CELL_TYPE == 'LSTM':
            rnn = nn.LSTM
        elif cfgs.CELL_TYPE == 'GRU':
            rnn = nn.GRU
        else:
            raise ValueError("Unsupported RNN type: {0}".format(rnn))
        if self.bidirectional:
            self.rnn = rnn(self.hidden_size * 3, self.hidden_size,
                           bidirectional=self.bidirectional, num_layers=self.num_layers)
            self.out = nn.Linear(self.hidden_size * 2, self.output_size)
        else:
            self.rnn = rnn(self.hidden_size * 2, self.hidden_size, bidirectional=self.bidirectional)
            self.out = nn.Linear(self.hidden_size, self.output_size)

    def _init_cell_hidden(self):
        '''
        return the initial hidden state at time 0
        '''
        if self.bidirectional:
            hidden = torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_size).to(self.device)
        else:
            hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        if self.cell_type == "GRU":
            return hidden
        elif self.cell_type == "LSTM":
            return (hidden, hidden)
        else:
            raise ValueError("Unsupport cell type")




    def forward(self, h_s, targets=None):
        '''
        h_s[maxlength,batchsize,ecoder_hidden_state]
        target:[batch_size,label length]
        step_input:[bachsize] all vallue is 0 which represents <sos>
        step_hidden:[1,batchsize,hiddensize] which is used to initialize encoder out at 0 time step
        '''
        probs = []
        self.batch_size = h_s.size(1)
        self.device = h_s.device
        h_t = self._init_cell_hidden()
        if targets is not None:
            assert isinstance(targets, tuple)
            targets, target_lengths = targets
            targets = targets.permute(1, 0).long()
            step_input = targets[0]
            for t in range(1, targets.size(0)):
                context = self.attentionunit(h_t, h_s)
                embeded = self.embedding(step_input).unsqueeze(1)
                concat = torch.cat((context, embeded), dim=2).transpose(0, 1)
               # print(concat.size())
                y, h_t = self.rnn(concat, h_t)
               # print(y.size())
                prob = F.log_softmax(self.out(y.squeeze(0)), dim=-1)
                probs.append(prob)
                step_input = self.sampler.sampling(targets[t], prob.max(1, keepdim=False)[1], self.iter,self.sampling_method)
            self.iter = self.iter + 1
            probs = torch.stack(probs)
            return probs
        else:
            step_input = torch.zeros(h_s.size(1)).to(self.device).long()
            for t in range(h_s.size(0)):
                context = self.attentionunit(h_t, h_s)
                embeded = self.embedding(step_input).unsqueeze(1)
                concat = torch.cat((context, embeded), dim=2).transpose(0, 1)
                y, h_t = self.rnn(concat, h_t)
                prob = self.out(y.squeeze(0))
                step_input = prob.max(1, keepdim=False)[1]
                probs.append(prob)
            probs = torch.stack(probs)
            return probs


class AttentionUnit(nn.Module):
    def __init__(self, cfgs):
        super(AttentionUnit, self).__init__()
        self.cell_type = cfgs.CELL_TYPE
        self.bidirectional = cfgs.BIDIRECTIONAL
        self.hidden_size = cfgs.HIDDEN_SIZE
        if self.bidirectional:
            self.inputsEmbed = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
            self.hEmbed = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
            self.wEmbed = nn.Linear(self.hidden_size * 2, 1)
        else:
            self.inputsEmbed = nn.Linear(self.hidden_size, self.hidden_size)
            self.hEmbed = nn.Linear(self.hidden_size, self.hidden_size)
            self.wEmbed = nn.Linear(self.hidden_size, 1)

    def forward(self, h_t_1, h_s):
        if self.cell_type == "LSTM":
            h_t_1 = h_t_1[0]
        if self.bidirectional:
            h_t_1 = torch.cat((h_t_1[-1], h_t_1[-2]), 1)
        else:
            h_t_1 = h_t_1[-1]
        h_s = h_s.transpose(0, 1).contiguous()
        batch_size, T, encoder_dim = h_s.size()
        inputs = h_s
        h_s = self.inputsEmbed(h_s).contiguous()
        h_s = h_s.view(batch_size, T, -1)
        h_t_1 = h_t_1.squeeze(0)
        h_t_1 = self.hEmbed(h_t_1).unsqueeze(1)
        h_t_1 = h_t_1.expand(h_s.size())
        sumTanh = torch.tanh(h_t_1 + h_s)
        sumTanh = sumTanh.view(-1, encoder_dim)
        vProj = self.wEmbed(sumTanh)  # [(b x T) x 1]
        vProj = vProj.view(batch_size, T)
        alpha = torch.softmax(vProj, dim=1).unsqueeze(1)
        context = alpha.bmm(inputs)
        return context


class cfg:
    def __init__(self):
        super(cfg, self).__init__()
        self.HIDDEN_SIZE = 256
        self.OUTPUT_SIZE = 7175
        self.NUM_LAYERS = 1
        self.BIDIRECTIONAL = True
        self.CELL_TYPE = "GRU"
        self. MAX_LENGTH = 90
        self.DROPOUT = 0.2
        self.METHOD = "General"
        self.ITER=0
        self.SAMPLING_METHOD = "Linear_decay"
        self.SAMPLING_ITER=8000
        self.SAMPLING_THRESHOLD=0.6


if __name__ == "__main__":
    x = torch.randn(90, 32, 512)
    target = torch.randint(0, 90, [32, 32])
    target2 = torch.randint(0, 90, [32, 32])
    t3 = torch.cat((target, target2), 1)
    # print(t3.size())
    target_length = torch.randint(0, 180, [32])
    cfg = cfg()
    qw = BahdanauAttentionDecoder(cfg)
    a = qw.forward(x,(target,target_length))

    pass

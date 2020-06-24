import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
from .sampling import sampling_schedule


class LuongAttentionDecoder(nn.Module):
    def __init__(self, cfgs):
        super(LuongAttentionDecoder, self).__init__()
        self.AttentionUnit = AttentionUnit(cfgs)
        self.hidden_size = cfgs.HIDDEN_SIZE
        self.output_size = cfgs.OUTPUT_SIZE
        self.max_length = cfgs.MAX_LENGTH
        self.iter=cfgs.ITER

        self.method = cfgs.METHOD
        self.sampling_method = cfgs.SAMPLING_METHOD
        self.num_layers = cfgs.NUM_LAYERS
        self.bidirectional = cfgs.BIDIRECTIONAL
        self.cell_type = cfgs.CELL_TYPE       
        self.sampling_iter=cfgs.SAMPLING_ITER
        self.sampling_threshold=cfgs.SAMPLING_THRESHOLD
        self.sampler=sampling_schedule( final_iter=self.sampling_iter,sampling_method=self.sampling_method,threshold=self.sampling_threshold)
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if self.cell_type == 'LSTM':
            rnn = nn.LSTM
        elif self.cell_type == 'GRU':
            rnn = nn.GRU
        else:
            raise ValueError("Unsupported RNN type: {0}".format(rnn))
        if self.bidirectional:
            self.concat_linear = nn.Linear(self.hidden_size * 4, self.hidden_size)
        else:
            self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.rnn = rnn(self.hidden_size*2, self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.softmax_linear = nn.Linear(self.hidden_size, self.output_size)

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
        if self.cell_type == "LSTM":
            return (hidden, hidden)

    def step_forward(self, context, y_t):
        y_t = y_t.squeeze(0)
        concat = torch.cat((y_t, context), 1)
        concat_output = F.tanh(self.concat_linear(concat))
        return concat_output

    def forward(self, h_s, targets=None):
        '''
             encoder_output[maxlength,batchsize,ecoder_hidden_state]
             target:[batch_size,label length]
             step_input:[bachsize] all vallue is 0 which represents <sos>
             step_hidden:[1,batchsize,hiddensize] which is used to initialize encoder out at 0 time step
        '''
        probs = []
        self.batch_size = h_s.size(1)
        self.device = h_s.device
        h_t = self._init_cell_hidden()
        concat_output=torch.zeros((self.batch_size,self.hidden_size)).to(self.device)
        if targets is not None:
            assert isinstance(targets, tuple)
            targets, target_lengths = targets
            targets = targets.permute(1, 0).long()
            step_input = targets[0]
            for t in range(1, targets.size(0)):
                embedded = self.embedding(step_input).unsqueeze(0)
                embedded=torch.cat((embedded,concat_output.unsqueeze(0)),2)
                y_t, h_t = self.rnn(embedded, h_t)
                context = self.AttentionUnit(h_t, h_s)
                concat_output = self.step_forward(context, y_t)
            #    print(embedded.size())
           #     print(concat_output.size())
                prob = F.log_softmax(self.softmax_linear(concat_output), dim=-1)
                step_input = self.sampler.sampling(targets[t], prob.max(1, keepdim=False)[1], self.iter,self.sampling_method)
                probs.append(prob)      
            probs = torch.stack(probs)
            self.iter+=1
            return probs
        else:
            step_input = torch.zeros(h_s.size(1)).to(self.device).long()
            for t in range(0, h_s.size(0)):
                embedded = self.embedding(step_input).unsqueeze(0)
                embedded=torch.cat((embedded,concat_output.unsqueeze(0)),2)
                y_t, h_t = self.rnn(embedded, h_t)
                context = self.AttentionUnit(h_t, h_s)
                concat_output = self.step_forward(context, y_t)

                prob = self.softmax_linear(concat_output)
                step_input = prob.max(1, keepdim=False)[1]
                probs.append(prob)
            probs = torch.stack(probs)
        #     print(probs.size())step_input = self._sampling(targets[t], prob.max(1, keepdim=False)[1], self.sampling_method)
            return probs


class AttentionUnit(nn.Module):
    def __init__(self, cfgs):
        super(AttentionUnit, self).__init__()
        self.cell_type = cfgs.CELL_TYPE
        self.bidirectional = cfgs.BIDIRECTIONAL
        self.hidden_size = cfgs.HIDDEN_SIZE
        self.method = cfgs.METHOD
        if self.method == "General":
            if self.bidirectional:
                self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
            else:
                self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == "Concat":
            if self.bidirectional:
                self.attn = nn.Linear(self.hidden_size * 4, self.hidden_size)
            else:
                self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Linear(self.hidden_size, 1)

    def score(self, h_t, h_s):
        '''
        different alignment function
        '''
        if self.cell_type == "LSTM":
            h_t = h_t[0]
        if self.bidirectional:
            h_t = torch.cat((h_t[-1], h_t[-2]), 1)
        else:
            h_t = h_t[-1]
        if self.method == "General":
            h_t = self.attn(h_t).unsqueeze(1)
            h_s = h_s.permute(1, 2, 0)
            attn_energies = torch.bmm(h_t, h_s)
            return attn_energies
        if self.method == "Concat":
            h_t = h_t.unsqueeze(0)
            h_t = h_t.expand(h_s.size())
            ht_hsconcat = torch.cat((h_t, h_s), dim=-1).transpose(0, 1)
            tanh_cat = F.tanh(self.attn(ht_hsconcat))
            attn_energies = self.v(tanh_cat).transpose(1, 2)
            return attn_energies

    def forward(self, h_t, h_s):
        attn_energies = self.score(h_t, h_s)
        attn_weights = F.softmax(attn_energies)
        context = attn_weights.bmm(h_s.transpose(0, 1)).squeeze(1)
        return context


class cfg:
    def __init__(self):
        super(cfg, self).__init__()
        self.HIDDEN_SIZE = 256
        self.OUTPUT_SIZE = 7175
        self.NUM_LAYERS = 2
       
        self.BIDIRECTIONAL = True
        self.CELL_TYPE = "GRU"
        self. MAX_LENGTH = 180
        self.DROPOUT = 0.2
        self.METHOD = "General"
        self.SAMPLING_METHOD = "Linear_decay"
        self.ITER=0
        self.SAMPLING_ITER=8000
        self.SAMPLING_THRESHOLD=0.6

if __name__ == "__main__":
    x = torch.randn(180, 32, 512)
    target = torch.randint(0, 180, [32, 32])
    target2 = torch.randint(0, 180, [32, 32])
    t3 = torch.cat((target, target2), 1)
    target_length = torch.randint(0, 180, [32])
    cfg = cfg()
    qw = LuongAttentionDecoder(cfg)
    a = qw.forward(x,(target,target_length))

    pass

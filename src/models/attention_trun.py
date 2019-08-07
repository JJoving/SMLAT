import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DotProductAttention(nn.Module):
    r"""Dot product attention.
    Given a set of vector values, and a vector query, attention is a technique
    to compute a weighted sum of the values, dependent on the query.

    NOTE: Here we use the terminology in Stanford cs224n-2018-lecture11.
    """

    def __init__(self):
        super(DotProductAttention, self).__init__()
        # TODO: move this out of this class?
        # self.linear_out = nn.Linear(dim*2, dim)

    def forward(self, queries, values, mask):
        """
        Args:
            queries: N x To x H
            values : N x Ti x H

        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        #import pdb
        #pdb.set_trace()
        batch_size = queries.size(0)
        hidden_size = queries.size(2)
        input_lengths = values.size(1)
        #print(queries.size())
        #print(values.size())
        #print(queries)
        #print(values)
        # (N, To, H) * (N, H, Ti) -> (N, To, Ti)
        attention_scores = torch.bmm(queries, values.transpose(1, 2))
        #print(attention_scores.size())
        #print(attention_scores)
        if mask is not None:
            attention_distribution = F.softmax(
                attention_scores.view(-1, input_lengths).masked_fill(mask, -99999.0), dim=1).view(batch_size, -1, input_lengths)
        else:
            attention_distribution = F.softmax(attention_scores.view(-1, input_lengths), dim=1).view(batch_size, -1, input_lengths)
        #print(attention_distribution.size())
        #print(attention_distribution)
        # (N, To, Ti) * (N, Ti, H) -> (N, To, H)
        #attention_distribution = self.dropout(attention_distribution)
        attention_output = torch.bmm(attention_distribution, values)
        #print(attention_output.size())
        #print(attention_output)
        # # concat -> (N, To, 2*H)
        # concated = torch.cat((attention_output, queries), dim=2)
        # # TODO: Move this out of this class?
        # # output -> (N, To, H)
        # output = torch.tanh(self.linear_out(
        #     concated.view(-1, 2*hidden_size))).view(batch_size, -1, hidden_size)

        return attention_output, attention_distribution

class ContentBasedAttention(nn.Module):

    def __init__(self, hidden_size):
        super(ContentBasedAttention, self).__init__()
        self.attenV = self.Linear(hidden_size, hidden_size, bias=False)
        # W * LSTM_outputs + b
        self.attenW = self.Linear(hidden_size, hidden_size)
        self.attenwT = self.Linear(hidden_size, 1, bias=False)

    def Linear(self, input_dim, output_dim, bias=True):
        linear = nn.Linear(input_dim, output_dim, bias=bias)
        return linear
    def forward(self, dec_inputs, enc_outputs, mask, scaling=1.0):
        """
        Args:
            dec_inputs: N x To x H
            enc_outputs : N x Ti x H
        Returns:
            output: N x To x H
            attention_distribution: N x To x Ti
        """
        """The content attention:
           c_t = softmax(e_t)
           e_t = w^T tanh(Ws_i-1 + Vh + b)
        """
        #import pdb
        #pdb.set_trace()
        batch_size = dec_inputs.size(0)
        hidden_size = dec_inputs.size(2)
        input_lengths = enc_outputs.size(1)
        atten_Vh = self.attenV(enc_outputs)
        atten_Ws = (self.attenW(dec_inputs)
                          .view(batch_size, 1, hidden_size)
                          .expand(batch_size, input_lengths, hidden_size))
        atten_e = F.tanh(atten_Vh + atten_Ws)
        atten_e = self.attenwT(atten_e).view(batch_size, -1)
        if mask is not None:
            atten_a = F.softmax(scaling * atten_e.masked_fill(mask, -1e-5), dim=1).view(batch_size, 1, -1)
        else:
            atten_a = F.softmax(scaling * atten_e, dim=1).view(batch_size, 1, -1)
        atten_c = torch.bmm(atten_a, enc_outputs).view(batch_size, hidden_size)
        return atten_c, atten_a
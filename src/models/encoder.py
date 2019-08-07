import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

splice=False
#splice=True

class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes = 0,
                 dropout=0.0, bidirectional=True, rnn_type='lstm', ctc_las = False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.ctc_las = ctc_las

        if bidirectional == 0:
            bid = 1
        else:
            bid = 2
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)
            if splice:
                self.rnn1 = nn.LSTM(input_size, hidden_size, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
                self.rnn2 = nn.LSTM(hidden_size * bid * 2, hidden_size, num_layers-1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        if self.ctc_las:
            self.fc = nn.Linear(hidden_size * bid, num_classes)


    def forward(self, padded_input, input_lengths):
        """
        Args:
            padded_input: N x T x D
            input_lengths: N

        Returns: output, hidden
            - **output**: N x T x H
            - **hidden**: (num_layers * num_directions) x N x H
        """
        # Add total_length for supportting nn.DataParallel() later
        # see https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        total_length = padded_input.size(1)  # get the max sequence length
        packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
        if splice:
            packed_output, hidden = self.rnn1(packed_input)
        else:
            packed_output, hidden = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
        if splice:
            n = output.shape[0]
            t = output.shape[1]
            d = output.shape[2]
            if int(output.shape[1]%2) == 1:
                z=torch.zeros((n, d)).cuda()
                output_t=output.contiguous().view(n,-1)
                input2=torch.cat((output_t,z),1).contiguous().view(n,int((t+1)/2),d*2)
            else :
                input2=output.contiguous().view(n,-1,d*2)
            total_length2 = input2.size(1)
            input_lengths=torch.div(torch.add(input_lengths,1),2)
            packed_input2 = pack_padded_sequence(input2, input_lengths, batch_first=True)
            packed_output2, hidden = self.rnn2(packed_input2)
            output, _ = pad_packed_sequence(packed_output2, batch_first=True, total_length=total_length2)
        if self.ctc_las:
            ctc_output = F.log_softmax(self.fc(output), dim=-1)
            return output, ctc_output, hidden
        return output, hidden

    def flatten_parameters(self):
        #if splice:
            #self.rnn1.flatten_parameters()
            #self.rnn2.flatten_parameters()
        #else:
            self.rnn.flatten_parameters()

import torch
import six
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    r"""Applies a multi-layer LSTM to an variable length input sequence.
    """

    def __init__(self, input_size, hidden_size, num_layers, projection_size, num_classes = 0,
                 dropout=0.0, bidirectional=True, rnn_type='lstm'):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.projection_size = projection_size
        self.use_projection = 1
        if self.use_projection:
            for i in six.moves.range(num_layers):
                rnn = nn.LSTM(input_size, hidden_size, num_layers = 1,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)
                setattr(self, "%s%d" % ("birnn" if bidirectional else "rnn", i), rnn)
                if bidirectional:
                    setattr(self, "bt%d" % i, torch.nn.Linear(hidden_size * 2, self.projection_size))
                else:
                    setattr(self, "bt%d" % i, torch.nn.Linear(hidden_size, self.projection_size))
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)
            if bidirectional:
                self.fc_last = torch.nn.Linear(hidden_size * 2, hidden_size * 2)
            else:
                self.fc_last = torch.nn.Linear(hidden_size, hidden_size)


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
        #import pdb
        #pdb.set_trace()
        if self.use_projection == 0:
            total_length = padded_input.size(1)  # get the max sequence length
            packed_input = pack_padded_sequence(padded_input, input_lengths,
                                            batch_first=True)
            packed_output, hidden = self.rnn(packed_input)
            output, output_lengths = pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)
            projected = torch.tanh(self.fc_last(output.contiguous().view(-1, output.size(2))))
            xs_pad = projected.view(output.size(0), output.size(1), -1)
            return output, output_lengths, hidden
        else:
            elayer_states = []
            for layer in six.moves.range(self.num_layers):
                xs_pack = pack_padded_sequence(padded_input, input_lengths, batch_first=True)
                rnn = getattr(self, ("birnn" if self.bidirectional else "rnn") + str(layer))
                rnn.flatten_parameters()
                ys, states = rnn(xs_pack)
                elayer_states.append(states)
                # ys: utt list of frame x cdim x 2 (2: means bidirectional)
                ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
                # (sum _utt frame_utt) x dim
                projected = getattr(self, 'bt' + str(layer)
                                    )(ys_pad.contiguous().view(-1, ys_pad.size(2)))
                xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

            return xs_pad, ilens, elayer_states

    def flatten_parameters(self):
            self.rnn.flatten_parameters()

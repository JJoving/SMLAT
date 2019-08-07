import torch
import torch.nn as nn
import time

from decoder_trun import Decoder
from encoder import Encoder
from data import align_truncate, align_process, align_process_2

class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder, decoder, ctc_las=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ctc_las = ctc_las

    def forward(self, padded_input, input_lengths, padded_target, align):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #import pdb
        #pdb.set_trace()
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        ys_pad, left_pad, right_pad = align_process(align)
        #print("ys_pad")
        #print(ys_pad)
        #print("left_pad")
        #print(left_pad)
        #print("right_pad")
        #print(right_pad)
        #xs, ys = align_truncate(align, padded_target, encoder_padded_outputs)
        #print(len(ys))
        #yy = [y[y != -1] for y in padded_target]
        #yyl = [len(y) for y in yy]
        #print(yyl,sum(yyl))
        #print(alen,sum(alen))
        #aa = [y[y != -1] for y in align]
        #aas=[]
        #for k in range(len(aa)):
        #    aal= []
        #    last = -1
        #    for i in range(0, len(aa[k])):
        #        if aa[k][i] != last and aa[k][i] != 0:
        #            aal.append(aa[k][i])
        #            last = aa[k][i]
        #    aas.append(aal)
        #aasl = [len(y) for y in aas]
        #print(aasl,sum(aasl))
        #import pdb
        #pdb.set_trace()
        loss = self.decoder(ys_pad, encoder_padded_outputs, left_pad, right_pad)
        #loss = self.decoder(ys, xs)
        t4=time.time()
        #print(1000*(t2-t1),1000*(t3-t2),1000*(t4-t3),1000*(t4-t1))
        return loss

    def recognize(self, input, input_length, char_list, align, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam

        Returns:
            nbest_hyps:
        """
        #import pdb
        #pdb.set_trace()
        encoder_outputs, _ = self.encoder(input.unsqueeze(0), input_length)
        lid = 0
        left=[]
        right=[]
        for i in range(1, len(align)):
            if align[i-1] != align[i]:
                if align[i-1] != 0:
                    left.append(lid)
                    right.append(i)
                lid = i
                if i == len(align)-1 and align[i] != 0:
                    left.append(lid)
                    right.append(lens)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 torch.Tensor(left),
                                                 torch.Tensor(right),
                                                 args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        encoder = Encoder(package['einput'],
                          package['ehidden'],
                          package['elayer'],
                          dropout=package['edropout'],
                          bidirectional=package['ebidirectional'],
                          rnn_type=package['etype'])
        decoder = Decoder(package['dvocab_size'],
                          package['dembed'],
                          package['dsos_id'],
                          package['deos_id'],
                          package['dhidden'],
                          package['dlayer'],
                          -2,
                          bidirectional_encoder=package['ebidirectional']
                          )
        encoder.flatten_parameters()
        model = cls(encoder, decoder)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # encoder
            'einput': model.encoder.input_size,
            'ehidden': model.encoder.hidden_size,
            'elayer': model.encoder.num_layers,
            'edropout': model.encoder.dropout,
            'ebidirectional': model.encoder.bidirectional,
            'etype': model.encoder.rnn_type,
            # decoder
            'dvocab_size': model.decoder.vocab_size,
            'dembed': model.decoder.embedding_dim,
            'dsos_id': model.decoder.sos_id,
            'deos_id': model.decoder.eos_id,
            'dhidden': model.decoder.hidden_size,
            'dlayer': model.decoder.num_layers,
            'doffset': model.decoder.offset,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

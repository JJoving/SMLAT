import torch
import torch.nn as nn
import numpy as np

from decoder_e2e import Decoder
from encoder_e2e import Encoder
from ctc import CTC

from itertools import groupby
from utils import IGNORE_ID, pad_list

import editdistance


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, encoder, decoder, ctc, args):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

        self.mode = args.mode
        self.trun = args.trun
        self.char_list = args.char_list
        self.space = "<unk>"
        self.blank = "<blank>"
        self.loss = None

    def forward(self, xs_pad, ilens, ys_pad, iter, epoch, aligns_pad =None):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #import pdb
        #pdb.set_trace()
        #import time
        #time1 = time.time()
        hs_pad, hlens, _ = self.encoder(xs_pad, ilens)
        #time2 = time.time()
        if self.mode == 0:
            loss_ctc = 0
        else:
            loss_ctc = self.ctc(hs_pad, hlens, ys_pad)
            if self.trun:
                lpz = self.ctc.log_softmax(hs_pad)
                ctc_greedy = torch.max(lpz, dim=-1)[1]
                #print(ctc_greedy)
                aligns = []
                for k in range(ctc_greedy.size()[0]):
                    align = (torch.nonzero(ctc_greedy[k]) + 1).reshape(-1).cpu().numpy().tolist()
                    align.insert(0, 0)
                    aligns.append(align)
                #print(aligns[0:2])
                #print(np.shape(aligns))
                #aligns = torch.Tensor(aligns).long().cuda()
                aligns_pad = pad_list([torch.Tensor(y).long() for y in aligns], IGNORE_ID)
        #time3 = time.time()
        if self.mode == 1:
            loss_att = 0
        else:
            loss_att = self.decoder(ys_pad, hs_pad, aligns_pad, self.trun, epoch)
        #time4 = time.time()
        if self.mode == 0:
            cer_ctc = None
        else:
            cers = []
            cer_ctc = 0

            y_hats = self.ctc.argmax(hs_pad).data
            show_detail = 0
            if iter % 100 == 0:
                for i, y in enumerate(y_hats):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = ys_pad[i]

                    seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
                    seq_true = [self.char_list[int(idx)] for idx in y_true if int(idx) != -1]
                    seq_hat_text = "".join(seq_hat).replace(self.space, ' ')
                    seq_hat_text = seq_hat_text.replace(self.blank, '')
                    seq_true_text = "".join(seq_true).replace(self.space, ' ')

                    hyp_chars = seq_hat_text.replace(' ', '')
                    ref_chars = seq_true_text.replace(' ', '')
                    if len(ref_chars) > 0:
                        cers.append(editdistance.eval(hyp_chars, ref_chars) / len(ref_chars))
                    #import pdb
                    #pdb.set_trace()
                    if i == (y_hats.size(0)-1):
                        print(hyp_chars)
                        print(ref_chars)
                        if self.trun:
                            print(aligns_pad[-1].numpy().tolist())

                cer_ctc = sum(cers) / len(cers) if cers else None


        #if self.report_wer:
        #    if self.ctc_weight > 0.0:
        #        lpz = self.ctc.log_softmax(hs_pad).data
        #    else:
        #        lpz = None
        #    wers, cers = [], []
        #    nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0], self.char_list, args)
        #time5 = time.time()
        alpha = self.mode
        if alpha == 0:
            self.loss = loss_att
        elif alpha == 1:
            self.loss = loss_ctc
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
        #self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
        if self.mode == 0:
            cer_ctc = 0
        #print(1000 * (time2 - time1), 1000 * (time3 - time2), 1000 * (time4 - time3), 1000 * (time5 - time4))
        print("ctc loss {0} | att loss {1} | loss {2} | cer {3}".format(float(loss_ctc), float(loss_att), float(self.loss), float(cer_ctc)))
        return self.loss

    def recognize(self, input, input_length, char_list, args):
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
        encoder_outputs, _, _ = self.encoder(input.unsqueeze(0), input_length)
        if args.ctc_weight > 0 or args.trun:
            lpz = self.ctc.log_softmax(encoder_outputs)[0]
        else:
            lpz = None
        aligns_pad = None
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 lpz,
                                                 aligns_pad,
                                                 args)
        return nbest_hyps

    def recognize_align(self, input, input_length, char_list, align, args):
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
        encoder_outputs, _, _ = self.encoder(input.unsqueeze(0), input_length)
        if args.ctc_weight > 0 or args.trun:
            lpz = self.ctc.log_softmax(encoder_outputs)[0]
        else:
            lpz = None
        aligns_pad = []
        aligns = [0,]
        for i in range(1, len(align)):
            if int(align[i-1]) != int(align[i]):
                if int(align[i-1]) != 0:
                    aligns.append(i)
                if i == len(align)-1 and int(align[i]) != 0:
                    aligns.append(lens)
        aligns_pad.append(aligns)
        aligns_pad = pad_list([torch.Tensor(y).long() for y in aligns_pad], IGNORE_ID)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 lpz,
                                                 aligns_pad,
                                                 args)
        return nbest_hyps
    @classmethod
    def load_model(cls, path, args):
        # Load to CPU
        #import pdb
        #pdb.set_trace()
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model, LFR_m, LFR_n = cls.load_model_from_package(package, args)
        return model, LFR_m, LFR_n

    @classmethod
    def load_model_from_package(cls, package, args):
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
                          package['doffset'],
                          package['atype'],
                          package['edropout'],
                          package['lsm_weight'],
                          package['sampling_probability'],
                          package['peak_left'],
                          package['peak_right'],
                          bidirectional_encoder=package['ebidirectional']
                          )
        ctc = CTC(package['dvocab_size'],
                eprojs = package['ehidden'] * 2 if package['ebidirectional'] else package['ehidden'],
                  dropout_rate = package['edropout'],
                  )
        encoder.flatten_parameters()
        model = cls(encoder, decoder, ctc, args)
        model.load_state_dict(package['state_dict'])
        LFR_m, LFR_n = package['LFR_m'], package['LFR_n']
        return model, LFR_m, LFR_n

    @staticmethod
    def serialize(model, optimizer, epoch, LFR_m, LFR_n, tr_loss=None, cv_loss=None):
        package = {
            # Low Frame Rate Feature
            'LFR_m': LFR_m,
            'LFR_n': LFR_n,
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
            'doffset':model.decoder.offset,
            'atype':model.decoder.atype,
            'sampling_probability':model.decoder.sampling_probability,
            'lsm_weight':model.decoder.lsm_weight,
            'peak_left':model.decoder.peak_left,
            'peak_right':model.decoder.peak_right,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

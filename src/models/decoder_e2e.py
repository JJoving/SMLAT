import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import codecs
import random


from ctc_prefix_score import CTCPrefixScore
from ctc_prefix_score import CTCPrefixScoreTH

from attention_trun import DotProductAttention, ContentBasedAttention
from utils import IGNORE_ID, pad_list
from label_smoothing_loss import LabelSmoothingLoss

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

class Decoder(nn.Module):
    """
    """

    def __init__(self, vocab_size, embedding_dim, sos_id, eos_id, hidden_size,
                 num_layers, offset, atype, dropout, lsm_weight, sampling_probability, bidirectional_encoder=True):
        super(Decoder, self).__init__()
        # Hyper parameters
        # embedding + output
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.offset = offset
        self.atype = atype
        self.dropout = nn.Dropout(dropout)
        self.lsm_weight = lsm_weight
        self.criterion = LabelSmoothingLoss(vocab_size, IGNORE_ID, self.lsm_weight)
        self.sampling_probability = sampling_probability
        # rnn
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder  # useless now
        self.encoder_hidden_size = hidden_size  # must be equal now
        # Components
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.ModuleList()
        self.rnn += [nn.LSTMCell(self.embedding_dim +
                                 self.encoder_hidden_size, self.hidden_size)]
        for l in range(1, self.num_layers):
            self.rnn += [nn.LSTMCell(self.hidden_size, self.hidden_size)]
        if self.atype == "dot":
            self.attention = DotProductAttention()
        elif self.atype == "content":
            self.attention = ContentBasedAttention(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_hidden_size + self.hidden_size,
                      self.hidden_size),
            #nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.vocab_size))

    def zero_state(self, encoder_padded_outputs, H=None):
        N = encoder_padded_outputs.size(0)
        H = self.hidden_size if H == None else H
        return encoder_padded_outputs.new_zeros(N, H)

    def forward(self, padded_input, encoder_padded_outputs, aligns, trun, epoch):
        """
        Args:
            padded_input: N x To
            # encoder_hidden: (num_layers * num_directions) x N x H
            encoder_padded_outputs: N x Ti x H

        Returns:
        """
        # *********Get Input and Output
        # from espnet/Decoder.forward()
        # TODO: need to make more smart way
        #import pdb
        #pdb.set_trace()
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        if aligns is not  None:
            aligns = [y[y != IGNORE_ID] for y in aligns]
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        #if len(aligns) != 0:
        #    aligns = [torch.cat([sos, y], dim=0) for y in aligns]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, IGNORE_ID)
        if aligns != None:
            aligns_pad = pad_list(aligns, 0)
            #if aligns_pad.size(1) < ys_in_pad.size(1):
            #    aligns_pad_end = aligns_pad.new_full((1, int(ys_in_pad.size(1) - aligns_pad.size(1))), 0)
            #    aligns_pad = [torch.cat([y, aligns_pad_end], dim=0) for y in aligns_pad]

        # print("ys_in_pad", ys_in_pad.size())
        assert ys_in_pad.size() == ys_out_pad.size()
        batch_size = ys_in_pad.size(0)
        output_length = ys_in_pad.size(1)
        #print(ys_in_pad[0])
        # max_length = ys_in_pad.size(1) - 1  # TODO: should minus 1(sos)?

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_padded_outputs)]
        c_list = [self.zero_state(encoder_padded_outputs)]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_padded_outputs))
            c_list.append(self.zero_state(encoder_padded_outputs))
        att_c = self.zero_state(encoder_padded_outputs,
                                H=encoder_padded_outputs.size(2))
        y_all = []

        # **********LAS: 1. decoder rnn 2. attention 3. concate and MLP
        #import pdb
        #pdb.set_trace()
        if self.sampling_probability:
            if epoch <= 8:
                sp = 0
            else:
                sp = 0.08 + 0.01 * epoch


        embedded = self.embedding(ys_in_pad)
        for t in range(output_length):
            #print(output_length)
            # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
            if t > 0 and self.sampling_probability and random.random() < sp:
                y_out = y_all[-1]
                y_out = np.argmax(y_out.detach(), axis=1)
                rnn_input = torch.cat((self.embedding(y_out.cuda()), att_c), dim=1)
            else:
                rnn_input = torch.cat((embedded[:, t, :], att_c), dim=1)
            h_list[0], c_list[0] = self.rnn[0](
                rnn_input, (h_list[0], c_list[0]))
            for l in range(1, self.num_layers):
                #h_list[l-1] = self.dropout(h_list[l-1])
                h_list[l], c_list[l] = self.rnn[l](
                    h_list[l-1], (h_list[l], c_list[l]))

            rnn_output = h_list[-1]  # below unsqueeze: (N x H) -> (N x 1 x H)
            # step 2. attention: c_i = AttentionContext(s_i,h)
            mask = None
            if aligns:
                #mask = torch.ones(encoder_padded_outputs.size(0),encoder_padded_outputs.size(1),dtype=torch.uint8).cuda()
                mask = torch.zeros(encoder_outputs.unsqueeze(0).size(0),encoder_outputs.unsqueeze(0).size(1),dtype=torch.uint8).cuda()
                if t + 1 < aligns_pad.size(1):
                    for m in range(mask.size(0)):
                        #left_bound = min(aligns_pad[m][t] + self.offset, rnn_output.size(1))
                        right_bound = max(min(aligns_pad[m][t+1] + self.offset, rnn_output.size(1)), 0)
                        #left_bound = 0
                        #mask[m][left_bound:right_bound] = 0
                        mask[m][right_bound:-1] = 1
            att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                          encoder_padded_outputs,
                                          mask)
            #att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
            #                              encoder_padded_outputs)
            att_c = att_c.squeeze(dim=1)
            # step 3. concate s_i and c_i, and input to MLP
            mlp_input = torch.cat((rnn_output, att_c), dim=1)
            predicted_y_t = self.mlp(mlp_input)
            y_all.append(predicted_y_t)

        y_all = torch.stack(y_all, dim=1)  # N x To x C
        # **********Cross Entropy Loss
        # F.cross_entropy = NLL(log_softmax(input), target))
        #import pdb
        #pdb.set_trace()
        if self.lsm_weight:
            ce_loss = self.criterion(y_all ,ys_out_pad) / (1.0 / (np.mean([len(y) for y in ys_in]) - 1) * np.sum([len(y) for y in ys_in]))
        else:
            y_all = y_all.view(batch_size * output_length, self.vocab_size)
            ce_loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                  ignore_index=IGNORE_ID,
                                  reduction='elementwise_mean')
        # TODO: should minus 1 here ?
            ce_loss *= (np.mean([len(y) for y in ys_in]) - 1)
        # print("ys_in\n", ys_in)
        # temp = [len(x) for x in ys_in]
        # print(temp)
        # print(np.mean(temp) - 1)
        return ce_loss

        # *********step decode
        # decoder_outputs = []
        # sequence_symbols = []
        # lengths = np.array([max_length] * batch_size)

        # def decode(step, step_output, step_attn):
        #     # step_output is log_softmax()
        #     decoder_outputs.append(step_output)
        #     symbols = decoder_outputs[-1].topk(1)[1]
        #     sequence_symbols.append(symbols)
        #     #
        #     eos_batches = symbols.data.eq(self.eos_id)
        #     if eos_batches.dim() > 0:
        #         eos_batches = eos_batches.cpu().view(-1).numpy()
        #         update_idx = ((step < lengths) & eos_batches) != 0
        #         lengths[update_idx] = len(sequence_symbols)
        #     return symbols

        # # *********Run each component
        # decoder_input = ys_in_pad
        # embedded = self.embedding(decoder_input)
        # rnn_output, decoder_hidden = self.rnn(embedded)  # use zero state
        # output, attn = self.attention(rnn_output, encoder_padded_outputs)
        # output = output.contiguous().view(-1, self.hidden_size)
        # predicted_softmax = F.log_softmax(self.out(output), dim=1).view(
        #     batch_size, output_length, -1)
        # for t in range(predicted_softmax.size(1)):
        #     step_output = predicted_softmax[:, t, :]
        #     step_attn = attn[:, t, :]
        #     decode(t, step_output, step_attn)

    def recognize_beam(self, encoder_outputs, char_list, lpz, aligns_pad, args):
        """Beam search, decode one utterence now.
        Args:
            encoder_outputs: T x H
            char_list: list of character
            args: args.beam

        Returns:
            nbest_hyps:
        """
        # search params
        beam = args.beam_size
        nbest = args.nbest
        ctc_weight = args.ctc_weight
        CTC_SCORING_RATIO = 1.5
        if args.decode_max_len != 0:
            maxlen = args.decode_max_len
        elif args.trun:
            maxlen = int(len(torch.nonzero(torch.max(lpz, dim=-1)[1])) * 1.5)
        elif args.align_trun:
            maxlen = int(aligns_pad.size(1) * 1.5)

        # *********Init decoder rnn
        h_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        c_list = [self.zero_state(encoder_outputs.unsqueeze(0))]
        for l in range(1, self.num_layers):
            h_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
            c_list.append(self.zero_state(encoder_outputs.unsqueeze(0)))
        att_c = self.zero_state(encoder_outputs.unsqueeze(0),
                                H=encoder_outputs.unsqueeze(0).size(2))
        # prepare sos
        y = self.sos_id
        vy = encoder_outputs.new_zeros(1).long()

        hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'h_prev': h_list,
               'a_prev': att_c}
        if lpz is not None:
            #import pdb
            #pdb.set_trace()
            ctc_prefix_score = CTCPrefixScore(lpz.detach().cpu().numpy(), 0, self.eos_id, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
            if args.trun:
                ctc_greedy = torch.max(lpz, dim=-1)[1].unsqueeze(dim=0)
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

        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                # vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                embedded = self.embedding(vy)
                # embedded.unsqueeze(0)
                # step 1. decoder RNN: s_i = RNN(s_i−1,y_i−1,c_i−1)
                rnn_input = torch.cat((embedded, hyp['a_prev']), dim=1)
                h_list[0], c_list[0] = self.rnn[0](
                    rnn_input, (hyp['h_prev'][0], hyp['c_prev'][0]))
                for l in range(1, self.num_layers):
                    h_list[l], c_list[l] = self.rnn[l](
                        h_list[l-1], (hyp['h_prev'][l], hyp['c_prev'][l]))
                rnn_output = h_list[-1]
                # step 2. attention: c_i = AttentionContext(s_i,h)
                # below unsqueeze: (N x H) -> (N x 1 x H)
                #import pdb
                #pdb.set_trace()
                mask = None
                if args.trun or args.align_trun:
                    #mask = torch.ones(encoder_outputs.unsqueeze(0).size(0),encoder_outputs.unsqueeze(0).size(1),dtype=torch.uint8).cuda()
                    mask = torch.zeros(encoder_outputs.unsqueeze(0).size(0),encoder_outputs.unsqueeze(0).size(1),dtype=torch.uint8).cuda()
                    if i + 1 < aligns_pad.size(1):
                        for m in range(mask.size(0)):
                            #left_bound = min(aligns_pad[m][i] + self.offset, rnn_output.size(1))
                            right_bound = max(min(aligns_pad[m][i+1] + self.offset, rnn_output.size(1)), 0)
                            #left_bound = 0
                            #mask[m][left_bound:right_bound] = 0
                            mask[m][right_bound:-1] = 1

                att_c, att_w = self.attention(rnn_output.unsqueeze(dim=1),
                                              encoder_outputs.unsqueeze(0),
                                              mask)
                att_c = att_c.squeeze(dim=1)
                # step 3. concate s_i and c_i, and input to MLP
                mlp_input = torch.cat((rnn_output, att_c), dim=1)
                predicted_y_t = self.mlp(mlp_input)
                local_att_scores = F.log_softmax(predicted_y_t, dim=1)

                local_scores = local_att_scores

                if args.ctc_weight > 0:
                    #import pdb
                    #pdb.set_trace()
                    local_best_scores, local_best_ids = torch.topk(local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev']).cuda()
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                # topk scores
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1)

                for j in range(beam):
                    new_hyp = {}
                    new_hyp['h_prev'] = h_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_c[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(
                        local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    if args.ctc_weight > 0:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    hyps_best_kept.append(new_hyp)
                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'].append(self.eos_id)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos_id:
                    # hyp['score'] += (i + 1) * penalty
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if len(hyps) > 0:
                print('remeined hypothes: ' + str(len(hyps)))
            else:
                print('no hypothesis. Finish decoding.')
                break
            #import pdb
            #pdb.set_trace()
            for hyp in hyps:
                print('hypo: ' + ' '.join([char_list[int(x)]
                                          for x in hyp['yseq'][1:]]))
        # end for i in range(maxlen)
        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[
            :min(len(ended_hyps), nbest)]
        #print(nbest_hyps)
        return nbest_hyps

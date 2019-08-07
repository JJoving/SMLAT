import torch
import torch.nn as nn

from warpctc_pytorch import CTCLoss
import editdistance as ed

from lstm import Lstm

def LetterErrorRate_ctc(pred_y, label, label_len):
    #import pdb
    #pdb.set_trace()
    ed_accumalate = []
    true_y = []
    st = 0
    for l in label_len:
        true_y.append(label[st:st+l])
        st += l
    for p,t in zip(pred_y,true_y):
        compressed_t = [w for w in t if (w!=1 and w!=0 and w!=2)]
        #compressed_t = collapse_phn(compressed_t)
        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                continue
            if p_w == 2:
                break
            compressed_p.append(p_w)
            #compressed_p = collapse_phn(compressed_p)
            ed_accumalate.append(ed.eval(compressed_p,compressed_t)/len(compressed_t))
    return ed_accumalate

class Lstmctc(nn.Module):
    """Sequence-to-Sequence architecture with configurable encoder and decoder.
    """

    def __init__(self, lstm):
        super(Lstmctc, self).__init__()
        self.lstm = lstm
        self.ctc_loss = CTCLoss(blank=4233, size_average=True)

    def forward(self, padded_input, input_lengths, padded_target, target_lengths):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        import pdb
        pdb.set_trace()
        encoder_padded_outputs, _ = self.lstm(padded_input, input_lengths)
        predict = torch.max(encoder_padded_outputs,dim=-1)[1].cpu().data.numpy().reshape(encoder_padded_outputs.size(0),-1)
        pred = [[v for i,v in enumerate(x) if (i==0 or v != x[i-1]) and i<input_lengths[n] and x[i]!=4233] for n,x in enumerate(predict)]
        batch_wer = LetterErrorRate_ctc(pred, torch.cat([y[y != -1] for y in padded_target]), target_lengths)
        #loss = self.ctc_loss(encoder_padded_outputs, padded_target, input_lengths, target_lengths)
        #import pdb
        #pdb.set_trace()
        loss = self.ctc_loss(encoder_padded_outputs.transpose(0,1), torch.cat([y[y != -1] for y in padded_target]).cpu().int(), input_lengths.cpu().int(), target_lengths.cpu().int())
        return loss, batch_wer
        #return encoder_padded_outputs

    def recognize(self, input, input_length, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam

        Returns:
            nbest_hyps:
        """
        encoder_outputs, _ = self.encoder(input.unsqueeze(0), input_length)
        nbest_hyps = self.encoder.recognize_beam(encoder_outputs[0], char_list, args)
        return nbest_hyps

    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        lstm = Lstm(package['einput'],
                          package['ehidden'],
                          package['elayer'],
                          package['num_classes'],
                          dropout=package['edropout'],
                          bidirectional=package['ebidirectional'],
                          rnn_type=package['etype'])
        lstm.flatten_parameters()
        model = cls(lstm)
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # encoder
            'einput': model.lstm.input_size,
            'ehidden': model.lstm.hidden_size,
            'elayer': model.lstm.num_layers,
            'num_classes': model.lstm.num_classes,
            'edropout': model.lstm.dropout,
            'ebidirectional': model.lstm.bidirectional,
            'etype': model.lstm.rnn_type,
            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package

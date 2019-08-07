"""
Logic:
1. AudioDataLoader generate a minibatch from AudioDataset, the size of this
   minibatch is AudioDataLoader's batchsize. For now, we always set
   AudioDataLoader's batchsize as 1. The real minibatch size we care about is
   set in AudioDataset's __init__(...). So actually, we generate the
   information of one minibatch in AudioDataset.
2. After AudioDataLoader getting one minibatch from AudioDataset,
   AudioDataLoader calls its collate_fn(batch) to process this minibatch.
"""
import json

import numpy as np
import torch
import torch.utils.data as data
import random
from torch.autograd import Variable

import kaldi_io
from utils import IGNORE_ID, pad_list


class AudioDataset(data.Dataset):
    """
    TODO: this is a little HACK now, put batch_size here now.
          remove batch_size to dataloader later.
    """

    def __init__(self, data_json_path, batch_size, max_length_in, max_length_out,
                 num_batches=0):
        # From: espnet/src/asr/asr_utils.py: make_batchset()
        """
        Args:
            data: espnet/espnet json format file.
            num_batches: for debug. only use num_batches minibatch but not all.
        """
        super(AudioDataset, self).__init__()
        with open(data_json_path, 'rb') as f:
            data = json.load(f)['utts']
        # sort it by input lengths (long to short)
        sorted_data = sorted(data.items(), key=lambda data: int(
            data[1]['input'][0]['shape'][0]), reverse=True)
        # change batchsize depending on the input and output length
        minibatch = []
        start = 0
        while True:
            ilen = int(sorted_data[start][1]['input'][0]['shape'][0])
            olen = int(sorted_data[start][1]['output'][0]['shape'][0])
            factor = max(int(ilen / max_length_in), int(olen / max_length_out))
            # if ilen = 1000 and max_length_in = 800
            # then b = batchsize / 2
            # and max(1, .) avoids batchsize = 0
            b = max(1, int(batch_size / (1 + factor)))
            end = min(len(sorted_data), start + b)
            minibatch.append(sorted_data[start:end])
            if end == len(sorted_data):
                break
            start = end
        if num_batches > 0:
            minibatch = minibatch[:num_batches]
        self.minibatch = minibatch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class AudioDataLoader(data.DataLoader):
    """
    NOTE: just use batchsize=1 here, so drop_last=True makes no sense here.
    """

    def __init__(self, *args, LFR_m=1, LFR_n=1, align_trun=0, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = LFRCollate(LFR_m=LFR_m, LFR_n=LFR_n, align_trun=align_trun)
        #self.collate_fn = _collate_fn

class LFRCollate(object):
    """Build this wrapper to pass arguments(LFR_m, LFR_n) to _collate_fn"""
    def __init__(self, LFR_m=1, LFR_n=1, align_trun=0):
        self.LFR_m = LFR_m
        self.LFR_n = LFR_n
        self.align_trun = align_trun

    def __call__(self, batch):
        return _collate_fn(batch, LFR_m=self.LFR_m, LFR_n=self.LFR_n, align_trun = self.align_trun)


# From: espnet/src/asr/asr_pytorch.py: CustomConverter:__call__
def _collate_fn(batch, LFR_m=1, LFR_n=1, align_trun = 0):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        xs_pad: N x Ti x D, torch.Tensor
        ilens : N, torch.Tentor
        ys_pad: N x To, torch.Tensor
    """
    # batch should be located in list
    assert len(batch) == 1
    #import pdb
    #pdb.set_trace()
    batch = load_inputs_and_targets(batch[0], LFR_m=LFR_m, LFR_n=LFR_n, align_trun = align_trun)
    xs, ys, aligns = batch
    #print(xs.size(), ys.size(), align.size())
    #print(xs[0][0])
    #print(align)
    # TODO: perform subsamping

    # get batch of lengths of input sequences
    ilens = np.array([x.shape[0] for x in xs])
    olens = np.array([y.shape[0] for y in ys])
    # perform padding and convert to tensor
    xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0)
    ilens = torch.from_numpy(ilens)
    ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], IGNORE_ID)
    if aligns:
        ys_pad, aligns_pad, olens = align_process(aligns)
        #align_pad = pad_list([torch.from_numpy(y).long() for y in align], IGNORE_ID)
        #return xs_pad, ilens, ys_pad, olens, aligns_pad
    else:
        aligns_pad = torch.from_numpy( np.asarray(aligns))
        #return xs_pad, ilens, ys_pad, olens, aligns_pad
    olens = torch.from_numpy(olens)
    #print(xs_pad.size(), ys_pad.size(), align_pad.size())
    return xs_pad, ilens, ys_pad, olens, aligns_pad


# ------------------------------ utils ------------------------------------
def load_inputs_and_targets(batch, LFR_m=1, LFR_n=1, align_trun=0):
    # From: espnet/src/asr/asr_utils.py: load_inputs_and_targets
    # load acoustic features and target sequence of token ids
    xs = [kaldi_io.read_mat(b[1]['input'][0]['feat']) for b in batch]
    ys = [b[1]['output'][0]['tokenid'].split() for b in batch]

    if LFR_m != 1 or LFR_n != 1:
        # xs = build_LFR_features(xs, LFR_m, LFR_n)
        xs = [build_LFR_features(x, LFR_m, LFR_n) for x in xs]

    aligns = []
    if align_trun:
        aligns = [b[1]['output'][0]['ctcid'].split() for b in batch]

    spec = 0
    ys_spec = ys
    xs_ori = xs
    #if spec:
    #    xs=xs_spec=spec_augment(xs_ori,frequency_mask_num=1)
    #    xs=np.concatenate((xs,xs_spec),axis=0)
    #    ys=np.concatenate((ys,ys_spec),axis=0)
    #if spec:
    #    xs_spec=spec_augment(xs_ori,time_mask_num=1)
    #    xs=np.concatenate((xs,xs_spec),axis=0)
    #    ys=np.concatenate((ys,ys_spec),axis=0)
    #import pdb
    #pdb.set_trace()
    if spec:
         xs=xs_spec=spec_augment(xs_ori,time_mask_num=1,frequency_mask_num=1, frequency_masking_para=27, time_masking_para=70)
         #xs=np.concatenate((xs,xs_spec),axis=0)
         #ys=np.concatenate((ys,ys_spec),axis=0)
    #print(np.size(xs[0]),xs[0])
    #print(len(xs))
    #print(np.size(xs_spec[0]),xs_spec[0])
    # get index of non-zero length samples
    nonzero_idx = filter(lambda i: len(ys[i]) > 0, range(len(xs)))
    # sort in input lengths
    nonzero_sorted_idx = sorted(nonzero_idx, key=lambda i: -len(xs[i]))
    if len(nonzero_sorted_idx) != len(xs):
        print("warning: Target sequences include empty tokenid")

    # remove zero-lenght samples
    xs = [xs[i] for i in nonzero_sorted_idx]
    ys = [np.fromiter(map(int, ys[i]), dtype=np.int64)
          for i in nonzero_sorted_idx]
    if align_trun:
        aligns = [np.fromiter(map(int, aligns[i]), dtype=np.int64)
               for i in nonzero_sorted_idx]
       #return xs, ys, ctcs

    return xs, ys, aligns

def build_LFR_features(inputs, m, n):
    """
    Actually, this implements stacking frames and skipping frames.
    if m = 1 and n = 1, just return the origin features.
    if m = 1 and n > 1, it works like skipping.
    if m > 1 and n = 1, it works like stacking but only support right frames.
    if m > 1 and n > 1, it works like LFR.

    Args:
        inputs_batch: inputs is T x D np.ndarray
        m: number of frames to stack
        n: number of frames to skip
    """
    # LFR_inputs_batch = []
    # for inputs in inputs_batch:
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i*n:i*n+m]))
        else: # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i*n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    return np.vstack(LFR_inputs)
    #     LFR_inputs_batch.append(np.vstack(LFR_inputs))
    # return LFR_inputs_batch

def spec_augment(batch, frequency_mask_num=0, time_mask_num=0, frequency_masking_para=27, time_masking_para=70):
    #frequency_mask_num = 1
    #time_mask_num = 1
    #frequency_masking_para = 27
    #time_masking_para = 70
    #ilens = np.array([x.shape[0] for x in batch])
    #dim = batch[0].shape[2]
    for x in batch:
        #print(x.shape,x)
        v = x.shape[1]
        tau = x.shape[0]
        # Step 1 : Time warping (give up)
        # Step 2 : Frequency masking
        for i in range(frequency_mask_num):
            f = np.random.uniform(low=0.0, high=min(frequency_masking_para, v))
            f = int(f)
            if f > 0:
                f0 = random.randint(0, v - f)
                x[:, f0:f0 + f] = 0
        # Step 3 : Time masking
        for i in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=min(time_masking_para, tau))
            t = int(t)
            if t > 0:
                t0 = random.randint(0, tau - t)
                x[t0:t0 + t, :] = 0
    return batch

def spec_augment_1(batch, frequency_mask_num=0, time_mask_num=0, frequency_masking_para=27, time_masking_para=70):
    for x in batch:
        v = x.shape[1]
        tau = x.shape[0]
        f0 = int(v/(frequency_mask_num+1))
        f = int(frequency_masking_para / frequency_mask_num)
        if f0 > f:
            for i in range(frequency_mask_num):
                x[:, f0*(i+1):f0*(i+1) + f] = 0
        t0 = int(tau/(time_mask_num+1))
        t = int(time_masking_para / time_mask_num)
        if t0 > t:
            for i in range(time_mask_num):
                x[t0*(i+1):t0*(i+1) + t, :] = 0
    return batch


def ctc_truncate(ctc_output, padded_target, encoder_padded_outputs, blank = 0):
    blank_id = blank
    ys = [y [y != -1] for y in padded_target]
    ctc_greedy = torch.max(ctc_output, dim=-1)[1]
    hpad_trunc=[]
    ctc_ys_trunc=[]
    xs_trunc=[]
    ys_trunc=[]
    for k in range(ctc_greedy.size()[0]):
        h_trunc = [blank_id,]
        ctc_y_trunc=[]
        len = ctc_greedy.size()[1]
        for i in range(1, len):
            if ctc_greedy[k][i-1] != ctc_greedy[k][i] and ctc_greedy[k][i-1] != blank_id:
                h_trunc.append(i)
                ctc_y_trunc.append(ctc_greedy[k][i-1])
        if ctc_greedy[k][len - 1] == blank_id:
            h_trunc[-1] = len
        else:
            h_trunc.append(len)
        hpad_trunc.append(h_trunc)
        ctc_ys_trunc.append(ctc_y_trunc)

        matrix = [[i+j for j in range(np.size(ctc_ys_trunc[k]) + 1)] for i in range(ys[k].size()[0] + 1)]
        matrix_per = [[-1 for j in range(np.size(ctc_ys_trunc[k]) + 1)] for i in range(ys[k].size()[0] + 1)]
        for i in range(1,ys[k].size()[0] + 1):
                for j in range(1,np.size(ctc_ys_trunc[k]) + 1):
                    if ctc_ys_trunc[k][j-1] == ys[k][i-1]:
                        d = 0
                    else:
                        d = 1
                    min_dist = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
                    matrix[i][j] = min_dist
                    if matrix[i-1][j] == min_dist - 1:
                        matrix_per[i][j] = 1
                    elif matrix[i][j-1] == min_dist - 1:
                        matrix_per[i][j] = 2
                    else:
                        matrix_per[i][j] = 0
        dist = []
        i = ys[k].size()[0]
        j = np.size(ctc_ys_trunc[k])
        rx = j
        ry = i
        while (i >= 1 and j >= 1):
                print(i,j)
                if matrix_per[i][j] == 0:
                    i = i - 1
                    j = j - 1
                    xs_trunc.append(encoder_padded_outputs[k][hpad_trunc[k][j]:hpad_trunc[k][rx]])
                    ys_trunc.append(ys[k][i:ry])
                    rx = j
                    ry = i
                elif matrix_per[i][j] == 1:
                    i = i - 1
                elif matrix_per[i][j] == 2:
                    j = j - 1
        #print(np.size(ys_trunc),np.size(xs_trunc))
        print("----------")
    return xs_trunc,ys_trunc

def align_truncate(align, padded_target, encoder_padded_outputs, blank = 0):
    align = [y[y != IGNORE_ID] for y in align]
    xs_trunc=[]
    ys_trunc=[]
    for k in range(len(align)):
        lens = len(align[k])
        lid = 0
        for i in range(1, lens):
            if align[k][i-1] != align[k][i]:
                if align[k][i-1] != 0:
                    xs_trunc.append(encoder_padded_outputs[k][lid:i])
                    ys_trunc.append(align[k][i-1])

                lid = i
            if i == lens-1 and align[k][i] != 0:
                xs_trunc.append(encoder_padded_outputs[k][lid:lens])
                ys_trunc.append(align[k][i])
    xs_pad = pad_list(xs_trunc, 0)
    ys_pad = pad_list((torch.from_numpy( np.asarray(ys_trunc)).unsqueeze(-1).cuda()), IGNORE_ID)
    #return xs_trunc, ys_trunc
    return xs_pad, ys_pad

def align_process(align):
    align = [y[y != IGNORE_ID] for y in align]
    truns=[]
    ys_truns=[]
    for k in range(len(align)):
        lens = len(align[k])
        trun=[0,]
        ys=[]
        for i in range(1, lens):
            if align[k][i-1] != align[k][i]:
                if align[k][i-1] != 0:
                    trun.append(i)
                    ys.append(align[k][i-1])
                lid = i
                if i == lens-1 and align[k][i] != 0:
                    trun.append(lens)
                    ys.append(align[k][i])
        truns.append(trun)
        ys_truns.append(ys)
    olnes = np.array([len(y) for y in ys_truns])
    aligns_pad = pad_list([torch.from_numpy(np.asarray(x)) for x in truns],IGNORE_ID)
    ys_pad = pad_list([torch.from_numpy(np.asarray(x)) for x in ys_truns],IGNORE_ID)
    return ys_pad, aligns_pad, olnes

def align_process_2(align):
    align = [y[y != IGNORE_ID] for y in align]
    left_trun=[]
    right_trun=[]
    ys_trunc=[]
    for k in range(len(align)):
        lens = len(align[k])
        lid = 0
        left=[]
        right=[]
        ys=[]
        for i in range(1, lens):
            if align[k][i-1] != align[k][i] and align[k][i-1] != 0:
                left.append(lid)
                right.append(i)
                ys.append(align[k][i-1])
                lid = i
            if i == lens-1 and align[k][i] != 0:
                left.append(lid)
                right.append(lens)
                ys.append(align[k][i])
        left_trun.append(left)
        right_trun.append(right)
        ys_trunc.append(ys)
    left_pad = pad_list([torch.from_numpy(np.asarray(x)) for x in left_trun],IGNORE_ID)
    right_pad = pad_list([torch.from_numpy(np.asarray(x)) for x in right_trun],IGNORE_ID)
    ys_pad = pad_list([torch.from_numpy(np.asarray(x)) for x in ys_trunc],IGNORE_ID)
    return ys_pad, left_pad, right_pad

def load_ctc_align(path, path2=""):
    f = open(path, 'r')
    ctc_align = {}
    for line in f.readlines():
        line = line.strip().split()
        ctc_align[line[0]]=line[1:-1]
    f.close()
    if path2:
        f = open(path, 'r')
        for line in f.readlines():
            line = line.strip().split()
            ctc_align[line[0]]=line[1:-1]
        f.close()
    return ctc_align

def ctc_to_align(lpz):
    ctc_greedy = torch.max(lpz, dim=-1)[1]
    aligns = []
    for k in range(ctc_greedy.size()[0]):
        aligns.append(append(torch.nonzero(ctc_greedy).reshape(-1).cpu().numpy().tolist()))
    return aligns




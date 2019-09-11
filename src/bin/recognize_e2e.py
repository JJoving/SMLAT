#!/usr/bin/env python
import argparse
import json

import torch
import random
import numpy as np

random.seed(1
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import kaldi_io
from e2e import Seq2Seq
from utils import add_results_to_json, process_dict
from data import build_LFR_features

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(
    "End-to-End Automatic Speech Recognition Decoding.")
# data
parser.add_argument('--recog_json', type=str, required=True,
                    help='Filename of recognition data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
parser.add_argument('--result_label', type=str, required=True,
                    help='Filename of result label data (json)')
# model
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model file created by training')
# decode
parser.add_argument('--beam_size', default=1, type=int,
                    help='Beam size')
parser.add_argument('--nbest', default=1, type=int,
                    help='Nbest size')
parser.add_argument('--decode_max_len', default=0, type=int,
                    help='Max output length. If ==0 (default), it uses a '
                    'end-detect function to automatically find maximum '
                    'hypothesis lengths')
parser.add_argument('--ctc_weight', default=0, type=float,
                    help='ctc_weight')
parser.add_argument('--mode', default=0.5, type=float,
                    help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss')
parser.add_argument('--trun', default=False, type=str2bool,
                     help='use trun')
parser.add_argument('--align_trun', default=False, type=str2bool,
                     help='use align trun')

def recognize(args):
    #import pdb
    #pdb.set_trace()

    char_list, sos_id, eos_id = process_dict(args.dict)
    args.char_list = char_list
    model, LFR_m, LFR_n  = Seq2Seq.load_model(args.model_path, args)
    print(model)
    model.eval()
    model.cuda()
    char_list, sos_id, eos_id = process_dict(args.dict)
    assert model.decoder.sos_id == sos_id and model.decoder.eos_id == eos_id

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            print('(%d/%d) decoding %s' %
                  (idx, len(js.keys()), name), flush=True)
            input = kaldi_io.read_mat(js[name]['input'][0]['feat'])  # TxD
            input = build_LFR_features(input, LFR_m, LFR_n)
            input = torch.from_numpy(input).float()
            input_length = torch.tensor([input.size(0)], dtype=torch.int)
            input = input.cuda()
            input_length = input_length.cuda()
            if args.align_trun:
                align = (js[name]['output'][0]['ctcid'].split())
                nbest_hyps = model.recognize_align(input, input_length, char_list, align, args)
            else:
                nbest_hyps = model.recognize(input, input_length, char_list, args)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, char_list)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4,
                           sort_keys=True).encode('utf_8'))


if __name__ == "__main__":
    args = parser.parse_args()
    print(args, flush=True)
    recognize(args)

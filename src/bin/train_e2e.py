#!/usr/bin/env python
import argparse

import torch

from data import AudioDataLoader, AudioDataset
from decoder_e2e import Decoder
from encoder_e2e import Encoder
from ctc import CTC
from e2e import Seq2Seq
from solver import Solver
from utils import process_dict

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
    "End-to-End Automatic Speech Recognition Training "
    "(Listen Attend and Spell framework).")
# General config
# Task related
parser.add_argument('--train_json', type=str, default=None,
                    help='Filename of train label data (json)')
parser.add_argument('--valid_json', type=str, default=None,
                    help='Filename of validation label data (json)')
parser.add_argument('--dict', type=str, required=True,
                    help='Dictionary which should include <unk> <sos> <eos>')
# Low Frame Rate (stacking and skipping frames)
parser.add_argument('--LFR_m', default=4, type=int,
                    help='Low Frame Rate: number of frames to stack')
parser.add_argument('--LFR_n', default=3, type=int,
                    help='Low Frame Rate: number of frames to skip')
# Network architecture
# encoder
# TODO: automatically infer input dim
parser.add_argument('--einput', default=80, type=int,
                    help='Dim of encoder input')
parser.add_argument('--ehidden', default=512, type=int,
                    help='Size of encoder hidden units')
parser.add_argument('--elayer', default=4, type=int,
                    help='Number of encoder layers.')
parser.add_argument('--edropout', default=0.0, type=float,
                    help='Encoder dropout rate')
parser.add_argument('--ebidirectional', default=1, type=int,
                    help='Whether use bidirectional encoder')
parser.add_argument('--etype', default='lstm', type=str,
                    help='Type of encoder RNN')
# attention
parser.add_argument('--atype', default='dot', type=str,
                    help='Type of attention (Only support Dot Product now)')
# decoder
parser.add_argument('--dembed', default=512, type=int,
                    help='Size of decoder embedding')
parser.add_argument('--dhidden', default=512*2, type=int,
                    help='Size of decoder hidden units. Should be encoder '
                    '(2*) hidden size dependding on bidirection')
parser.add_argument('--dlayer', default=1, type=int,
                    help='Number of decoder layers.')
parser.add_argument('--ctc_weight', default=0, type=float,
                    help='ctc_weight')
parser.add_argument('--offset', default=0, type=int,
                    help='Number of align attention offset.')

# Training config
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when halving lr but still get'
                    'small improvement')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
parser.add_argument('--mode', default=0.5, type=float,
                    help='Multitask learning coefficient, alpha: alpha*ctc_loss + (1-alpha)*att_loss')
parser.add_argument('--half_lr_epoch', dest='half_lr_epoch', default=0, type=int,
                    help='Halving learning rate epoch at least')
# minibatch
parser.add_argument('--batch_size', '-b', default=32, type=int,
                    help='Batch size')
parser.add_argument('--maxlen_in', default=800, type=int, metavar='ML',
                    help='Batch size is reduced if the input sequence length > ML')
parser.add_argument('--maxlen_out', default=150, type=int, metavar='ML',
                    help='Batch size is reduced if the output sequence length > ML')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
parser.add_argument('--align_trun', default=False, type=str2bool,
                     help='use align trun')
parser.add_argument('--ctc_trun', default=False, type=str2bool,
                     help='use ctc trun')
parser.add_argument('--trun', default=False, type=str2bool,
                     help='use trun')
parser.add_argument('--lsm_weight', default=0.0, type=float,
                    help='lsm_weight')
parser.add_argument('--sampling_probability', default=0.0, type=float,
                    help='sampling_probability')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--ctc_model', default='',
                    help='CTC model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_id', default='LAS training',
                    help='Identifier for visdom run')


def main(args):
    # Construct Solver
    # data
    tr_dataset = AudioDataset(args.train_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out)
    cv_dataset = AudioDataset(args.valid_json, args.batch_size,
                              args.maxlen_in, args.maxlen_out)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n,
                                align_trun=args.align_trun)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers,
                                LFR_m=args.LFR_m, LFR_n=args.LFR_n,
                                align_trun=args.align_trun)
    # load dictionary and generate char_list, sos_id, eos_id
    char_list, sos_id, eos_id = process_dict(args.dict)
    args.char_list = char_list
    vocab_size = len(char_list)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    #import pdb
    #pdb.set_trace()
    encoder = Encoder(args.einput * args.LFR_m, args.ehidden, args.elayer, vocab_size,
                      dropout=args.edropout, bidirectional=args.ebidirectional,
                      rnn_type=args.etype)
    decoder = Decoder(vocab_size, args.dembed, sos_id,
                      eos_id, args.dhidden, args.dlayer, args.offset, args.atype,
                      dropout=args.edropout,lsm_weight=args.lsm_weight,sampling_probability=args.sampling_probability,
                      bidirectional_encoder=args.ebidirectional)
    if args.ebidirectional:
        eprojs = args.ehidden * 2
    else:
        eprojs = args.ehidden
    ctc = CTC(odim = vocab_size,eprojs = eprojs, dropout_rate = args.edropout)
    #lstm_model = Lstmctc.load_model(args.continue_from)

    model = Seq2Seq(encoder, decoder, ctc, args)
    #model_dict = model.state_dict()
    print(model)
    #print(lstm_model)
    #pretrained_dict = torch.load(args.ctc_model)
    #pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    #pretrained_dict = {(k.replace('lstm','encoder')):v for k, v in pretrained_dict['state_dict'].items() if (k.replace('lstm','encoder')) in model_dict}
    #model_dict.update(pretrained_dict)
    #model.load_state_dict(model_dict)
    #for k,v in model.named_parameters():
    #    if k.startswith("encoder"):
    #        print(k)
    #        v.requires_grad=False
    model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr,
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return

    # solver
    ctc = 0
    solver = Solver(data, model, optimizier, args)
    solver.train()

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

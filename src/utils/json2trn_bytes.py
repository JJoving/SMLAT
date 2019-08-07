#!/usr/bin/env python
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
from utils import process_dict

def not_empty(s):
    return s and s.strip()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='json files')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('ref', type=str, help='ref')
    parser.add_argument('hyp', type=str, help='hyp')
    parser.add_argument('ref_', type=str, help='ref_')
    parser.add_argument('hyp_', type=str, help='hyp_')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    logging.info("reading %s", args.json)
    with open(args.json, 'r') as f:
        j = json.load(f)

    logging.info("reading %s", args.dict)
    char_list, sos_id, eos_id = process_dict(args.dict)
    # with open(args.dict, 'r') as f:
    #     dictionary = f.readlines()
    # char_list = [unicode(entry.split(' ')[0], 'utf_8') for entry in dictionary]
    # char_list.insert(0, '<blank>')
    # char_list.append('<eos>')
    # print([x.encode('utf-8') for x in char_list])

    logging.info("writing hyp trn to %s", args.hyp)
    logging.info("writing ref trn to %s", args.ref)
    h = open(args.hyp, 'w', encoding="utf-8")
    r = open(args.ref, 'w', encoding="utf-8")

    h_ = open(args.hyp_, 'w', encoding="utf-8")
    r_ = open(args.ref_, 'w', encoding="utf-8")

    for x in j['utts']:
        seq = [int(i) for i in j['utts'][x]
               ['output'][0]['rec_tokenid'].split()]
        #seq = " ".join(str(i) for i in seq).replace(str(eos_id), '')
        while len(seq) > 0 and seq[-1] == eos_id :
            seq.pop()
        #print(seq)
        h.write(" ".join(str(i) for i in seq)),
        h.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")
        h_.write(" ".join(bytes([i for i in seq]).decode('utf-8','ignore'))),
        h_.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")

        seq = [int(i) for i in j['utts'][x]
               ['output'][0]['tokenid'].split()]
        #seq = " ".join(str(i) for i in seq).replace(str(eos_id), '')
        #print(seq)
        if seq[-1] == eos_id:
            seq.pop()
        r.write(" ".join(str(i) for i in seq)),
        r.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")
        r_.write(" ".join(bytes([i for i in seq]).decode('utf-8','ignore'))),
        r_.write(
            " (" + j['utts'][x]['utt2spk'].replace('-', '_') + "-" + x + ")\n")


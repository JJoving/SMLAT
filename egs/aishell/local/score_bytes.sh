#!/bin/bash

[ -f path.sh ] && . ./path.sh

nlsyms=""

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>"
    exit 1
fi

dir=$1
dic=$2

json2trn_bytes.py ${dir}/data.json ${dic} ${dir}/ref_ber.trn ${dir}/hyp_ber.trn ${dir}/ref_cer.trn ${dir}/hyp_cer.trn

#if [ ! -z ${nlsyms} ]; then
#  cp ${dir}/ref.trn ${dir}/ref.trn.org
#  cp ${dir}/hyp.trn ${dir}/hyp.trn.org
#  filt.py -v $nlsyms ${dir}/ref.trn.org > ${dir}/ref.trn
#  filt.py -v $nlsyms ${dir}/hyp.trn.org > ${dir}/hyp.trn
#fi

sclite -r ${dir}/ref_ber.trn trn -h ${dir}/hyp_ber.trn trn -i rm -o all stdout > ${dir}/result_ber.txt
sclite -r ${dir}/ref_cer.trn trn -h ${dir}/hyp_cer.trn trn -i rm -o all stdout > ${dir}/result_cer.txt

echo "write a CER (or TER) result in ${dir}/result.txt"
grep -e Avg -e SPKR -m 2 ${dir}/result_ber.txt
grep -e Avg -e SPKR -m 2 ${dir}/result_cer.txt

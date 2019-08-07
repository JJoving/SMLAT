#!/bin/bash

# -- IMPORTANT
data=/search/speech/chenjunjie/data/aishell1 # Modify to your aishell data path
stage=3 # Modify to control start from witch stage
# --

ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=40

dumpdir=dump_1kh   # directory to dump full features
# Feature configuration
do_delta=false
splice=true
train_info=""

# Network architecture
# Encoder
einput=40
ehidden=256
elayer=3
edropout=0.2
ebidirectional=0
etype=lstm
# Attention
atype=dot
# Decoder
dembed=512
dhidden=512
dlayer=1

# Training config
epochs=20
half_lr=1
early_stop=0
max_norm=5
batch_size=128
maxlen_in=800
maxlen_out=150
# optimizer
optimizer=adam
lr=1e-3
momentum=0
l2=1e-5
# logging and visualize
checkpoint=0
continue_from=""
print_freq=10
visdom=0
visdom_id="LAS Training"

# Decode config
beam_size=10
nbest=1
decode_max_len=100

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
    echo "Stage 0: Data Preparation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    # Generate wav.scp, text, utt2spk, spk2utt (segments)
    local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;
    # remove space in text
    for x in train test dev; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
    done
fi

feat_train_dir=${dumpdir}/train/delta${do_delta}; mkdir -p ${feat_train_dir}
feat_test_dir=${dumpdir}/test/delta${do_delta}; mkdir -p ${feat_test_dir}
feat_dev_dir=${dumpdir}/dev/delta${do_delta}; mkdir -p ${feat_dev_dir}

if [ $stage -le 1 ]; then
    echo "Stage 1: Feature Generation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    #fbankdir=fbank
    #for data in train test dev; do
    #    steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
    #        data/$data exp/make_fbank/$data $fbankdir/$data || exit 1;
    #done
    # compute global CMVN
    compute-cmvn-stats scp:data_1kh/train/feats.scp data_1kh/train/cmvn.ark
    # dump features for training
    for data in train dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
            data_1kh/$data/feats.scp data_1kh/train/cmvn.ark exp/dump_feats/$data $feat_dir
    done
    for data in 8000 ios nos; do
      mkdir -p $feat_test_dir/$data
      dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
        data_1kh/test/${data}/feats.scp data_1kh/train/cmvn.ark exp/dump_feats/$data $feat_test_dir/$data
    done
fi

dict=data_1kh/lang_1char/train_chars.txt
echo "dictionary: ${dict}"
nlsyms=data_1kh/lang_1char/non_lang_syms.txt
if [ $stage -le 2 ]; then
    echo "Stage 2: Dictionary and Json Data Preparation"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p data_1kh/lang_1char/

    echo "make a non-linguistic symbol list"
    # It's empty in AISHELL-1
    cut -f 2- data_1kh/train/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 0" >  ${dict}
    echo "<sos> 1" >> ${dict}
    echo "<eos> 2" >> ${dict}
    text2token.py -s 1 -n 1 -l ${nlsyms} data_1kh/train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for data in train dev; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        data2json.sh --feat ${feat_dir}/feats.scp --nlsyms ${nlsyms} \
             data_1kh/$data ${dict} > ${feat_dir}/data.json
    done
    for data in 8000 ios nos; do
      data2json.sh --feat $feat_test_dir/${data}/feats.scp --nlsyms ${nlsyms} \
        data_1kh/test/${data} ${dict} > ${feat_test_dir}/${data}/data.json
    done
fi

if [ $ebidirectional -eq 0 ]; then
  ehidden=$[ehidden*2]
  echo $ebidirectional
fi
echo $ebidirectional
echo $ehidden

if [ -z ${tag} ]; then
    expdir=exp_1kh/bidirectionaltrain${ebidirectional}_in${einput}_hidden${ehidden}_e${elayer}_${etype}_drop${edropout}_${atype}_emb${dembed}_hidden${dhidden}_d${dlayer}_epoch${epochs}_norm${max_norm}_bs${batch_size}_mli${maxlen_in}_mlo${maxlen_out}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
    if $splice; then
      expdir=${expdir}_splice
    fi
    if [ -n ${train_info} ]; then
      expdir=${expdir}_${train_info}
    fi
else
    expdir=exp_1kh/${tag}
fi

mkdir -p ${expdir}

if [ ${stage} -le 3 ]; then
    echo "Stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        train.py \
        --train_json ${feat_train_dir}/data.json \
        --valid_json ${feat_dev_dir}/data.json \
        --dict ${dict} \
        --einput $einput \
        --ehidden $ehidden \
        --elayer $elayer \
        --edropout $edropout \
        --ebidirectional $ebidirectional \
        --etype $etype \
        --atype $atype \
        --dembed $dembed \
        --dhidden $dhidden \
        --dlayer $dlayer \
        --epochs $epochs \
        --half_lr $half_lr \
        --early_stop $early_stop \
        --max_norm $max_norm \
        --batch_size $batch_size \
        --maxlen_in $maxlen_in \
        --maxlen_out $maxlen_out \
        --optimizer $optimizer \
        --lr $lr \
        --momentum $momentum \
        --l2 $l2 \
        --save_folder ${expdir} \
        --checkpoint $checkpoint \
        --continue_from "$continue_from" \
        --print_freq ${print_freq} \
        --visdom $visdom \
        --visdom_id "$visdom_id"
fi

if [ ${stage} -le 4 ]; then
    echo "Stage 4: Decoding"
  for data in 8000 ios nos dev; do
    decode_dir=${expdir}/decode_test_beam${beam_size}_nbest${nbest}_ml${decode_max_len}_test_set${data}
    mkdir -p ${decode_dir}
    ${cuda_cmd} --gpu ${ngpu} ${decode_dir}/decode.log \
        recognize.py \
        --recog_json ${feat_test_dir}/${data}/data.json \
        --dict $dict \
        --result_label ${decode_dir}/data.json \
        --model_path ${expdir}/final.pth.tar \
        --beam_size $beam_size \
        --nbest $nbest \
        --decode_max_len $decode_max_len

    # Compute CER
    local/score.sh --nlsyms ${nlsyms} ${decode_dir} ${dict}
  done
fi

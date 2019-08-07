#!/bin/bash

# -- IMPORTANT
#data=/search/speech/chenjunjie/data/aishell1 # Modify to your aishell data path
stage=3 # Modify to control start from witch stage
data_set=2500h
data_path=/search/speech/chenjunjie/github/Listen-Attend-Spell/egs/aishell/data/$data_set
no_cv=false
#手动建立data_path，把feats.scp和text复制过来
#mkdir -p $data_path
# --

ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=40

dumpdir=dump_$data_set   # directory to dump full features
# Feature configuration
do_delta=false
LFR_m=1
LFR_n=1
lsm_weight=0
sampling_probability=0
splice=false
train_info=""

# Network architecture
# Encoder
einput=71
ehidden=512
elayer=3
edropout=0.2
ebidirectional=1
etype=lstm
# Attention
atype=dot
# Decoder
dembed=512
dhidden=512
dlayer=2

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
mode=0.5
trun=False
offset=0
ctc_weight=0.3

# Decode config
beam_size=10
nbest=1
decode_max_len=100

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;
. ./cmd.sh
. ./path.sh

if [ $ebidirectional -eq 1 ]; then
  echo $ehidden
  ehidden=$[ehidden/2]
fi
echo $ehidden
if $do_delta; then
  einput=$[einput*3]
fi

if [ $stage -le 0 ]; then
    echo "Stage 0: Data Preparation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    # Generate wav.scp, text, utt2spk, spk2utt (segments)
    local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;
    # remove space in text
    for x in train test cv; do
        cp data/${x}/text data/${x}/text.org
        paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
            > data/${x}/text
    done
fi

feat_train_dir=dump/${dumpdir}/train/delta${do_delta}; mkdir -p ${feat_train_dir}
feat_test_dir=dump/${dumpdir}/test/delta${do_delta}; mkdir -p ${feat_test_dir}
feat_cv_dir=dump/${dumpdir}/cv/delta${do_delta}; mkdir -p ${feat_cv_dir}

if [ $stage -le 1 ]; then
    echo "Stage 1: Feature Generation"
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    #fbankdir=fbank
    #for data in train test cv; do
    #    steps/make_fbank.sh --cmd "$train_cmd" --nj $nj --write_utt2num_frames true \
    #        data/$data exp/make_fbank/$data $fbankdir/$data || exit 1;
    #done

    if $no_cv; then
      mkdir -p $data_path/train
      mkdir -p $data_path/cv
      ./utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $data_path $data_path/train/ $data_path/cv/
      filter_scp.pl $data_path/train/utt2spk < $data_path/feat.scp >$data_path/train/feats.scp
    fi

    # compute global CMVN
    compute-cmvn-stats scp:$data_path/train/feats.scp $data_path/train/cmvn.ark
    # dump features for training
    for data in train cv; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
            $data_path/$data/feats.scp $data_path/train/cmvn.ark exp/dump_feats/$data $feat_dir
    done
    #for data in 8000 ios nos; do
    #  mkdir -p $feat_test_dir/$data
    #dump.sh --cmd "$train_cmd" --nj $nj --do_delta $do_delta \
    #    data_1kh/test/${data}/feats.scp data_1kh/train/cmvn.ark exp/dump_feats/$data $feat_test_dir/$data
    #done
fi

dict=$data_path/lang_1char/train_chars_ctc.txt
echo "dictionary: ${dict}"
nlsyms=$data_path/lang_1char/non_lang_syms.txt
if [ $stage -le 2 ]; then
    echo "Stage 2: Dictionary and Json Data Preparation"
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    mkdir -p $data_path/lang_1char/

    echo "make a non-linguistic symbol list"
    # It's empty in AISHELL-1
    cut -f 2- $data_path/train/text | grep -o -P '\[.*?\]' | sort | uniq > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 0" >  ${dict}
    echo "<sos> 1" >> ${dict}
    echo "<eos> 2" >> ${dict}
    text2token.py -s 1 -n 1 -l ${nlsyms} $data_path/train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+2}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    for data in train cv; do
        feat_dir=`eval echo '$feat_'${data}'_dir'`
        data2json.sh --feat ${feat_dir}/feats.scp --nlsyms ${nlsyms} \
             $data_path/$data ${dict} > ${feat_dir}/data.json
    done

    #for data in 8000 ios nos; do
    #  data2json.sh --feat $feat_test_dir/${data}/feats.scp --nlsyms ${nlsyms} \
    #    data_1kh/test/${data} ${dict} > ${feat_test_dir}/${data}/data.json
    #done
fi

if [ -z ${tag} ]; then
  expdir=exp/$data_set/train_in${einput}_hidden${ehidden}_e${elayer}_${etype}_drop${edropout}_${atype}_emb${dembed}_hidden${dhidden}_d${dlayer}_epoch${epochs}_norm${max_norm}_bs${batch_size}_mli${maxlen_in}_mlo${maxlen_out}_${optimizer}_lr${lr}_mmt${momentum}_l2${l2}_bidirectionaltrain${ebidirectional}_mode${mode}_trun${trun}_offset${offset}_m${LFR_m}_n${LFR_n}_lsm${lsm_weight}_ss${sampling_probability}
    if ${do_delta}; then
        expdir=${expdir}_delta
    fi
else
    expdir=exp/$data_set/train_${tag}
fi

if $splice; then
  expdir=${expdir}_splice
fi
if [ -n ${train_info} ]; then
  expdir=${expdir}_${train_info}
fi

mkdir -p ${expdir}
echo $expdir
if [ ${stage} -le 3 ]; then
    echo "Stage 3: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        train_e2e.py \
        --train_json ${feat_train_dir}/data.json \
        --valid_json ${feat_cv_dir}/data.json \
        --dict ${dict} \
        --LFR_m ${LFR_m} \
        --LFR_n ${LFR_n} \
        --lsm_weight $lsm_weight \
        --sampling_probability $sampling_probability \
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
        --visdom_id "$visdom_id" \
        --mode $mode \
        --trun $trun \
        --offset $offset
fi

if [ ${stage} -le 4 ]; then
    echo "Stage 4: Decoding"
  for data in 8000 ios nos; do
    decode_dir=${expdir}/decode_test_beam${beam_size}_nbest${nbest}_ml${decode_max_len}_test_set${data}_cweight${ctc_weight}
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

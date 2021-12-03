#!/bin/bash
   
conda activate negationminpairs
export PYTHONPATH=$PYTHONPATH:../code

lang=de
bert_model=bert-base-multilingual-cased
model_dir=../trained_models
model_name=bert-base-multilingual-cased#mnli_bs8_gradacc4_lr2e-5_maxlen128split_warmup0_epochs3_run1/best_model
bs=8
max_len=128
seed=1

outdir=results

# --test_data takes a list of files containing test data
python ../code/training/predict_nli.py \
       --test_data minimal_pairs_All_xnli_${lang}.test.jsonl,xnli_${lang}.test.jsonl \
       --data_path ../data/minimal_pairs/eval_splits \
       --model_name ${model_name} \
       --model_dir ${model_dir} \
       --config ./training/config.cfg \
       --seed ${seed}\
       --tokenizer ${bert_model} \
       --outdir ${outdir}\
       --bs ${bs}\
       --max_seq_len ${max_len}

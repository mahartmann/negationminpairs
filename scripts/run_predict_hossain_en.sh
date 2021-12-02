#!/bin/bash
   
conda activate negationminpairs
export PYTHONPATH=$PYTHONPATH:../code

 
bert_model=bert-base-multilingual-cased
model_dir=../trained_models
model_name=bert-base-multilingual-cased#mnlifiltered_bs8_gradacc4_lr2e-5_maxlen128split_warmup0_epochs3_run1/best_model
bs=8
max_len=128
outdir=../results
seed=1

python ../code/training/predict_nli.py \
       --test_data xnli_en.test.jsonl,hossain_minimal_pairs_All_mnli_en.test.jsonl,hossain_minimal_pairs_Orig_mnli_en.test.jsonl,hossain_minimal_pairs_Mod_mnli_en.test.jsonl,xnli_en.neg.test.jsonl,xnli_en.pos.test.jsonl \
       --data_path ../data/minimal_pairs/eval_splits \
       --model_name ${model_name} \
       --model_dir ${model_dir} \
       --config ./training/config.cfg \
       --seed ${seed}\
       --tokenizer ${bert_model} \
       --outdir ${outdir}\
       --bs ${bs}\
       --max_seq_len ${max_len}
      


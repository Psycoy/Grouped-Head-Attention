#!/bin/bash

#  bash ./SweepRuns/NUMBER/Debug.sh

pwd

mkdir "./Experimental_Results/debug"
echo "New debug entry created."

DataPath="iwslt14.tokenized.de-en"               # TODO: Change dataset path accordingly
CUDA_VISIBLE_DEVICES=0                           # TODO: Change gpu id accordingly
srun_command=(srun -p RTXA6Kq -w node12 -n 1)     # TODO: Change partition and node accordingly

EFFICIENT_MULTIHEAD_ATTENTION=true 
USE_EFFICIENT_EN_DE_ATTN=false
RESIDUAL_WHA=true
OPTIMIZER='adam' 
CLIP_NORM=0.0 
LR=5e-4 
LR_SCHEDULER='inverse_sqrt' 
WARMUP_UPDATES=4000 
DROPOUT=0.3 
WEIGHT_DECAY=0.0001
CRITERION='label_smoothed_cross_entropy_headclustering' 
LABEL_SMOOTHING=0.1 
_use_attentionmatrix_regularization=true 
_use_subspace_regularization=false
_use_headoutput_regularization=false
_use_attentionmatrix_regularization_HWA=false 
_use_subspace_regularization_HWA=false
_use_headoutput_regularization_HWA=false
_coefficient_attentionmatrix=1.0
_coefficient_subspace=1.0
_coefficient_headoutput=1.0
_coefficient_attentionmatrix_HWA=1.0
_coefficient_subspace_HWA=1.0
_coefficient_headoutput_HWA=1.0
_en_de_attn_regularization=false
_headout_regularization_afterHWA=false

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${srun_command[@]} fairseq-train \
    ../data-bin/${DataPath} \
    --user_dir ./ --arch efficient_transformer\
    --debug_mode true\
    \
    \
    --efficient_multihead_attention ${EFFICIENT_MULTIHEAD_ATTENTION}\
    --use_efficient_en_de_attn ${USE_EFFICIENT_EN_DE_ATTN}\
    --optimizer ${OPTIMIZER} --adam_betas '(0.9, 0.98)' --clip_norm ${CLIP_NORM} \
    --lr ${LR} --lr_scheduler ${LR_SCHEDULER} --warmup_updates ${WARMUP_UPDATES} \
    --dropout ${DROPOUT} --weight_decay ${WEIGHT_DECAY} \
    --criterion ${CRITERION} --label_smoothing ${LABEL_SMOOTHING} \
    \
    \
    --distributed_world_size 1\
    --share_decoder_input_output_embed true\
    --fixed_validation_seed 1\
    --tensorboard_logdir "./Experimental_Results/debug/tensorboard_logdir"\
    --log_file "./Experimental_Results/debug/logfile"\
    --max_tokens 4096 \
    --eval_bleu true\
    --eval_bleu_args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval_bleu_detok moses \
    --eval_bleu_remove_bpe\
    --eval_bleu_print_samples true\
    --save_dir "./Experimental_Results/debug/checkpoints"\
    --best_checkpoint_metric bleu --maximize_best_checkpoint_metric true\
    --patience 5\
    --no_epoch_checkpoints true

rm -rf ./Experimental_Results/debug
echo "Debug entry deleted."

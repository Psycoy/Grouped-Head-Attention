#!/bin/bash

name=SweepRun_61_ClusterHead_iwslt14_use_inclassloss_attentionmatrix_bothinterandinclass
_Train_Name=0


srun -p PV1003q -w node15 python3 ../fairseq/fairseq_cli/interactive_visualize.py ../data-bin/iwslt14.tokenized.de-en\
    --input utils/interactive_text.txt \
    --buffer_size 2\
    --tokenizer moses\
    --bpe subword_nmt\
    --bpe_codes ../translation/iwslt14.tokenized.de-en/code\
    --source_lang de --target_lang en \
    --user_dir ./\
    --distributed_world_size 1\
    --path "./Experimental_Results/${name}/${_Train_Name}/checkpoints_headcolab/checkpoint_best.pt" \
    --results_path "./Experimental_Results/${name}/${_Train_Name}"\
    --batch_size 1 --beam 5 --remove_bpe\
    --cpu true

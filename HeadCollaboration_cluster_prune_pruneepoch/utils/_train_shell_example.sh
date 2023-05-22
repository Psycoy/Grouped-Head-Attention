### IWSLT'14 German to English (Transformer)

# The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

# Next we'll train a Transformer translation model over this data:
# ```bash

# TODO 多卡训练

 # The default seed for preprocess, train, validation, and test is 1

# traintag describes what are we changing in this experiemnt
specifictag="tuning-----share-kbias-across-encoder-decoder-----true"
# generaltag describes what are we changing in this experiemnt
generaltag="efficienttransformer_original_transformersettings_for_iwslt14_kbiastrainable"

_Train_Name="(${specifictag})_____(${generaltag})"

source ./Machine_shell.sh #TODO 这里要换成具体的partition而不是node，这样不用自己一个一个去找， 比如_Machine_Used_Slurm_Command=(srun -p K80q --gres=gpu:1)

if [ -d "./Experimental_Results/${_Train_Name}" ]
then
    echo "This trainname already exists, please check and specify a new name according to your current setting."
    exit 1
else
    mkdir "./Experimental_Results/${_Train_Name}"
    echo "New experimental result entry created."
fi

# TODO 注意每次做实验前检查：
    # --efficient-multihead-attention
    # --bias-mode-k 'bias_trainable' 'bias_allone'
    # --share-kbias-across-layers
    # --share-kbias-across-encoder-decoder
# TODO 等参数与trainname符不符合

"${_Machine_Used_Slurm_Command[@]}" fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --user-dir ./ --arch efficient_transformer\
    \
    --efficient-multihead-attention\
    --bias-mode-k 'bias_trainable'\
    --share-kbias-across-encoder-decoder\
    \
    \
    --share-decoder-input-output-embed \
    --fixed-validation-seed 1\
    --tensorboard-logdir "./Experimental_Results/tensorboardruns_efficienttransformer/${_Train_Name}"\
    --log-file "./Experimental_Results/${_Train_Name}/logfiles_efficienttransformer"\
    --wandb-project\
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --save-dir "./Experimental_Results/${_Train_Name}/checkpoints_efficienttransformer"\
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric\
    --no-epoch-checkpoints\
    --patience 9\

echo " "
echo "------------------------------------------------------------------------------"
echo "Start evaluation..."
echo " "
"${_Machine_Used_Slurm_Command[@]}" fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --user-dir ./ --arch efficient_transformer\
    --path "./Experimental_Results/${_Train_Name}/checkpoints_efficienttransformer/checkpoint_best.pt" \
    --results-path "./Experimental_Results/${_Train_Name}"\
    --batch-size 128 --beam 5 --remove-bpe\
    --quiet
echo " "
echo "End of evaluation."
echo "------------------------------------------------------------------------------"
echo " "

echo " "
echo "------------------------------------------------------------------------------"
echo "Git updating changes..."
echo " "
git add "./Experimental_Results/${_Train_Name}/tensorboardruns_efficienttransformer"
git add "./Experimental_Results/${_Train_Name}/logfiles_efficienttransformer"
git commit -m "Added the train results of: ${_Train_Name}."
git add -u
git commit -m ${_Train_Name}
git push origin main
echo " "
echo "------------------------------------------------------------------------------"
echo " "
# ```
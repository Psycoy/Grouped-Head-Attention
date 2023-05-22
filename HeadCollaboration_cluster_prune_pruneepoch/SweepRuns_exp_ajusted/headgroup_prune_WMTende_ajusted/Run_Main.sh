#!/bin/bash

#####################################################################################
# TODO: Area of change for dfferent sweeps
# todo: Hyperparameters to tune. (one-to-one correspons with the parameters in sweep.yaml; change the hyperparameters you want to tune in sweep.yaml)
#----------------------------------------------------------------------------------------------------------------------#
ARGS=($@)

DataPath="wmt14_en_de_devnews2013_a"  # TODO: Change dataset path accordingly
CUDA_VISIBLE_DEVICES=0             # TODO: Change dataset path accordingly
project_NAME='headcolab'            # TODO: Change dataset path accordingly, need to change the same variable on the top of cleanup_results_sweep.py
resumedir=none          # TODO set to none if you don't want to resume from a history run.

#-----------------------------------------------------------#
# EFFICIENT_MULTIHEAD_ATTENTION
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--EFFICIENT_MULTIHEAD_ATTENTION" ]
    then
        EFFICIENT_MULTIHEAD_ATTENTION=${ARGARRAY[1]}
        echo "EFFICIENT_MULTIHEAD_ATTENTION: ${EFFICIENT_MULTIHEAD_ATTENTION}"
        break
    fi
done

#-----------------------------------------------------------#
# _voting_to_prune
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_voting_to_prune" ]
    then
        _voting_to_prune=${ARGARRAY[1]}
        echo "_voting_to_prune: ${_voting_to_prune}"
        break
    fi
done

#-----------------------------------------------------------#
# _debug_mode
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_debug_mode" ]
    then
        _debug_mode=${ARGARRAY[1]}
        echo "_debug_mode: ${_debug_mode}"
        break
    fi
done

#-----------------------------------------------------------#
# _Supervise_mode
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_Supervise_mode" ]
    then
        _Supervise_mode=${ARGARRAY[1]}
        echo "_Supervise_mode: ${_Supervise_mode}"
        break
    fi
done

#-----------------------------------------------------------#
# _cluster_matrix
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_cluster_matrix" ]
    then
        _cluster_matrix=${ARGARRAY[1]}
        echo "_cluster_matrix: ${_cluster_matrix}"
        break
    fi
done

#-----------------------------------------------------------#
# _supervised_matrix
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_supervised_matrix" ]
    then
        _supervised_matrix=${ARGARRAY[1]}
        echo "_supervised_matrix: ${_supervised_matrix}"
        break
    fi
done

#-----------------------------------------------------------#
# _cluster_loss_coefficient_inclass
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_cluster_loss_coefficient_inclass" ]
    then
        _cluster_loss_coefficient_inclass=${ARGARRAY[1]}
        echo "_cluster_loss_coefficient_inclass: ${_cluster_loss_coefficient_inclass}"
        break
    fi
done

#-----------------------------------------------------------#
# _cluster_loss_coefficient_interclass
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_cluster_loss_coefficient_interclass" ]
    then
        _cluster_loss_coefficient_interclass=${ARGARRAY[1]}
        echo "_cluster_loss_coefficient_interclass: ${_cluster_loss_coefficient_interclass}"
        break
    fi
done

#-----------------------------------------------------------#
# _N_head_clusters
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_N_head_clusters" ]
    then
        _N_head_clusters=${ARGARRAY[1]}
        echo "_N_head_clusters: ${_N_head_clusters}"
        break
    fi
done

#-----------------------------------------------------------#
# _use_interclass_loss
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_use_interclass_loss" ]
    then
        _use_interclass_loss=${ARGARRAY[1]}
        echo "_use_interclass_loss: ${_use_interclass_loss}"
        break
    fi
done

#-----------------------------------------------------------#
# _use_inclass_loss
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_use_inclass_loss" ]
    then
        _use_inclass_loss=${ARGARRAY[1]}
        echo "_use_inclass_loss: ${_use_inclass_loss}"
        break
    fi
done

#-----------------------------------------------------------#
# _use_efficient_en_de_attn
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_use_efficient_en_de_attn" ]
    then
        _use_efficient_en_de_attn=${ARGARRAY[1]}
        echo "_use_efficient_en_de_attn: ${_use_efficient_en_de_attn}"
        break
    fi
done

#-----------------------------------------------------------#
# _kmeans_distance_metric
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_kmeans_distance_metric" ]
    then
        _kmeans_distance_metric=${ARGARRAY[1]}
        echo "_kmeans_distance_metric: ${_kmeans_distance_metric}"
        break
    fi
done

#-----------------------------------------------------------#
# _epoch_start_to_prune
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_epoch_start_to_prune" ]
    then
        _epoch_start_to_prune=${ARGARRAY[1]}
        echo "_epoch_start_to_prune: ${_epoch_start_to_prune}"
        break
    fi
done

#-----------------------------------------------------------#
# _need_prune
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_need_prune" ]
    then
        _need_prune=${ARGARRAY[1]}
        echo "_need_prune: ${_need_prune}"
        break
    fi
done

#-----------------------------------------------------------#
# _keep_updating_cluster
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--_keep_updating_cluster" ]
    then
        _keep_updating_cluster=${ARGARRAY[1]}
        echo "_keep_updating_cluster: ${_keep_updating_cluster}"
        break
    fi
done

#----------------------------------------------------------------------------------------------------------------------#
# Training HPs
#-----------------------------------------------------------#
# OPTIMIZER
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--OPTIMIZER" ]
    then
        OPTIMIZER=${ARGARRAY[1]}
        echo "OPTIMIZER: ${OPTIMIZER}"
        break
    fi
done
#-----------------------------------------------------------#
# CLIP_NORM
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--CLIP_NORM" ]
    then
        CLIP_NORM=${ARGARRAY[1]}
        echo "CLIP_NORM: ${CLIP_NORM}"
        break
    fi
done
#-----------------------------------------------------------#
# LR
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--LR" ]
    then
        LR=${ARGARRAY[1]}
        echo "LR: ${LR}"
        break
    fi
done
#-----------------------------------------------------------#
# LR_SCHEDULER
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--LR_SCHEDULER" ]
    then
        LR_SCHEDULER=${ARGARRAY[1]}
        echo "LR_SCHEDULER: ${LR_SCHEDULER}"
        break
    fi
done
#-----------------------------------------------------------#
# WARMUP_UPDATES
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--WARMUP_UPDATES" ]
    then
        WARMUP_UPDATES=${ARGARRAY[1]}
        echo "WARMUP_UPDATES: ${WARMUP_UPDATES}"
        break
    fi
done
#-----------------------------------------------------------#
# DROPOUT
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--DROPOUT" ]
    then
        DROPOUT=${ARGARRAY[1]}
        echo "DROPOUT: ${DROPOUT}"
        break
    fi
done
#-----------------------------------------------------------#
# WEIGHT_DECAY
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--WEIGHT_DECAY" ]
    then
        WEIGHT_DECAY=${ARGARRAY[1]}
        echo "WEIGHT_DECAY: ${WEIGHT_DECAY}"
        break
    fi
done
#-----------------------------------------------------------#
# CRITERION
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--CRITERION" ]
    then
        CRITERION=${ARGARRAY[1]}
        echo "CRITERION: ${CRITERION}"
        break
    fi
done
#-----------------------------------------------------------#
# LABEL_SMOOTHING
for ARG in ${ARGS[@]}
do
    IFS='=' read -r -a ARGARRAY <<< $ARG
    if [ ${ARGARRAY[0]} = "--LABEL_SMOOTHING" ]
    then
        LABEL_SMOOTHING=${ARGARRAY[1]}
        echo "LABEL_SMOOTHING: ${LABEL_SMOOTHING}"
        break
    fi
done

#----------------------------------------------------------------------------------------------------------------------#
# Basic Arch HPs



#####################################################################################
FOLDERPATH=$(dirname ${BASH_SOURCE})
sweep_yaml="$FOLDERPATH/sweep.yaml"
source utils/yaml_parser.sh
eval $(parse_yaml $sweep_yaml)

source $FOLDERPATH/SweepID.sh
#####################################################################################
# Run Info

 # The default seed for preprocess, train, validation, and test is 1

_Train_Name=`cat "./Experimental_Results/${name}/_Train_Counter.txt"`

if [ -d "./Experimental_Results/${name}/_SweepDone.txt" ]
then
    echo "You are running another Run_Main.sh that is not settled for your current sweep (current running script is under $FOLDERPATH). Please modify the foldername behind 'SweepRuns/' in 'program' and 'command' of your intended sweep file."
    exit 1
fi

if [ -d "./Experimental_Results/${name}/${_Train_Name}" ]
then
    _Train_Name="$((${_Train_Name}+1))"
    echo ${_Train_Name} > "./Experimental_Results/${name}/_Train_Counter.txt"
    if [ -d "./Experimental_Results/${name}/${_Train_Name}" ]
    then
        echo "This sweep result folder is not a new one, please check and specify ***a new yaml config*** according to your current setting."
        # TODO change this line if you want to resume
    else
        mkdir "./Experimental_Results/${name}/${_Train_Name}"
        echo "New run result entry created."
    fi
else
    mkdir "./Experimental_Results/${name}/${_Train_Name}"
    echo "New run result entry created."
fi

if [ ${resumedir} != none ]; then   # TODO change this line if you want to resume
    _Train_Name=${resumedir}
    echo "Resuming from "./Experimental_Results/${name}/${_Train_Name}" ..."
fi

_Train_Name_c=${_Train_Name//[^[:alnum:]]/}
_Wandb_ID="${_Train_Name_c}${RANDOM}${RANDOM}${RANDOM}"

#####################################################################################

echo " "
echo "------------------------------------------------------------------------------"
echo " "
echo "Train name: ${_Train_Name}"
echo "Run id: ${_Wandb_ID}"
echo " "
echo "------------------------------------------------------------------------------"
echo " "

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-train \
    ../data-bin/${DataPath} \
    --user_dir ./ --arch efficient_transformer\
    \
    \
    --efficient_multihead_attention ${EFFICIENT_MULTIHEAD_ATTENTION}\
    --optimizer ${OPTIMIZER} --adam_betas '(0.9, 0.98)' --clip_norm ${CLIP_NORM} \
    --lr ${LR} --lr_scheduler ${LR_SCHEDULER} --warmup_updates ${WARMUP_UPDATES} \
    --dropout ${DROPOUT} --weight_decay ${WEIGHT_DECAY} \
    --criterion ${CRITERION} --label_smoothing ${LABEL_SMOOTHING} \
    --update_freq 8\
    --Supervise_mode ${_Supervise_mode}\
    --cluster_matrix ${_cluster_matrix}\
    --supervised_matrix ${_supervised_matrix}\
    --cluster_loss_coefficient_interclass ${_cluster_loss_coefficient_interclass}\
    --cluster_loss_coefficient_inclass ${_cluster_loss_coefficient_inclass}\
    --N_head_clusters ${_N_head_clusters}\
    --debug_mode ${_debug_mode}\
    --use_interclass_loss ${_use_interclass_loss}\
    --use_inclass_loss ${_use_inclass_loss}\
    --use_efficient_en_de_attn ${_use_efficient_en_de_attn}\
    --kmeans_distance_metric ${_kmeans_distance_metric}\
    --epoch_start_to_prune ${_epoch_start_to_prune}\
    --need_prune ${_need_prune}\
    --keep_updating_cluster ${_keep_updating_cluster}\
    --voting_to_prune ${_voting_to_prune}\
    \
    \
    --distributed_world_size 1\
    --share_decoder_input_output_embed true\
    --fixed_validation_seed 1\
    --tensorboard_logdir "./Experimental_Results/tensorboardruns_${project_NAME}/${name}/${_Train_Name}"\
    --log_file "./Experimental_Results/${name}/${_Train_Name}/logfiles_${project_NAME}"\
    --wandb_project HeadCollaboration\
    --wandb_runname "${_Wandb_ID}"\
    --wandb_runid "${_Wandb_ID}"\
    --wandb_runtags "${username_projectname_sweepid}"\
    --wandb_runnotes "${username_projectname_sweepid}"\
    --max_tokens 4096 \
    --eval_bleu true\
    --eval_bleu_args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval_bleu_detok moses \
    --eval_bleu_remove_bpe\
    --eval_bleu_print_samples true\
    --save_dir "./Experimental_Results/${name}/${_Train_Name}/checkpoints_${project_NAME}"\
    --best_checkpoint_metric bleu --maximize_best_checkpoint_metric true\
    --patience 12\
    --keep_best_checkpoints 5\
    --no_epoch_checkpoints true\


echo " "
echo "------------------------------------------------------------------------------"
echo "Start evaluation..."
echo " "

echo "Averaging over the last five checkpoints..."
python ./utils/average_checkpoints.py \
--inputs ./Experimental_Results/${name}/${_Train_Name}/checkpoints_${project_NAME} \
--num-update-checkpoints 5 --output ./Experimental_Results/${name}/${_Train_Name}/checkpoints_${project_NAME}/averaged_model.pt

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} fairseq-generate ../data-bin/${DataPath} \
    --user_dir ./ --arch efficient_transformer\
    --distributed_world_size 1\
    --wandb_project HeadCollaboration\
    --wandb_runid "${_Wandb_ID}"\
    --path "./Experimental_Results/${name}/${_Train_Name}/checkpoints_${project_NAME}/averaged_model.pt" \
    --results_path "./Experimental_Results/${name}/${_Train_Name}"\
    --batch_size 128 --beam 4 --remove_bpe\
    --lenpen 0.6\
    --quiet true

echo " "
echo "End of evaluation."
echo "------------------------------------------------------------------------------"
echo " "



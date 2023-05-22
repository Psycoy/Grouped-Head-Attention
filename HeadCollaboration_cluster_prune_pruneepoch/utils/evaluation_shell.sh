#!/bin/bash

#SBATCH -o ./sbatchlogs/gpu-job-%j.output
#SBATCH -p RTXA6Kq -w node08
#SBATCH -n 1


# echo "------------------------------------------------------------------------------"
# echo "Start evaluation..."
# echo " "
# project_NAME=headcolab
# _Sweep_Name=SweepRun_2_ClusterHead_iwslt14_use_inclassloss_v_bothinterandinclass_usevariantencdecattention_5checkpointsaveraged_compoundsplit_prunefromepoch10_noearlystop_rerun5
# _Train_id=1
# DataPath="iwslt14.tokenized.de-en.compoundsplit"  # TODO: Change dataset path accordingly
# CUDA_VISIBLE_DEVICES=0             # TODO: Change dataset path accordingly
# average_checkpoints=true

# # _evaluation_HP_to_tune="len_pen"
# # HPs=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)

# _evaluation_HP_to_tune="beamsize"
# HPs=(5)
# lenpen=0.6

# if [ ${average_checkpoints} == 'true' ]
# then
# echo "Averaging over the last five checkpoints..."
# python ./utils/average_checkpoints.py \
# --inputs ./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME} \
# --num-update-checkpoints 5 --output ./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/averaged_model.pt

# for HP in ${HPs[@]}; do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun fairseq-generate ../data-bin/${DataPath} \
#         --user_dir ./ --arch efficient_transformer\
#         --debug_mode true\
#         --path "./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/averaged_model.pt" \
#         --results_path "./Experimental_Results/evaluations/${_Sweep_Name}/${_Train_id}/${_evaluation_HP_to_tune}: ${HP}"\
#         --distributed_world_size 1\
#         --batch_size 128 --beam ${HP} --remove_bpe\
#         --quiet true\
#         # --lenpen ${lenpen}\
#         # TODO modify the configs is important
# done
# else
# for HP in ${HPs[@]}; do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun fairseq-generate ../data-bin/${DataPath} \
#         --user_dir ./ --arch efficient_transformer\
#         --debug_mode true\
#         --path "./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/checkpoint_best.pt" \
#         --results_path "./Experimental_Results/evaluations/${_Sweep_Name}/${_Train_id}/${_evaluation_HP_to_tune}: ${HP}"\
#         --distributed_world_size 1\
#         --batch_size 128 --beam ${HP} --remove_bpe\
#         --quiet true\
#         # --lenpen ${lenpen}\
# done
# fi
# echo " "
# echo "------------------------------------------------------------------------------"








# echo "------------------------------------------------------------------------------"
# echo "Start evaluation..."
# echo " "
# _Sweep_Name=SweepRun_62copy_ClusterHead_iwslt14_use_inclassloss_v_bothinterandinclass_usevariantencdecattention
# _Train_id=8
# DataPath="iwslt14.tokenized.de-en"  # TODO: Change dataset path accordingly
# CUDA_VISIBLE_DEVICES=0             # TODO: Change dataset path accordingly

# _evaluation_HP_to_tune="len_pen"
# HPs=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)

# # _evaluation_HP_to_tune="beamsize"
# # HPs=(1 2 3 4 5 6 7 8 9)

# for HP in ${HPs[@]}; do
#     CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun fairseq-generate ../data-bin/${DataPath} \
#         --user_dir ./ --arch efficient_transformer\
#         --debug_mode true\
#         --path "./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_headcolab/checkpoint_best.pt" \
#         --results_path "./Experimental_Results/evaluations/${_Sweep_Name}/${_Train_id}/${_evaluation_HP_to_tune}: ${HP}"\
#         --distributed_world_size 1\
#         --batch_size 128 --beam 5 --remove_bpe\
#         --lenpen ${HP}\
#         --quiet true
# done
# echo " "
# echo "------------------------------------------------------------------------------"

echo "------------------------------------------------------------------------------"
echo "Start evaluation..."
echo " "
project_NAME=headcolab
_Sweep_Name=SweepRuns_exp_ajusted_headgroup_prune_WMTenfr_ajusted_Headclustering_prune_use_bothin_and_interclass_loss_useenc_decattention_patience12
_Train_id=0
DataPath="wmt14_en_fr_standardsplit_compoundsplit"  # TODO: Change dataset path accordingly
CUDA_VISIBLE_DEVICES=7             # TODO: Change dataset path accordingly
average_checkpoints=true

# _evaluation_HP_to_tune="len_pen"
# HPs=(0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0)

_evaluation_HP_to_tune="beamsize"
HPs=(4)
lenpen=0.6

if [ ${average_checkpoints} == 'true' ]
then
echo "Averaging over the last n checkpoints..."
python ./utils/average_checkpoints.py \
--inputs ./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME} \
--num-update-checkpoints 5 --output ./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/averaged_model.pt

for HP in ${HPs[@]}
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun fairseq-generate ../data-bin/${DataPath} \
        --user_dir ./ --arch efficient_transformer\
        --debug_mode true\
        --path "./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/averaged_model.pt" \
        --results_path "./Experimental_Results/evaluations/${_Sweep_Name}/${_Train_id}/${_evaluation_HP_to_tune}: ${HP}"\
        --distributed_world_size 1\
        --batch_size 640 --beam ${HP} --remove_bpe\
        --quiet true\
        --lenpen ${lenpen}\
        # TODO modify the configs is important
done
else
for HP in ${HPs[@]}
do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} srun fairseq-generate ../data-bin/${DataPath} \
        --user_dir ./ --arch efficient_transformer\
        --debug_mode true\
        --path "./Experimental_Results/${_Sweep_Name}/${_Train_id}/checkpoints_${project_NAME}/checkpoint_best.pt" \
        --results_path "./Experimental_Results/evaluations/${_Sweep_Name}/${_Train_id}/${_evaluation_HP_to_tune}: ${HP}"\
        --distributed_world_size 1\
        --batch_size 640 --beam ${HP} --remove_bpe\
        --lenpen ${lenpen}
done
fi

echo " "
echo "------------------------------------------------------------------------------"

wait
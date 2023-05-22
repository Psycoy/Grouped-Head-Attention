#!/bin/bash
#SBATCH -o ./sbatchlogs/gpu-job-%j.output
#SBATCH -p PA40q -w node07                # TODO change the node if not available
#SBATCH -n 1


module load cuda11.1/toolkit/11.1.1
# module load slurm/17.11.5

PATH_OF_THIS_SBATCH_SCRIPT=$(awk -F'/Sbatch.sh' '{print $1}' <<< "$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')")  
sweep_yaml="${PATH_OF_THIS_SBATCH_SCRIPT}/sweep.yaml"

source ${PATH_OF_THIS_SBATCH_SCRIPT}/SweepID.sh

source utils/yaml_parser.sh
eval $(parse_yaml $sweep_yaml)

echo "Sweep name: ${name}"
echo "Sweep description: ${description}"

if [ -d "./Experimental_Results/${name}" ]
then
    echo "This sweepname: ./Experimental_Results/${name} already exists, please check and specify ***a new yaml config*** according to your current setting."
    exit 1
else
    mkdir "./Experimental_Results/${name}"
    echo "New sweep entry created."
    echo "0" > "./Experimental_Results/${name}/_Train_Counter.txt"
fi


srun wandb agent "${username_projectname_sweepid}"

python3 utils/cleanup_results_sweep.py --SweepFolderName "${name}" --k 3
# clean up the sub-optimum saved models to save disk space, only keep the top k ones. (Scores of runs not evaluated will be treated as 0.)

rm "./Experimental_Results/${name}/_Train_Counter.txt"
echo "This Sweep is done!" > "./Experimental_Results/${name}/_SweepDone.txt"

wait # ensure that everything is done before killing the session

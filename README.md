# ACL-2023-Grouped-Head-Attention


Code for the paper Finding the Pillars of Strength for Multi-Head Attention, Jinjie Ni, Rui Mao, Zonglin Yang, Han Lei, and Erik Cambria.

## Requirements
torch==1.9.0+cu111<br>
python==3.8.5<br>
wandb==0.12.9 

## Install
cd fairseq<br>
pip install -e .


## General Usage
Our code uses W&B Sweep pipeline, and the running pipeline is integrated with Slurm Workload Manager.

To reproduce the results for GHT, please change your current working directory to `HeadCollaboration`; to reproduce the results for GHT-PS, please change your current working directory to `HeadCollaboration_cluster_prune_pruneepoch`.

### On Slurm Clusters

1. Download & Process the data: `bash data_process/data_preparation.sh`.
2. `wandb sweep #TheSweepYamlPath`.
3. Copy & Paste the generated sweep id to the corresponding section of `SweepID.sh`.
4. Configure the hyperparameters in `sweep.yaml` according to the specifications in Appendix A.
5. Configure your partition and node in `Sbatch.sh`; configure the data, gpu id, and W&B project name in Run_Main.sh, run `sbatch #path_to_Sbatch.sh`.

Notes:
1. The above-mentioned files are under the `SweepRuns_exp_ajusted` folder of `HeadCollaboration` and `HeadCollaboration_cluster_prune_pruneepoch` respectively.
2. For users not using slurm clusters, you need to run `wandb agent #username_projectname_sweepid` (`#username_projectname_sweepid` is the variable in `SweepID.sh`).
3. If you want to run on multiple GPUs (our experiments are mostly based on a single A100-80GB GPU), you need to configure the fairseq-train command in `Run_Main.sh` according to https://fairseq.readthedocs.io/en/latest/.
4. More Sweep and W&B usage can be found at https://docs.wandb.ai/guide.

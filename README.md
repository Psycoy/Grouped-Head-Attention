# ACL'23 Grouped Head Attention


Code for the paper [<b>Finding the Pillars of Strength for Multi-Head Attention</b>](https://arxiv.org/abs/2305.14380), Jinjie Ni, Rui Mao, Zonglin Yang, Han Lei, and Erik Cambria.

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



If you use this code in your work then please cite the paper [<b>Finding the Pillars of Strength for Multi-Head Attention</b>](https://arxiv.org/abs/2305.14380) with the following:

```
@article{DBLP:journals/corr/abs-2305-14380,
  author       = {Jinjie Ni and
                  Rui Mao and
                  Zonglin Yang and
                  Han Lei and
                  Erik Cambria},
  title        = {Finding the Pillars of Strength for Multi-Head Attention},
  journal      = {CoRR},
  volume       = {abs/2305.14380},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.14380},
  doi          = {10.48550/arXiv.2305.14380},
  eprinttype    = {arXiv},
  eprint       = {2305.14380},
  timestamp    = {Mon, 26 Jun 2023 20:50:08 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-14380.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

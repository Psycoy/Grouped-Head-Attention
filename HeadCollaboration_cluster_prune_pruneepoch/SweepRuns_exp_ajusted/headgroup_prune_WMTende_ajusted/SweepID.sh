#!/bin/bash

#####################################################################################
# TODO: Area of change for dfferent sweeps
# TODO: Configure the sweep.yaml accordingly (Always change the foldername in sweep.yaml !!!!!!!!)
# wandb sweep ./SweepRuns_exp_ajusted/headgroup_prune_WMTende_ajusted/sweep.yaml

# sbatch ./SweepRuns_exp_ajusted/headgroup_prune_WMTende_ajusted/Sbatch.sh

Sweepid=
Projectname=
username=
username_projectname_sweepid="${username}/${Projectname}/${Sweepid}"

#####################################################################################
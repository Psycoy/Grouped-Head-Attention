#!/bin/bash

#####################################################################################
# TODO: Area of change for dfferent sweeps
# TODO: Configure the sweep.yaml accordingly (Always change the foldername in sweep.yaml !!!!!!!!)
# wandb sweep ./SweepRuns_exp_ajusted/headgroup_iwslt_ajusted_v_deen/sweep.yaml

# sbatch ./SweepRuns_exp_ajusted/headgroup_iwslt_ajusted_v_deen/Sbatch.sh

Sweepid=
Projectname=
username=
username_projectname_sweepid="${username}/${Projectname}/${Sweepid}"

#####################################################################################
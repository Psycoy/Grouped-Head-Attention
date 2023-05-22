#!/bin/bash

# Machine Lists - specify node list


# See machine status
echo "Node: -p NV100q -w node07"
srun -p NV100q -w node07 nvidia-htop.py -l
echo "Node: -p NV100q -w node08"
srun -p NV100q -w node08 nvidia-htop.py -l
echo "Node: -p NV100q -w node23"
srun -p NV100q -w node23 nvidia-htop.py -l
echo "Node: -p NV100q -w node24"
srun -p NV100q -w node24 nvidia-htop.py -l
echo "Node: -p PV100q -w node09"
srun -p PV100q -w node09 nvidia-htop.py -l
echo "Node: -p K80q -w node05"
srun -p K80q -w node05 nvidia-htop.py -l
echo "Node: -p K80q -w node06"
srun -p K80q -w node06 nvidia-htop.py -l
echo "Node: -p DGXq -w node18"
srun -p DGXq -w node18 nvidia-htop.py -l
echo "Node: -p DGXq -w node19"
srun -p DGXq -w node19 nvidia-htop.py -l
echo "Node: -p DGXq -w node20"
srun -p DGXq -w node20 nvidia-htop.py -l
echo "Node: -p DGXq -w node21"
srun -p DGXq -w node21 nvidia-htop.py -l
echo "Node: -p PV1003q -w node14"
srun -p PV1003q -w node14 nvidia-htop.py -l
echo "Node: -p PV1003q -w node15"
srun -p PV1003q -w node15 nvidia-htop.py -l
echo "Node: -p PV1003q -w node16"
srun -p PV1003q -w node16 nvidia-htop.py -l
echo "Node: -p PV1003q -w node17"
srun -p PV1003q -w node17 nvidia-htop.py -l
# todo temporarily unavailable
echo "Node: -p PA100q -w node25"
srun -p PA100q -w node25 nvidia-htop.py -l
echo "Node: -p PA100q -w node26"
srun -p PA100q -w node26 nvidia-htop.py -l
echo "Node: -p PA100q -w node29"
srun -p PA100q -w node29 nvidia-htop.py -l
echo "Node: -p PA100q -w node30"
srun -p PA100q -w node30 nvidia-htop.py -l
echo "Node: -p RTX8Kq -w node22"
srun -p RTX8Kq -w node22 nvidia-htop.py -l
echo "Node: -p PA40q -w node01"
srun -p PA40q -w node01 nvidia-htop.py -l
echo "Node: -p PA40q -w node04"
srun -p PA40q -w node04 nvidia-htop.py -l
echo "Node: -p PA40q -w node13"
srun -p PA40q -w node13 nvidia-htop.py -l
echo "Node: -p RTXA6Kq -w node10"
srun -p RTXA6Kq -w node10 nvidia-htop.py -l
echo "Node: -p RTXA6Kq -w node12"
srun -p RTXA6Kq -w node12 nvidia-htop.py -l
echo "Node: -p RTXA6Kq -w node27"
srun -p RTXA6Kq -w node27 nvidia-htop.py -l
echo "Node: -p RTXA6Kq -w node28"
srun -p RTXA6Kq -w node28 nvidia-htop.py -l

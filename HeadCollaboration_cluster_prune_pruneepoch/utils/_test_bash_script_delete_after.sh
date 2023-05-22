# specifictag="wandb project log file - delete after"
# generaltag="testrun2"

# _Train_Name="(${specifictag})_____(${generaltag})"
# _Train_Name_c=${_Train_Name//[^[:alnum:]]/}
# _Wandb_ID="${_Train_Name_c}${RANDOM}${RANDOM}${RANDOM}"
# echo ${_Wandb_ID}

# a="8"

# echo "$(($a + 1))"

# for directory_path in ./Experimental_Results/*
# do
#     echo "$directory_path"
# done

# python3 utils/cleanup_results_sweep.py --SweepFolderName "test_sweep_folder" --k 2

# source utils/yaml_parser.sh
# eval $(parse_yaml sweep_configs/sweep1.yaml)         # TODO Always check these configs before running a sweep
# echo $name


# a="$(grep -oP '(?<="username_projectname_sweepid": ")[^"]*' sweep_configs/sweep1_otherconfigs.json)"
# echo $a
# echo $a

# echo $1

# echo "1" > test.txt
# value=`cat "test.txt"`
# echo ${value}
# rm test.txt
# echo $(dirname ${BASH_SOURCE})

# IFS='=' read -r -a BIAS_MODE_K <<< "$1"
# BIAS_MODE_K=${BIAS_MODE_K[1]}
# if BIAS_MODE_K==""
# then
#     echo "Nan"
# else
#     echo "BIAS_MODE_K: ${BIAS_MODE_K}"
# fi
# ARGEND=3
# for i in $(seq 1 $ARGEND)
#     do 
#         IFS='=' read -r -a ARGS <<< ${${i}}
#         echo $ARGS
# done

# ARGS=($@)
# echo ${ARGS[@]}
# for ARG in ${ARGS[@]}
# do  
#     echo $ARG
#     IFS='=' read -r -a ARGARRAY <<< $ARG
#     if [ ${ARGARRAY[0]} = "--CLIP_NORM" ]
#     then
#         CLIP_NORM=${ARGARRAY[1]}
#         echo "CLIP_NORM: ${CLIP_NORM}"
#         break
#     fi
# done

# a=`cat "./_SweepDone.txt"`

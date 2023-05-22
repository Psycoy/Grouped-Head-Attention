#!/bin/bash

# test=(echo "test!")

# ${test[@]}

for i in "a" "b";do
    for j in "c" "d";do
        echo $i$j
    done
done
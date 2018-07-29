#!/bin/bash

unset outbase
unset experiment_basedir
unset trainlog_filename
unset data_dir
unset data_loader
unset SG_psi
unset LG_uniformPa
unset attributes_weight
unset LG_true_compl_x
unset batch_size


test_set=0
num_repeats=${num_repeats:-5} # default value is 5 repeatitions
max_epochs=200
patience=$max_epochs

dataset=CUB
modelname=KSoft
lr=3e-05
max_epochs=200
# beta=0.00001
lambda=0.00001
SG_gain=1
SG_num_K=30
SG_lr=1

## declare an array for beta
declare -a beta_arr=("0.00001" "0.001")

## now loop through the above array
for beta in "${beta_arr[@]}"
do

    echo "Training on $dataset with $modelname for $num_repeats repeatitions. is_test_set=$test_set"

    for (( rpt=0; rpt<$num_repeats; rpt+=1 ));
    do
        source scripts/train.sh
    done
done
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

test_set=1

dataset=CUB
modelname=ESZSL
ES_gamma=1000
ES_lambda=0.1

echo "Training on $dataset with $modelname"

# Execute script
source scripts/train.sh

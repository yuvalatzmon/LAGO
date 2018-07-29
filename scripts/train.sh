#!/bin/bash

# Choose the dev or test set
if [ "$test_set" -eq "0" ]; then
    set_name="dev"
else
    set_name="test"
fi

# Erase large files when done (model and predictions),
# keeping only a summary of results.
# 0:keep large files, 1:erase large files
erase_large_files_when_done=0

### Default values:
#   The syntax var2=${var2:-foo} will take on the value "foo"
#   if not overridden by caller script (or commandline)

## Default values for directories
outbase=${outbase:-"output"} # output base dir
experiment_basedir=${experiment_basedir:-"${outbase}/${set_name}/${dataset}/${modelname}"}
trainlog_filename=${trainlog_filename:-"${experiment_basedir}/pid${BASHPID}_train_log.txt"}
data_dir=${data_dir:-"data/${dataset}"} # output base dir
data_loader=${data_loader:-"${dataset}_Xian"} # output base dir

## Default values for commandline arguments
gpuid=${gpuid:-0}
batch_size=${batch_size:-64}
SG_psi=${SG_psi:-0}

# These are the 3 parameters described by the in
# Section 4.2:"Design decisions" and in Section 4.4 (Ablation Experiments)
LG_uniformPa=${LG_uniformPa:-1} # UNIFORM @Ablation
LG_true_compl_x=${LG_true_compl_x:-1} # CONST @Ablation
attributes_weight=${attributes_weight:-0} # IMPLICIT @Ablation
##########

case $modelname in
KSoft)
    model=LAGO
    model_variant=LAGO_KSoft
    SG_trainable=1
    LG_norm_groups_to_1=0
    ;;
SemanticSoft)
    model=LAGO
    model_variant=LAGO_SemanticSoft
    SG_trainable=1
    LG_norm_groups_to_1=1
    ;;
SemanticHard)
    model=LAGO
    model_variant=LAGO_SemanticSoft
    SG_trainable=0
    LG_norm_groups_to_1=1
    ;;
Singletons)
    model=LAGO
    model_variant=Singletons
    SG_trainable=0
    LG_norm_groups_to_1=0
    ;;
ESZSL)
    model=ESZSL

    # ESZSL gets the max accuracy when it uses L2 normalized class
    # description (as in Xian, CVPR 2017), rather than the probabilisti class
    # description (the case for LAGO). Therefore, we set
    # use_xian_normed_class_description to default 1 with ESZSL
    #
    # See more documentation in code comments
    #     (search for use_xian_normed_class_description)
    use_xian_normed_class_description=${use_xian_normed_class_description:-1} #
    ;;
*)
    echo "Usage: \$modelname {KSoft|SemanticSoft|SemanticHard|Singletons}"
    return 1
;;
esac

case $model in
LAGO)
    params_path="lr=$lr%mxE=$max_epochs%b=$beta%ld=$lambda%\
nrm1=$LG_norm_groups_to_1%uPA=$LG_uniformPa%aw=$attributes_weight%\
TCx=$LG_true_compl_x%SGk=$SG_num_K%SGt=$SG_trainable%SGg=$SG_gain%\
SGlr=$SG_lr%SGp=$SG_psi%bs=$batch_size%rpt=$rpt"

    outdir=$experiment_basedir/$params_path
    mkdir -p $outdir/
    cp -f scripts/gitignore_output_subdirs $experiment_basedir/.gitignore
    cp -f scripts/gitignore_output_subdirs $outdir/.gitignore
    echo "Saving training outcomes under $outdir"

    cmd="KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=$gpuid PYTHONPATH="./" \
    python zero_shot_src/ZStrain.py \
    --base_output_dir=$outbase \
    --train_dir=$outdir \
    --initial_learning_rate=$lr  \
    --max_epochs=$max_epochs \
    --use_trainval_set=$test_set \
    --LG_beta=$beta  \
    --LG_lambda=$lambda \
    --LG_norm_groups_to_1=$LG_norm_groups_to_1  \
    --LG_uniformPa=$LG_uniformPa  \
    --attributes_weight=$attributes_weight \
    --LG_true_compl_x=$LG_true_compl_x \
    --SG_num_K=$SG_num_K  \
    --SG_trainable=$SG_trainable  \
    --SG_gain=$SG_gain  \
    --SG_gamma_lr=$SG_lr \
    --SG_psi=$SG_psi  \
    --model_variant=$model_variant  \
    --repeat=$rpt  \
    --data_loader=$data_loader \
    --data_dir=$data_dir \
    --model_name=$model  \
    --batch_size=$batch_size  \
    --in_progress_timeout_minutes=2 "

;;
ESZSL)
    params_path="gm=$ES_gamma%ld=$ES_lambda%nrm_cls_desc=$use_xian_normed_class_description"

    outdir=$experiment_basedir/$params_path
    mkdir -p $outdir/
    cp -f scripts/gitignore_output_subdirs $experiment_basedir/.gitignore
    cp -f scripts/gitignore_output_subdirs $outdir/.gitignore
    echo "Saving training outcomes under $outdir"

    cmd="KERAS_BACKEND=tensorflow CUDA_VISIBLE_DEVICES=$gpuid PYTHONPATH="./" \
    python zero_shot_src/ZStrain.py \
    --base_output_dir=$outbase \
    --train_dir=$outdir \
    --use_trainval_set=$test_set \
    --ES_gamma=$ES_gamma \
    --ES_lambda=$ES_lambda \
    --use_xian_normed_class_description=$use_xian_normed_class_description \
    --data_loader=$data_loader \
    --data_dir=$data_dir \
    --model_name=$model  \
    --in_progress_timeout_minutes=2 "
    ;;
*)
    echo "Error, unknown model: $model"
    return 1
;;
esac

echo $cmd
eval $cmd  |& tee -a $trainlog_filename

if [ "$erase_large_files_when_done" -eq 1 ]; then
    echo "\$erase_large_files_when_done is set"
    echo "Deleting *.hdf5"
    rm $outdir/*.hdf5
    echo "Deleting *.pkl"
    rm $outdir/*.pkl
fi

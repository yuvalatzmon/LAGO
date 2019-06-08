# Probabilistic AND-OR Attribute Grouping for Zero-Shot Learning
Code for our paper: *Atzmon & Chechik, "Probabilistic AND-OR Attribute Grouping for Zero-Shot Learning", UAI 2018* <br>

<a href="https://arxiv.org/abs/1806.02664" target="_blank">paper</a> <br>
<a href="https://www.youtube.com/watch?v=7nj3OPuGTCY" target="_blank">short video</a> <br>
<a href="https://chechiklab.biu.ac.il/~yuvval/LAGO/" target="_blank">project page</a> <br>


## Installation
### Code and Data

 1. Download or clone the code in this repository.
 2. cd to the project directory
 3. Download the data (~500MB) by typing <br> `wget http://chechiklab.biu.ac.il/~yuvval/LAGO/LAGO_data.zip`
 4. Extract the data by typing <br> `unzip -o LAGO_data.zip`
 5. `LAGO_data.zip` can be deleted
### Anaconda Environment
Below are installation instructions under Anaconda.<br>
IMPORTANT: We use python 3.6 and Keras 2.0.2 with TensorFlow 1.1 backend. 

    # Setup a fresh Anaconda environment and install packages:

    # create and switch to new anaconda env
    conda create -n LAGO python=3.6 

    source activate LAGO

    conda install -c anaconda tensorflow-gpu=1.1.0
    conda install -c conda-forge keras=2.0.2
    conda install matplotlib
    conda install pandas ipython jupyter nb_conda


## Directory Structure

directory | file | description
---|---|---
`zero_shot_src/` | * | Sources for the experimental framework, LAGO model, and our python implementation for the ESZSL model. 
`zero_shot_src/`|`ZStrain.py` | The main source file for our Zero-Shot Learning experimental framework. 
`zero_shot_src/`|`LAGO_model.py` | The source file for LAGO model.
`zero_shot_src/`|`ESZSL_model.py` | The source file for ESZSL model. We used it to make sure our ZSL framework can reproduce already published results.
`zero_shot_src/`|`definitions.py` | Common definitions and commandline arguments.
`utils_src/` | * | Sources to useful utility procedures. 
`scripts/` | * | Scripts to reproduce our main results. 
`data/` | * | Xian (CVPR, 2017) zero-shot data for CUB, AWA2 and SUN. With **additional meta data** required by LAGO (sematic groups & original class-descriptions). 
`output/` | * | Contains the outputs of the experimental framework (results & models). **To retrain the models, you have to delete the `output` dir.**


**NOTE:**

* The essence of the model (the soft AND-OR layer) is implemented by the
method `build_transfer_attributes_to_classes_LAGO` ([link](https://github.com/yuvval/LAGO/blob/master/zero_shot_src/LAGO_model.py#L33)) under `zero_shot_src/LAGO_model.py`. 
It is written in pure TensorFlow, and can be easily ported to other TensorFlow
projects that can benefit from such mapping.

## Execute LAGO
Calling `ZStrain.py` directly requires setting several command line arguments. For simplicity, we included scripts that set the relevant arguments and reproduce our main results.  <br>
The `train_dev_*.sh` scripts run LAGO with the development split (100/50 train/validation classes)<br>
The other `train_*.sh` scripts run LAGO or ESZSL with the test split (150/50 train/test classes)<br>

For example:

    source gpu_id=0 scripts/train_SemanticHard_CUB.sh
    
**NOTES:**

1. You **must** run the code from the project root directory.
2. You can select the GPU id to use by either setting gpu_id or the CUDA_VISIBLE_DEVICES environment variable.
3. You can launch several instances of a train script in parallel. 
For example, an instance per GPU. 
Our experimental framework uses a simple locking mechanism, 
that allows to launch without conflicts several instances of a train script in parallel (on multiple
    machines or GPUs). The only requirement is that they are executed on the same filesystem.
4. When an experiment is completed, a file named `metric_results.json` is written
to the training dir. This indicates that an experiment is completed, and there is no need to launch that experiment again.
Therefore, **you have to delete the `output` dir in order to retrain the models**, since this directory already contains results for the trained experiments.

5. For calling `ZStrain.py` directly, see example in `train.sh`. Important: in such case, don't forget to set the following variables: `KERAS_BACKEND=tensorflow PYTHONPATH="./"`

To view the results, see the Jupyter notebooks

## Cite our paper
If you use this code, please cite our paper.

    @inproceedings{atzmon2018LAGO,
    title={Probabilistic AND-OR Attribute Grouping for Zero-Shot Learning},
    author={Atzmon, Yuval and Chechik, Gal},
    booktitle={Proceedings of the Thirty-Forth Conference on Uncertainty in Artificial Intelligence},
    year={2018},
    } 


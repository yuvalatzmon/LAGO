"""
The main script for this zero-shot learning Framework.
It implements training and evaluating the performance for LAGO and ESZSL models.

Author: Yuval Atzmon
"""

import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import tempfile
import pprint
import json

from keras import callbacks
from keras import optimizers
from keras import backend as K

from zero_shot_src import definitions
from zero_shot_src import load_xian_data
from zero_shot_src import semantic_transfer
from zero_shot_src import ESZSL_model
from zero_shot_src import LAGO_model

from utils_src import ml_utils
import utils_src.keras_utils

# Force matplotlib to be headless (no GUI).
plt.switch_backend('Agg')

# Get values for the relevant command line arguments
common_args, unknown_args = definitions.get_common_commandline_args()

data_loaders_factory = {# Features, labels and splits from Xian, CVPR 2017
                        'SUN_Xian': load_xian_data.SUN_Xian,
                        # Features, labels and splits from Xian, CVPR 2017
                        'CUB_Xian': load_xian_data.CUB_Xian,
                        'AWA2_Xian': load_xian_data.AWA2_Xian,
                        }

ZSL_models_factory = {'LAGO': LAGO_model.get_model,
                      'ESZSL': ESZSL_model.get_model}

# Main function for doing the zero-shot-learning framework.
def main():

    ########################################################
    # Initialization
    initialization()
    ########################################################


    ########################################################
    # Get Data
    print_and_flush('Loading Data ...')
    data_loader = data_loaders_factory[common_args.data_loader]()
    data = data_loader.get_data()

    # Prepare Data for ZSL model
    print_and_flush('Prepare Data for ZSL model ...')
    data, \
    X_train, Y_train, Attributes_train, train_classes, \
    X_val, Y_val, Attributes_val, val_classes, \
    input_dim, categories_dim, attributes_dim, \
    class_descriptions_crossval, \
    attributes_groups_ranges_ids = prepare_data_for_ZSL_model(data)
    ########################################################


    ########################################################
    # Get ZSL Keras model
    print_and_flush('Prepare model ...')
    f_build_model = ZSL_models_factory[common_args.model_name]
    model = f_build_model(input_dim, categories_dim, attributes_dim,
        class_descriptions_crossval, attributes_groups_ranges_ids)
    ########################################################


    ########################################################
    # If model is not trained: Train the model. Otherwise, skip training.
    if common_args.model_dir_rerun_inference is None:

        # Get Keras Callbacks for training
        print_and_flush('Configure Training ...');
        sys.stdout.flush()
        training_CB = prepare_callbacks_for_training(model, common_args)


        fit_kwargs = dict(batch_size=common_args.batch_size,
                          epochs=common_args.max_epochs,
                          verbose=2,
                          x=X_train,
                          y=[Y_train, Y_train, Attributes_train],
                          validation_data=(X_val,
                                           [Y_val, Y_val, Attributes_val]),
                          # [Y_val, Attributes_val, Y_val, Y_val]),
                          callbacks=training_CB,
                          )
        if type(model) == ESZSL_model.ESZSL_Model:
            fit_kwargs['model_fullpathname'] = get_file('best-checkpoint')


        # Zero-Shot accuracy metrics to use with Keras callbacks.
        #
        # IMPORTANT NOTE: These metrics calculate the mean *imbalanced* accuracy.
        # They are **ONLY** used for early stopping, since they differ than the
        # final evaluation metric, which is a balanced accuracy metric, in
        # concordance with Xian (CVPR 2017) protocol. We use this metric because
        # it is less complex to implement than the balanced accuracy metric in
        # tensor notation, as required by Keras, and because we observed that
        # there is a reasonable correlation between the imbalanced and balanced
        # metrics. We welcome pull-request to fix this issue, as it may improve
        # the performance of LAGO.
        # NOTE: To comply with Xian protocol, the balanced accuracy metric is executed
        # on the selected model after training completes.
        # See evaluate_performance()
        def val_acc(y, y_pred):
            return subset_accuracy(y, y_pred, val_classes)
        def train_acc(y, y_pred):
            return subset_accuracy(y, y_pred, train_classes)


        # Fit model to data (training)
        print_and_flush('Training ...'); sys.stdout.flush()
        ml_utils.tic() # Initiate tic - toc for timing (like in matlab).
        fit_wrapper(model, f_train_acc=train_acc, f_val_acc=val_acc,
                    fit_kwargs=fit_kwargs)


        # On test stage, we don't monitor the performance.
        # We only save the model when optimization ends.
        # We do it here:
        if common_args.is_test:
            print_and_flush('Saving the test stage model under %s' %
                            get_file('best-checkpoint'))
            model.save(get_file('best-checkpoint'))

            # # For Debugging: Save the data as given to the fit() training method
            #     print('Saving data as given to the fit() method: %s' % get_file('fit_data'))
            #     fit_data = ml_utils.slice_dict(fit_kwargs, ('x', 'y', 'validation_data'))
            #     ml_utils.data_to_pkl(fit_data, get_file('fit_data'))

        ml_utils.toc()
    ########################################################


    ########################################################
    # Evaluation:
    # (1) Load model to evaluate
    # (2) Evaluate performance
    # (3) Save to disk:
    #     (3.a) The evaluation intermediate products
    #                                       (predictions, scores,
    #                                       ground_truth,
    #                                       layers activations (commented out),
    #                                       etc..)
    #     (3.b) Evaluated metrics and other outcomes, as text files.

    # Load the model for evaluation
    print_and_flush(
        'Load the model weights from %s' % get_file('best-checkpoint'))
    model.load_weights(get_file('best-checkpoint'))

    # Evaluate performance and save its intermediate products
    metric_results = evaluate_performance(data, model, save_predictions=
                                          common_args.is_test)

    # Save evaluated metrics and other outcomes, as text files.
    save_outcomes_to_text_files(metric_results, model)
    ########################################################


    ########################################################
    # Finalize
    # (1) Delete temporary directory and files
    # (2) Clear Keras Session
    # (3) Flush stdout
    finalization()
    ########################################################
    return


def get_file(file_type):
    """
    Returns a full path name of a file.

    :param file_type: string, describing the filename to retrieve
    :return: string
    """

    # Cast command line arguments (argparse) to dictionary.
    args = vars(common_args)

    # Get a directory name to save the training and evaluation outcomes
    train_dir = os.path.expanduser(args['train_dir'])
    # Get a directory name to save temporary model checkpoints
    tmp_ckpt_dir = os.path.expanduser(args['tmp_ckpt_dir'])

    # A filename to save the best model checkpoint
    if file_type == 'best-checkpoint':
        fullname = os.path.join(train_dir,
                                'model-ckpt-best.hdf5')

    # A CSV filename to save training log
    elif file_type == 'csv-log':
        fullname = os.path.join(train_dir, 'training_log.csv')

    # CSV filename to dump the soft-groups (Gamma) matrix of the final model.
    elif file_type == 'gamma_kernel':
        fullname = os.path.join(train_dir, 'gamma_kernel.csv')

    # A pkl filename to save the val or test predictions
    elif file_type.startswith('predictions_'):
        fullname = os.path.join(train_dir, '%s.pkl'%file_type)

    # A file name to save the metrics (the performance results).
    elif file_type == 'metric_results':
        fullname = os.path.join(train_dir,
                                'metric_results.json')

    # A file name to touch on the train_dir, that indicates that training is
    # in progress in this directory.
    elif file_type == 'touch':
        fullname = os.path.join(train_dir,
                                'progress.touch')

    # A filename that holds the fit data, as given to the fit() training method.
    # Mainly used for debugging. Currently it is commented out.
    elif file_type == 'fit_data':
        fullname = os.path.join(train_dir,
                                'fit_data.pkl')
    else:
        raise ValueError('Unknown file type: %s' % file_type)

    # noinspection PyUnboundLocalVariable
    return fullname



def is_alternating_train_config():
    """ Returns True if training is configured for an alternating optimization.
        Raises an exception if something is not right
    """
    if common_args.SG_alternating:
        if common_args.model_variant in ['LAGO_SemanticSoft', 'LAGO_KSoft']\
                and common_args.SG_trainable:
            return True
        else:
            # Raise exception (currently only relevant to Singletons)
            raise ValueError('SG_alternating is Set (1), but model_variant or '
                             'SG_trainable are not correctly set.')
    return False

def prepare_data_for_ZSL_model(data):
    """
    Prepare (parse) data required for the ZSL model build, and for training
    """
    # TODO: Document this method better

    X_train, Y_train, X_val, Y_val, \
    df_class_descriptions_by_attributes = \
        ml_utils.slice_dict_to_tuple(data,
                                     'X_train, Y_train, X_val, Y_val, '
                                     'df_class_descriptions_by_attributes')
    attributes_name = df_class_descriptions_by_attributes.columns

    input_dim = X_train.shape[1]
    attributes_dim = len(attributes_name)
    categories_dim = 1 + df_class_descriptions_by_attributes.index.max()

    train_classes = np.unique(Y_train)
    val_classes = np.unique(Y_val)

    # Get start index of each group (sequentially).
    attributes_groups_ranges_ids = \
        semantic_transfer.get_attributes_groups_ranges_ids(
            attributes_name)

    # class_descriptions_crossval is for taking only the specific set of
    # class descriptions per cross-validation splits. The other classes are nulled.
    # In principal, this is redundant. We could use just use a single matrix
    # for all the classes, since we won't push gradients for classes that don't
    # participate. Yet, it is here to make sure that there is  no leakage of
    # information from validation/test class descriptions during training.
    class_descriptions_crossval = {}

    # Repeat for 'train' and 'validation' sets.
    # Note: If we are in the testing phase, then:
    #       'train' relates to **trainval** samples
    #       'val'   relates to **test** samples
    for xvset in ['train', 'val']:
        # Extract class descriptions only for current cross-val (xv) set.
        class_descriptions_crossval[xvset] = np.zeros(
            (categories_dim, len(attributes_name)))
        class_descriptions_crossval[xvset][locals()[xvset + '_classes'],
        :] = \
            df_class_descriptions_by_attributes.loc[
            locals()[xvset + '_classes'], :]

        if common_args.LG_norm_groups_to_1:
            """ Normalize the semantic description in each semantic group to 
                sum to 1, in order to comply with the mutual-exclusion 
                approximation. This is crucial for the LAGO_Semantic* variant.

                See "IMPLEMENTATION AND TRAINING DETAILS" in paper.
            """
            class_descriptions_crossval[xvset] = \
                semantic_transfer.norm_groups_larger_1(
                    class_descriptions_crossval[xvset],
                    attributes_groups_ranges_ids)

    # Ground-truth attribute-labels per-image, are the attributes that
    # describe the image class. Only used when attributes regularization is positive
    Attributes_train = df_class_descriptions_by_attributes.loc[Y_train,
                       :].values
    Attributes_val = df_class_descriptions_by_attributes.loc[Y_val,
                     :].values
    # Add per image attributes supervision to the data dict.
    # It will be used during evaluations
    data['Attributes_train'] = Attributes_train
    data['Attributes_val'] = Attributes_val

    return data, \
           X_train, Y_train, Attributes_train, train_classes, \
           X_val, Y_val, Attributes_val, val_classes, \
           input_dim, categories_dim, attributes_dim, \
           class_descriptions_crossval, \
           attributes_groups_ranges_ids


def prepare_callbacks_for_training(model, common_args):
    """
    Prepare Keras Callbacks for model training
    Returns a list of keras callbacks
    """
    training_CB = []

    # Note that checkpoint is saved after metrics eval and *before*
    # categorical zero-shot transfer update. I.e. For reflecting the metric,
    # it is not saved with the latest custom updated leaf weights
    if common_args.is_dev:
        # Set the monitor (metric) for validation.
        # This is used for early-stopping during development.
        monitor, mon_mode = model.monitor, model.mon_mode
        print(f'monitor = {monitor}')

        # Save a model checkpoint only when monitor indicates that the best
        # performance so far
        training_CB += [
            callbacks.ModelCheckpoint(monitor=monitor, mode=mon_mode,
                                      save_best_only=True,
                                      filepath=get_file('best-checkpoint'),
                                      verbose=common_args.verbose)]

        # Set an early stopping callback
        training_CB += [callbacks.EarlyStopping(monitor=monitor, mode=mon_mode,
                                                patience=common_args.patience,
                                                verbose=common_args.verbose,
                                                min_delta=common_args.min_delta)]

    # An option to dump results to tensorboard
    if common_args.tensorboard_dump:
        training_CB += [callbacks.TensorBoard(
            log_dir=os.path.expanduser(common_args.train_dir))]

    # Log training history to CSV
    training_CB += [callbacks.CSVLogger(get_file('csv-log'), separator='|',
                                        append=is_alternating_train_config())]

    # Touch a progress indicator file on every epoch end
    training_CB += [callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: ml_utils.touch(get_file('touch')))]
    # Flush stdout buffer on every epoch
    training_CB += [callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: sys.stdout.flush())]

    return training_CB


def print_and_flush(*args, **kwargs):
    """
    Print to stdout and immediately flush the buffer
    """
    print(*args, **kwargs)
    # Flush stdout buffer (to screen)
    sys.stdout.flush()


def fit_wrapper(base_model, f_train_acc, f_val_acc, fit_kwargs):
    """
    Fit the model with either normal keras fit, or alternating optimization

    Alternating training case is used for learning with soft-groups. This was
    empirically beneficial for learning the Gamma (soft-group assignments)
    parameters. Here on each epoch we alternate between training the attribute
    prediction model, or the soft-group assignments matrix.
    In paper: Section 4.2
    """

    if not is_alternating_train_config():
        # Regular training case: call Keras model compile, and Keras model fit

        if common_args.model_name == "LAGO":
            # Setting the attribute weights to trainable
            base_model.layers[1].trainable = True
            base_model.layers[2].trainable = common_args.SG_trainable
            base_model.layers[3].trainable = common_args.SG_trainable

        base_model.compile(optimizer=init_optimizer(common_args.initial_learning_rate),
                           loss=base_model.loss_list,
                           loss_weights=base_model.loss_weights,
                           metrics={'ZS_train_layer': [f_train_acc],
                                    'ZS_val_layer': [f_val_acc]})

        base_model.summary()
        base_model.fit(**fit_kwargs)
        return
    else:
        # -- YES alternating_training --
        # We compile two models.
        #  (1) The base model, for attributes prediction,
        #      where the parameters of Gamma are frozen.
        #  (2) The groups model, for learning the soft-group assignments,
        #      where the parameters of attributes prediction are frozen.


        # Setting the attribute weights to trainable
        base_model.layers[1].trainable = True
        base_model.layers[2].trainable = False
        base_model.layers[3].trainable = False
        base_model.compile(optimizer=init_optimizer(common_args.initial_learning_rate),
                           loss=base_model.loss_list,
                           loss_weights=base_model.loss_weights,
                           metrics={'ZS_train_layer': [f_train_acc],
                               'ZS_val_layer': [f_val_acc]})
        base_model.summary()

        # Setting another model with trainable Gamma matrix
        groups_model = LAGO_model.LAGO_Model(base_model.inputs, base_model.predictions)
        groups_model.layers[1].trainable = False
        groups_model.layers[2].trainable = True
        groups_model.layers[3].trainable = False
        groups_model.compile(optimizer=init_optimizer(common_args.SG_gamma_lr),
                       loss=base_model.loss_list,
                       loss_weights=base_model.loss_weights,
                       metrics={'ZS_train_layer': [f_train_acc],
                               'ZS_val_layer': [f_val_acc]})
        groups_model.summary()

        # # For debugging, log to a text file the updates on every batch (commented out)
        # if False:
        #     fit_kwargs['callbacks'].append(
        #         utils_src.keras_utils.LogUpdatesCallback(
        #         base_model, os.path.expanduser(common_args.train_dir),
        #             start_epoch=1, end_epoch=2))

        fit_kwargs['epochs'] = 1
        for iter in range(common_args.max_epochs):
            print_and_flush(f'\n\niter {iter}:')
            # Gamma_ker = base_model.get_weights()[2]
            # print_and_flush('DEBUG: Max of row0 @Gamma:',
            #       softmax(common_args.SG_gain*Gamma_ker)[0, :].max())

            print_and_flush('Training W')
            base_model.fit(**fit_kwargs)

            print_and_flush('Training Gamma')
            groups_model.fit(**fit_kwargs)

        return


def config_tf_session(common):
    # Configuring the TF session while prohibiting it from using
    # all available memory. TF will use only required amount of
    # memory at a time
    if K.backend() == 'tensorflow':
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=common_args.gpu_memory_fraction, )
        if common_args.gpu_memory_fraction == 1:
            gpu_options.allow_growth = True
        session_config = tf.ConfigProto(gpu_options=gpu_options)
        sess = tf.Session(config=session_config)
        if False:  # Replace with a TF debugging session
            from tensorflow.python import debug as tf_debug
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        K.set_session(sess)


def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div


def initialization():

    def finalize_current_method():
        """
        Steps to execute when before exiting initialization() method
        """

        # Config tensorflow session according to setting commandline arguments
        config_tf_session(common_args)

        # Pretty print command line arguments (including default values)
        print('args:')
        print(json.dumps(vars(common_args), indent=4))
        print('Unknown args:')
        pprint.pprint(unknown_args)
        sys.stdout.flush()

    # if not None: Re-run inference, from the path in model_dir_rerun_inference,
    # without training the model.
    if common_args.model_dir_rerun_inference is not None:
        finalize_current_method()
        return

    # If train dir is None, set it to a random dir under
    # <common_args.temp_ckpt_dir>/ZStrain/<temporary name>
    # (useful during dev/debugging)
    base_tmp = os.path.join(os.path.expanduser(common_args.temp_ckpt_dir),
                            'ZStrain')
    if not os.path.exists(base_tmp):
        os.makedirs(base_tmp)
    common_args.base_tmp = base_tmp
    common_args.tmp_ckpt_dir = tempfile.mkdtemp(dir=common_args.base_tmp)

    if common_args.train_dir is None:
        # Set --train_dir to a temp dir
        common_args.train_dir = tempfile.mkdtemp(dir=base_tmp)
        common_args.base_output_dir = base_tmp


    # ml_utils.check_experiment_progress checks if the current selection of
    # params is already In-progress or Completed. If True, then exit, and don't train the model.
    # This allows to execute **in parallel**, without conflicts, several
    # hyper-param search scripts that use the same filesystem (on multiple
    # machines or GPUs).
    #
    # Note: This was not exhaustively tested to avoid conflict 100%of the times,
    # but we are OK if on rare times, two training scripts are executed with
    # the same parameters. The only problem in such case is that the
    # training_log csv file may become corrupt.
    #
    # The general idea, is that we give a meaningful name to the train_dir, such
    # that it describes the configuration of the current experiment.
    # Then, on every epoch a file name is "touched" on the train_dir. This file
    # indicates that training is in progress in this directory, and therefore
    # we will exit this instance.
    # Moreover, if the metric_results file exists, it means that this experiment
    # had completed. In that case, we also exit this instance.
    # There are two exceptions for the above conditions:
    # (1) if --obsolete_result_timeout_hours exceeds, then we rerun this instance.
    # (2) if --in_progress_timeout_minutes exceeds, then we will keep running
    #     this instance.
    #

    # check_experiment_progress() returns 'Unfulfilled' if the current instance is
    # Ok to run.
    current_experiment_progress = ml_utils.check_experiment_progress(
        common_args.train_dir, get_file('metric_results'), get_file('touch'),
        common_args.in_progress_timeout_minutes,
        common_args.obsolete_result_timeout_hours,
        common_args.base_output_dir)

    if current_experiment_progress != 'Unfulfilled':
        # if not 'Unfulfilled' then exit
        print('initialization(): Exiting because current experiment ' +
              f'is {current_experiment_progress}')
        exit()

    # Make train dir if not exists
    if not os.path.exists(os.path.expanduser(common_args.train_dir)):
        os.makedirs(os.path.expanduser(common_args.train_dir))



    # Select the gpu id according to command line argument --default_gpu_id
    # If CUDA_VISIBLE_DEVICES is set, it overrides this selection.
    try:
        int(os.environ.get('CUDA_VISIBLE_DEVICES'))
    except:
        gpu_id = common_args.default_gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

    finalize_current_method()

def finalization():
    """
    (1) Delete temporary directory and files
    (2) Clear Keras Session
    (3) Flush stdout
    """

    # Delete temp dir
    if hasattr(common_args, 'base_tmp'):
        if os.path.expanduser(common_args.train_dir).startswith(common_args.base_tmp):
            del_file = os.path.expanduser(common_args.train_dir)
            print('Deleting %s' % del_file)
            print(ml_utils.run_bash('rm -r %s' % del_file))

    # Clear Keras session
    K.clear_session()

    # Flush stdout
    sys.stdout.flush()


def xian_per_class_accuracy(y_true, y_pred, num_class=None):
    """ A balanced accuracy metric as in Xian (CVPR 2017). Accuracy is
        evaluated individually per class, and then uniformly averaged between
        classes.
    """

    y_true = y_true.flatten().astype('int32')

    counts_per_class = pd.Series(y_true).value_counts().to_dict()
    if num_class is None:
        num_class = len(counts_per_class)  # e.g. @CUB: 50, 100, 150
        # num_class = len(np.unique(y_true))

    accuracy = ((y_pred == y_true) / np.array(
        [counts_per_class[y] for y in y_true])).sum() / num_class

    return accuracy.astype('float32')


def subset_accuracy(y_gt, y_prediction, subset_indices):
    y_prediction = tf.transpose(
        tf.gather(tf.transpose(y_prediction), subset_indices))
    arg_p = tf.gather(subset_indices, tf.arg_max(y_prediction, 1))
    y_gt = tf.transpose(y_gt)
    return tf.reduce_mean(tf.to_float(tf.equal(tf.to_int64(y_gt), arg_p)))

def evaluate_performance(data, model, save_predictions=False):
    X_train, Y_train, Attributes_train, \
    X_val, Y_val, Attributes_val, \
    df_class_descriptions_by_attributes =\
        ml_utils.slice_dict_to_tuple(data, 'X_train, Y_train, Attributes_train, '
                                           'X_val, Y_val, Attributes_val, '
                                           'df_class_descriptions_by_attributes')

    train_classes = np.unique(Y_train)
    val_classes = np.unique(Y_val)

    print('Inference on train data ...')
    predictions_train = model.predict(X_train,
                                      batch_size=common_args.batch_size)
    print('This run arguments ')
    print(json.dumps(vars(common_args), indent=4))
    print('Inference on validation data ...')
    predictions_val = model.predict(X_val, batch_size=common_args.batch_size)

    # # tensor debugging: allowing to inspect tensor activations
    # S_val_layer = model.output_layers_dict['ZS_val_layer']
    # debug_out_tensors = {}
    # try:
    #     debug_out_tensors['Pr_a_cond_x']=S_val_layer.debug_out_tensors['Pr_a_cond_x']
    #     debug_out_tensors['Pr_a_cond_z']=S_val_layer.debug_out_tensors['Pr_a_cond_z']
    #     debug_out_tensors['val_out']=S_val_layer.debug_out_tensors['out']
    #     debug_out_tensors['val_normalized(a|z)'] = S_val_layer.debug_out_tensors['normalized(a|z)']
    #     debug_out_tensors['val_G']=S_val_layer.debug_out_tensors['G']
    # except:
    #     pass
    #
    # debug_activations = {}
    # if debug_out_tensors:
    #     functors = dict(
    #         (k, K.function([model.inputs] + [K.learning_phase()], [out])) for
    #         k, out in debug_out_tensors.items())
    #     debug_activations = dict((k, func([X_val])[0]) for k, func in functors.items())
    #     # assert(np.isclose(predictions_val[1], debug_activations['val_out'], rtol=1e-3, atol=1e-3).all())

    pred_val = val_classes[(predictions_val[1][:, val_classes]).argmax(axis=1)]

    val_zs_accuracy = float(xian_per_class_accuracy(Y_val, pred_val))
    print('Val ZS accuracy (%d categories) = %2.2f %%' % (
        len(val_classes), 100 * val_zs_accuracy))


    pred_zs_train = train_classes[
        (predictions_train[0][:, train_classes]).argmax(axis=1)]

    train_zs_accuracy = float(xian_per_class_accuracy(Y_train, pred_zs_train))
    print('Train ZS accuracy (%d categories) = %2.2f %%' % (
        len(train_classes), 100 * train_zs_accuracy))
    if save_predictions:
        ml_utils.data_to_pkl(dict(pred_argmax_classes=pred_val,
                                  pred_score_classes=predictions_val[1],
                                  gt_classes=Y_val,
                                  classes_ids=val_classes,
                                  pred_attributes=predictions_val[2],
                                  # pred_G=debug_activations.get('val_G', None),
                                  # a_given_z_normed_weights=debug_activations.get('val_normalized(a|z)', None),
                                  gt_attributes=Attributes_val,
                                  common_args=common_args,
                                  ),
                             get_file('predictions_val'))
        print('Written val predictions to %s'%get_file('predictions_val'))
        ml_utils.data_to_pkl(dict(pred_argmax_classes=pred_zs_train,
                                  pred_score_classes=predictions_train[0],
                                  gt_classes=Y_train,
                                  classes_ids=train_classes,
                                  pred_attributes=predictions_train[2],
                                  gt_attributes=Attributes_train,
                                  common_args=common_args,
                                  ),
                             get_file('predictions_train'))
        print('Written train predictions to %s'%get_file('predictions_train'))

    metric_results = dict(metric_train_zs_accuracy=train_zs_accuracy,
                          metric_val_zs_accuracy=val_zs_accuracy)
    return metric_results

def save_outcomes_to_text_files(metric_results, model):

    if common_args.dump_SG_to_text:
        # Dump model Soft-Groups Gamma matrix to CSV text file
        # (if applicable)
        try:
            gamma_id = utils_src.keras_utils.layer_name_to_ids('Gamma', model)[0]
            Gamma_kernel = model.get_weights()[gamma_id]
            # assert(model.get_layer(index=gamma_id).name ==  'ZS_train_ctg')
            print(Gamma_kernel.shape)
            fname = os.path.expanduser(get_file('gamma_kernel'))
            np.savetxt(fname, Gamma_kernel, delimiter=",", fmt='%1.4f')
        except IndexError:
            # Gamma matrix is not applicable to this model
            pass


    # Save metric results to JSON text file
    # Adding the execution parameters (arguments) to the results
    metric_results.update(vars(common_args))
    fname = os.path.expanduser(get_file('metric_results'))
    print(f'Writing metric results to {fname}')
    with open(fname, 'w') as f:
        f.write(json.dumps(metric_results, indent=4))


def init_optimizer(lr):
    opt_name = common_args.optimizer.lower()
    if opt_name == 'adam':
        optimizer = optimizers.Adam(lr=lr)
    elif opt_name == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=lr)
    elif opt_name == 'sgd':
        optimizer = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError('unknown optimizer %s' % opt_name)

    return optimizer


if __name__ == "__main__":
    main()

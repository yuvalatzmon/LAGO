"""
common definitions and commandline arguments for zero_shot_src

Author: Yuval Atzmon
"""

import argparse
import tensorflow as tf
import re
import os, pickle

tf.logging.set_verbosity(tf.logging.INFO)

import abc # support for abstract classes (in order to define an API)

class ZSLData(abc.ABC):
    """ Defines the class API for getting ZSL data in this framework
    """

    @abc.abstractmethod
    def get_data(self):
        """
        load the data, and return a dictionary with the following keys:
        'X_train': input features train matrix. shape=[n_samples, n_features]
        'Y_train': train labels vector. shape=[n_samples, ]
        'X_val': input features validation (or test) matrix. shape=[n_samples, n_features]
        'Y_val': validation (or test) labels vector. shape=[n_samples, ]
        'df_class_descriptions_by_attributes': a dataframe of class description
            by attributes for all classes (train&val).
            shape=[n_classes, n_attributes]
            rows index = class ids
            column index = attributes names
        'attributes_name': simply df_class_descriptions_by_attributes.columns
        attributes naming format is: <group_name>::<attribute_name>, e.g.:
                                     shape::small
                                     shape::round
                                     head_color::red
                                     head_color::orange


        :return: dict()
        """

def get_common_commandline_args():
    parser = argparse.ArgumentParser()
    # region UAI publication
    parser.add_argument("--sort_attr_by_names", type=int, default=0,
                        help="If this flag is set, then we sort attributes by "
                             "names. The underlying assumtion is that the naming"
                             " convention is 'group_name::attribute_name'. "
                             "Therefore enabling this sort will cluster together"
                             "attributes from the same group. This is needed"
                             "because LAGO with Semantic groups requires that "
                             "kind of name clustering.")
    # endregion

    parser.add_argument("--model_name", type=str, default='LAGO',
                        help="Attributes model name. \in {'LAGO', 'ESZSL'}.")
    parser.add_argument("--model_variant", type=str, default=None,
                        help="The model variant \in { 'LAGO_SemanticSoft', "
                             "'Singletons', 'LAGO_KSoft', None }. "
                             "For LAGO-SemanticHARD choose LAGO_SemanticSoft"
                             "and set --SG_trainable=0")


    # region Data params
    parser.add_argument("--use_trainval_set", type=int, default=0,
                        help="Use trainval (train + val) set for training")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset dir")
    parser.add_argument("--data_loader", type=str, default=None,
                        help="Loader class for the data. Select key from "
                             "ZStrain.data_loaders_factory")
    parser.add_argument("--use_xian_normed_class_description", type=int, default=0,
                        help="Use Xian (CVPR 2017) class description. This is a "
                             "L2 normalized version of the mean attribute values"
                             "that are provided with the datasets. "
                             "This can **not** be used with LAGO.")
    # endregion

    # region LAGO params
    parser.add_argument("--LG_beta", type=float, default=0,
                        help="hyper-param: beta")
    parser.add_argument("--LG_lambda", type=float, default=1e-7,
                        help="hyper-param: gamma")
    parser.add_argument("--LG_norm_groups_to_1", type=int, default=0,
                        help="Normalize the semantic description in each "
                             "semantic group to sum to 1, in order to comply "
                             "with the mutual-exclusion approximation. "
                             "This is crucial for the LAGO_Semantic* variants."
                             "See IMPLEMENTATION AND TRAINING DETAILS on LAGO paper.")
    parser.add_argument("--LG_uniformPa", type=int, default=1,
                        help="LAGO: Use a uniform Prior for Pa")
    parser.add_argument("--LG_true_compl_x", type=int, default=1,
                        help="LAGO: Set P(complementary attrib.|x)=1")
    parser.add_argument("--attributes_weight", type=float, default=1,
                        help="Attributes weight in loss function.")
    parser.add_argument("--orth_init_gain", type=float, default=0.1,
                        help="Gain for keras initializers.Orthogonal: "
                             "We didn't tune this hyper param. Except once, on "
                             "a very preliminary experiment.")

    # LAGO Soft Groups (SG) params:
    parser.add_argument("--SG_trainable", type=int, default=1,
                        help="Set SoftGroup weights to be trainable.")
    parser.add_argument("--SG_alternating", type=int, default=None,
                        help="Alternating training for soft-groups. This was "
                             "empirically beneficial when learning the Gamma "
                             "(soft-group assignments) parameters. Here on each "
                             "epoch we alternate between training the attribute "
                             "prediction model, or the soft-group assignments "
                             "matrix. (In paper: Section 4.2)"
                             "0=False, 1=True, None=Select automatically "
                             "(by setting SG_alternating = SG_trainable)")
    parser.add_argument("--SG_gain", type=float, default=1,
                        help="hyper-param: Softmax kernel gain with SoftGroups")
    parser.add_argument("--SG_gamma_lr", type=float, default=1e-2,
                        help="hyper-param: The learning rate for learning the "
                             "parameters of Gamma when SG_alternating=1")
    parser.add_argument("--SG_num_K", type=int, default=None,
                        help="hyper-param: Number of groups for LAGO_KSoft")
    parser.add_argument("--SG_seed", type=int, default=None,
                        help="Random seed for Gamma matrix when using LAGO_KSoft.")
    parser.add_argument("--SG_psi", type=float, default=0,
                        help="hyper-param: Psi, the regularization coefficient "
                             "for Semantic prior on Gamma.")

    # endregion
    # region ES-ZSL params
    parser.add_argument("--ES_use_mean", type=float, default=0,
                        help="ES-ZSL: Use mean SE loss instead of sum SE loss.")

    parser.add_argument("--ES_gamma", type=float, default=1e3,
                        help="ES-ZSL gamma hyper-param.")
    parser.add_argument("--ES_lambda", type=float, default=1,
                        help="ES-ZSL lambda hyper-param.")
    # endregion


    # region Optimization params
    parser.add_argument("--initial_learning_rate", type=float,
                        default=3e-3,
                        help='Initial learning rate')
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help='Optimizer name. \in {"Adam, RMSprop, sgd"}')
    parser.add_argument("--repeat", type=int, default=0,
                        help='Repetition id')
    parser.add_argument("--batch_size", type=int, default=64,
                        help='Batch size')
    parser.add_argument("--max_epochs", type=int, default=100,
                        help='Max number of epochs to train')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='Momentum for SGD (with Nesterov) when applicable')

    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping: number of epochs with no improvement after which training "
                             "will be stopped.")
    parser.add_argument("--min_delta", type=float, default=1e-7,
                        help='minimum change in the monitored quantity to qualify as an improvement')
    # endregion

    # region Training script configuration
    parser.add_argument("--default_gpu_id", type=int, default=0,
                        help='Default GPU id to set as CUDA_VISIBLE_DEVICES, '
                             'ONLY USED when CUDA_VISIBLE_DEVICES is undefined.'
                             'Default None automatically chooses available gpu.')
    parser.add_argument("--dump_SG_to_text", type=int, default=0,
                        help='Dump the soft-groups Gamma matrix to text fil')

    parser.add_argument("--train_dir", type=str, default=None,
                        help="Directory to write checkpoints and training "
                             "history. Default (None) will write to a random "
                             "temporary dir ")
    parser.add_argument("--base_output_dir", type=str,
                        default=None,
                        help="Base directory to write the results."
                             "THIS ARGUMENT MUST BE SET!")
    parser.add_argument("--model_dir_rerun_inference", type=str,
                        default=None,
                        help='Model path name for re-running inference, without'
                             'training the model. If this not None, then we '
                             'skip training phase.')
    parser.add_argument("--temp_ckpt_dir", type=str,
                        default='/tmp/',
                        help='Base directory name when train_dir is None.')
    parser.add_argument("--gpu_memory_fraction", type=float, default=1,
                        help='GPU fix memory fraction to use, in (0,1].'
                             'NOTE: Setting 1, instead allows '
                             'dynamic growth (allow_growth=True).')
    parser.add_argument("--verbose", type=int, default=1, help='Verbose')
    parser.add_argument("--tensorboard_dump", type=int, default=0,
                        help='Dump tensorboard events to train_dir')
    parser.add_argument("--in_progress_timeout_minutes", type=int, default=10,
                        help='Any modification in the train_dir below this '
                             'amount of time (minutes), indicates that the '
                             'current experiment (with specific params) is '
                             'already in-progress, and will be skipped.')
    parser.add_argument("--obsolete_result_timeout_hours", type=int,
                        default=None,
                        help='If a result file exists for longer than the '
                             'amount of time (hours), indicates that the '
                             'current experiment (with specific params) is '
                             'already obsolete. In that case the current path '
                             'will be deleted')

    # endregion


    args, unknown_args = parser.parse_known_args()

    # Add arguments, for better readability:
    #  (1) Using trainval set, means that we on test stage.
    vars(args)['is_test'] = args.use_trainval_set
    #  (2) NOT using trainval set, means that we on development stage.
    vars(args)['is_dev'] = not args.is_test

    # Default computed values
    if args.SG_seed is None:
        args.SG_seed = args.repeat+1000
    vars(args)['inference_noise_seed'] = args.repeat + 1001

    if args.base_output_dir is None:
        raise ValueError('--base_output_dir must be set')
    # Assertion tests
    if args.SG_psi > 0:
        # Allow only LAGO_SemanticSoft for semantic prior
        assert(args.model_variant == 'LAGO_SemanticSoft')

    if args.LG_norm_groups_to_1:
        # Allow only when the model_variant uses semantic groups.
        # It's irrelevant for 'LAGO-Singletons' and unsupported for 'LAGO-KSoft'.
        assert('Semantic' in args.model_variant)

    if 'LAGO' in args.model_name:
        # LAGO can't allow using Xian (CVPR 2017) class description,
        # because this description is a **L2 normalized** version of the mean
        # attribute values. Such normalization removes the probabilistic meaning
        # of the attribute-class description, which is a key ingredient of LAGO.
        #
        # To be more convinced that Xian class description is L2 normalized,
        # search load_xian_data.py for:
        #   assert (check_official_class_description_l2_norm_equals_xian2017)
        assert(args.use_xian_normed_class_description == 0)

    # Set the default behaviour for SG_alternating:
    # Make it alternating when the soft-group assignment are trainable
    if args.SG_alternating is None and args.model_name == 'LAGO':
        args.SG_alternating = args.SG_trainable

    if args.model_dir_rerun_inference is not None:
        path = os.path.expanduser(args.model_dir_rerun_inference)
        results_fname = os.path.join(os.path.expanduser(path), 'predictions_val.pkl')
        with open(results_fname, 'rb') as f:
            rval = pickle.load(f)  # rval - a bad name for a dictionary that aggregate the experiment results
        vars(args).update(**vars(rval['common_args']))
        args.model_dir_rerun_inference = path

        # change path name from absolute to relative
        for key in ('train_dir', ):
            renamed_dir = re.sub('/(.*/ln/)', '~/ln/', vars(args)[key])
            print(renamed_dir)
            vars(args)[key] = renamed_dir


    return args, unknown_args


common_args, _ = get_common_commandline_args()

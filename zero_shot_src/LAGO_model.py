"""
LAGO implementation in Keras, with TensorFlow backend.

Note that the crux of the model (the soft AND-OR layer) is implemented by the
procedure build_transfer_attributes_to_classes_LAGO().
It is written in pure TensorFlow, and can be easily ported to other TensorFlow
projects that can benefit from such mapping.

Author: Yuval Atzmon
"""

import numpy as np
import tensorflow as tf

import keras
from keras import models
from keras import initializers
from keras import layers
from keras import backend as K
Model = models.Model
Input = layers.Input
Dense = layers.Dense

from zero_shot_src import semantic_transfer
from zero_shot_src.semantic_transfer import semantic_transfer_Layer
from zero_shot_src import definitions

from utils_src.keras_utils import sparse_categorical_loss

common_args, _ = definitions.get_common_commandline_args()


def build_transfer_attributes_to_classes_LAGO(
        Pr_a_cond_x, # attributes prediction
        Pr_a_cond_z, # class descriptions, shape: Z x A
        Gamma, # group assignments for attributes
        debug_out_tensors # Tensors to query for debugging
        ):

    """ This builds the LAGO soft AND-OR layer. Equations 4, 5 in paper.
    """

    ######### init / definitions
    # Set the number of groups (K)
    num_K = Gamma.shape.as_list()[1]
    # Set the number of attributes (A)
    num_A = Gamma.shape.as_list()[0]

    Pr_a_cond_z = Pr_a_cond_z.astype('float32')

    # Get the categories for this classification head.
    # As those that have non-zero class descriptions
    current_category_ids = np.flatnonzero(Pr_a_cond_z.sum(axis=1))

    # # Adding tensor debugging for Pr_a_cond_z
    # # (allowing to inspect its activations)
    # debug_out_tensors['Pr_a_cond_z'] = Pr_a_cond_z

    # Get ids of non-zero categories
    # (because other zeroed categories are placeholders)

    # uniform prior over the categories
    Pr_z = 1. / len(current_category_ids)

    ######### Prepare for soft-OR and approximate the complementary terms
    # In paper: Section 3 --> "The Within-Group Model", Section 3.1 and
    #           Supplementary Eq. (A.24)
    #
    # Although the soft-OR is a simple weighted sum, this part is a little
    # cumbersome. This is because we need to calculate the complementary term
    # in each group, for Pr(a|z), Pr(a|x), Pr(a) and then concatenate it to the
    # groups.

    # Eval Pr(a|z) for complementary terms.
    # In Supplementary: Equation (A.24)
    Pr_acomp_cond_z = tf.reduce_prod(
        1 - Pr_a_cond_z[:, :, None] * Gamma[None, :, :], axis=1)

    # Concat complementary terms to Pr(a|z).
    # As in paper, we denote by "prime" when groups include complementary terms
    Pr_a_prime_cond_z = tf.concat((Pr_a_cond_z, Pr_acomp_cond_z), 1)
    # Concat Gamma columns for complementary terms
    Gamma_prime = tf.concat((Gamma, np.eye(num_K)), 0)

    # Approximate Pr(a_complementary|x).
    # In paper: Section 4.2 --> Design Decisions
    if common_args.LG_true_compl_x:
        # Alternative 1: By a constant Pr(a_k^c|x)=1
        Pr_acomp_cond_x = tf.ones((tf.shape(Pr_a_cond_x)[0], num_K))
    else:
        # Alternative 2: With De-Morgan and product-factorization
        Pr_acomp_cond_x = tf.reduce_prod(
            1 - Pr_a_cond_x[:, :, None] * Gamma[None, :, :], axis=1)

    # Concat complementary terms to Pr(a|x)
    Pr_a_prime_cond_x = tf.concat((Pr_a_cond_x, Pr_acomp_cond_x), 1)

    # Evaluate (by marginalize) Pr(a)
    Pr_a = tf.reduce_sum(tf.gather(Pr_a_cond_z, current_category_ids),
                         axis=0) * Pr_z
    if common_args.LG_uniformPa == 1:  # Uniform Prior
        Pr_a = tf.reduce_mean(Pr_a) * tf.ones((num_A,))

    # Approximate P(a_complementary) with De-Morgan and product-factorization
    Pr_acomp = tf.transpose(
        tf.reduce_prod(1 - Pr_a[:, None] * Gamma[None, :, :], axis=1))
    # Concat complementary terms to Pr(a)
    Pr_a_prime = tf.concat((Pr_a[:, None], Pr_acomp), 0)

    ######### Make the Soft-OR calculation
    # Weighted attributes|class: Pr(a_m|z)/Pr(a_m)
    Pr_a_prime_cond_z_norm = tf.transpose(
        tf.transpose(Pr_a_prime_cond_z) / Pr_a_prime)

    # Weighted attributes|image: [Pr(a_m|z)/Pr(a_m)]*Pr(a_m|x)
    Pr_a_cond_x_weighted = (Pr_a_prime_cond_x[:, :, None] *
                            tf.transpose(Pr_a_prime_cond_z_norm[:, :, None]))

    # Generate each gkz by G = dot(Gamma, Weighted attributes|x)
    # In paper: Equation 5, Soft-Groups Soft-OR
    # Result is a batch of matrices of g_{kz} transposed. shape=[?, |Z|, K]
    G = tf.tensordot(tf.transpose(Pr_a_cond_x_weighted, perm=[0, 2, 1]),
                     Gamma_prime, axes=((2,), (0,)))

    # # Adding tensor debugging
    # # (allowing to inspect its activations)
    # debug_out_tensors['G'] = G  # (gkz matrix)
    # debug_out_tensors['normalized(a|z)'] = Pr_a_prime_cond_z_norm
    # debug_out_tensors['Pr_a_cond_x'] = Pr_a_cond_x

    ######### Make the soft AND calculatio as product over groups (AND)
    # In paper, Section 3 --> "Conjunction of Groups", Eq. 4

    ## Product of groups (faster in log space)
    logG = tf.log(G)
    log_Pr_z_cond_x = tf.reduce_sum(logG, axis=2)

    # Move to 64bit precision (just for few computations),
    # because 32bit precision cause NaN when number of groups is large (>40).
    # This happens because the large number of groups multiplication requires
    # a high dynamic range.
    log_Pr_z_cond_x = tf.to_double(log_Pr_z_cond_x)
    Pr_z_cond_x = tf.exp(log_Pr_z_cond_x)

    ##### Normalize the outputs by their sum across classes.
    # This way we make sure the output is a probability distribution,
    # since some approximations we took may render the values out of the simplex
    # In paper: Section 3.2
    eps = 1e-12
    Pr_z_cond_x = Pr_z_cond_x / (tf.reduce_sum(Pr_z_cond_x, axis=1, keep_dims=True) + eps)

    # Move back to 32 bit precision
    Pr_z_cond_x = tf.to_float(Pr_z_cond_x)
    return Pr_z_cond_x


def get_model(input_dim, categories_output_dim, attributes_output_dim,
              class_descriptions, attributes_groups_ranges_ids):
    """
    Build a Keras model for LAGO
    """

    # Build model
    print('Building LAGO_Model')

    # Define inputs for the model graph
    inputs = Input(shape=(input_dim,))


    ############################################
    # Define the layer that maps an image representation to semantic attributes.
    # In the paper, this layer is f^1_W, i.e. The mapping X-->A, with parameters W.
    # Its output is an estimation for Pr(a|x)

    # Set bias regularizer of the keras layer according to beta hyper param
    # Note that the matrix weights are regularized explicitly through the loss
    # function. Therefore, no need to also set them when defining the layer
    L2_coeff = common_args.LG_beta
    bias_regularizer = keras.regularizers.l2(L2_coeff)

    semantic_embed_layer = Dense(attributes_output_dim,
                                 activation='sigmoid',
                                 trainable=True,
                                 name='attribute_predictor',
                                 kernel_initializer=initializers.Orthogonal(
                                    gain=common_args.orth_init_gain),
                                 bias_regularizer=bias_regularizer,)
    # Connect that layer to the model graph
    Pr_a_cond_x = semantic_embed_layer(inputs)
    ############################################


    ############################################
    # Define the zero shot layer.
    # This layer that maps semantic attributes to class prediction.
    # In the paper, this layer is f^3∘f^2_{U,V}
    # i.e. The mapping A-->G-->Z, with parameters U, V
    # U is the class description, V is the (soft) group assignments
    # Its output is an estimation for Pr(z|x)

    # Define initializers for the matrices that hold the class description
    def init_train_class_description(*args):
        return class_descriptions['train']
    def init_val_class_description(*args):
        return class_descriptions['val']

    # Builds a custom LAGO layer for mapping attributes to train classes
    ZS_train_layer = semantic_transfer_Layer(categories_output_dim,
                                             init_train_class_description(),
                                             attributes_groups_ranges_ids,
                                             trainable=common_args.SG_trainable,
                                             ZS_type=common_args.model_variant,
                                             f_build=build_transfer_attributes_to_classes_LAGO,
                                             name='ZS_train_layer')

    # Builds a custom LAGO layer for mapping attributes to validation class
    ZS_val_layer = semantic_transfer_Layer(categories_output_dim,
                                           init_val_class_description(),
                                           attributes_groups_ranges_ids,
                                           trainable=False,
                                           ZS_type=common_args.model_variant,
                                           train_layer_ref=ZS_train_layer,
                                           f_build=build_transfer_attributes_to_classes_LAGO,
                                           name='ZS_val_layer')

    # Connect those layers to model graph
    ctg_train_predictions = ZS_train_layer(Pr_a_cond_x)
    ctg_val_predictions = ZS_val_layer(Pr_a_cond_x)

    # Define the prediction heads
    predictions = [ctg_train_predictions, ctg_val_predictions,
                   Pr_a_cond_x]

    # Define the Keras model
    model = LAGO_Model(inputs=inputs, outputs=predictions)

    # Add additional (object oriented) attributes to the keras model.
    # Most will be used as inputs for Keras model.compile,
    # or for the alternating optimization
    model.predictions = predictions

    # The loss heads are only over the train class predictions and attributes
    model.loss_weights = [1., 0., common_args.attributes_weight]

    # Define the monitor on validation data for early-stopping.
    # 'val_ZS_val_ctg_val_acc' is a metric name automatically generated by Keras
    # Its meaning factors to the following:
    #   'val_'          indicates using samples of the validation set
    #   'ZS_val_layer'    indicates fetching prediction from the validation head
    #                   of the model graph.
    #   'val_acc'       indicates using the accuracy metric that compares
    #                   against the validation classes
    model.monitor, model.mon_mode = 'val_ZS_val_layer_val_acc', 'max'

    # Saving pointer for the layers. It is used for accessing layer activations
    # during debugging.
    model.output_layers_dict = dict(ZS_train_layer=ZS_train_layer,
                                    ZS_val_layer=ZS_val_layer,
                                    semantic_embed_layer=semantic_embed_layer)

    # Define the loss
    def LAGO_train_loss(y_true, y_pred):
        """ Wraps the LAGO_loss according to Keras API
        """
        return LAGO_loss(y_true, y_pred,
                            semantic_embed_layer.kernel,
                            ZS_train_layer.class_descriptions.T,
                            ZS_train_layer.Gamma,
                            attributes_groups_ranges_ids)

    def zero_loss(y_true, y_pred):
        """ ZERO loss (above the validation classes head) according to Keras API.
            This makes sure that no computations are made
            through the validation classes head.
        """
        return K.constant(0.)

    model.loss_list = [LAGO_train_loss, zero_loss, LAGO_attr_loss]

    return model

def LAGO_loss(y_true, y_pred, semantic_embed_kernel, class_description_kernel,
              Gamma_tensor, attributes_groups_ranges_ids):
    """
    The LAGO loss, according to Equation 6 in paper.
    Except the BXE part of Eq. 6 which is implemented on LAGO_attr_loss()
    """

    U = K.constant(class_description_kernel)
    num_attributes = class_description_kernel.shape[0]

    # lambda hyper param
    ld = common_args.LG_lambda
    # beta hyper param
    beta = common_args.LG_beta


    # CXE at Eq. 6
    y_pred = K.cast(y_pred, 'float32')
    loss = sparse_categorical_loss(y_true, y_pred)

    # beta*||W||^2 at Eq. 6
    loss += beta * K.sum(K.square(semantic_embed_kernel))

    # lambda*||W U||^2 at Eq. 6
    loss += ld * K.sum(K.square(K.dot(semantic_embed_kernel, U)))

    # Semantic prior on Gamma
    if common_args.SG_psi > 0:
        Gamma_semantic_prior = semantic_transfer.one_hot_groups_init(
            attributes_groups_ranges_ids, num_attributes)

        loss += common_args.SG_psi * \
                K.sum(K.square(K.cast(Gamma_tensor, 'float32') - Gamma_semantic_prior))

    return loss

def LAGO_attr_loss(y_true, y_pred):
    """
    The BXE (Eq. 6) part of the LAGO loss.
    This allows to directly supervise the semantic attributes layer.
    """
    loss = K.cast(0, 'float32')
    if common_args.attributes_weight>0:
        loss += common_args.attributes_weight*keras.losses.binary_crossentropy(y_true, y_pred)
    return loss

class LAGO_Model(Model):
    """ LAGO_model is a derived class of the Keras.Model
    """
    pass


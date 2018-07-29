"""
An implementation of ESZSL (Romera-Paredes & Torr,  2015) under this framework.
Author: Yuval Atzmon
"""

import tensorflow as tf
from numpy.linalg import inv
import numpy as np

from keras import models
from keras import layers
from keras import initializers
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras import callbacks as callbacks

Model = models.Model
Input = layers.Input
Dense = layers.Dense

from zero_shot_src.semantic_transfer import semantic_transfer_Layer
from zero_shot_src import definitions


common_args, _ = definitions.get_common_commandline_args()



def build_transfer_attributes_to_classes_ESZSL(semantic_embedding,  # attributes prediction
                                               class_description,  # class descriptions
                                               debug_out_tensors,  # Tensors to query for debugging
                                               ):
    """ This builds the ESZSL output layer (ESZSL's "S" layer).
    """
    semantic_embedding = tf.to_double(semantic_embedding)
    class_description = tf.constant(class_description)


    class_prediction_scores = tf.transpose(
        tf.reduce_sum(class_description[:, :, None] * tf.transpose(semantic_embedding),
                      axis=1))

    # debug_out_tensors['class_description'] = class_description
    # debug_out_tensors['semantic_embedding'] = semantic_embedding

    return tf.to_float(class_prediction_scores)

class ESZSL_Model(Model):
    """ Overriding the Keras Model, so we can fit the model with ESZSL closed
        form solution instead of gradient steps with keras."""

    def fit(self, x, y, *args, **kwargs):
        """ Use the ES-ZSL closed form solution to fit the
            model, instead of gradient steps with keras."""

        X_train, predictions = x, y

        # Do the following closed form fit:
        # V= (X*X.T + gamma*I)^-1*X*Y*S.T*(S*S^T + lambda*I)^-1
        #     * is dot-prod.
        V_layer = self.layers[1]
        # S_train_layer = self.output_layers[3]
        S_train_layer = self.output_layers[0]

        # S = S_train_layer.get_weights()[0]
        S = S_train_layer.class_descriptions.T
        num_class = S.shape[1]
        X = X_train.T  # change to samples in columns, as in paper
        Y_sparse = predictions[0]  # Y_train

        # Converting to 1-hot. Note the paper Y is encoded to in {-1,1}
        # However in implementation, the author uses 1-hot encoded, which indeed
        # Achieves better performance
        Y = to_categorical(Y_sparse, num_class)
        gamma = common_args.ES_gamma
        ld = common_args.ES_lambda
        Ild = np.eye(S.shape[0])
        Ig = np.eye(X.shape[0])

        # Use mean SE loss instead of sum SE loss.
        if common_args.ES_use_mean:
            N = X_train.shape[0]
            Ild *= N
            Ig *= N


        # Code implementation copied from ESZSL official MATLAB repo
        """ MATLAB Code @ https://github.com/bernard24
        /Embarrassingly-simple-ZSL/blob/master/validationClasses_generalscript.m
        KYS = KTrain * Y * S;
        KYS_invSS = KYS / (S'*S+sigma*eye(size(S,2)));
        Alpha=(KK+ lambda * eye(size(KTrain)))\KYS_invSS;"""

        invSS = inv(np.dot(S, S.T) + ld * Ild)
        XYS_invSS = np.linalg.multi_dot([X, Y, S.T, invSS])
        V = np.linalg.lstsq(np.dot(X, X.T) + gamma * Ig, XYS_invSS)[0]

        sess = K.get_session()
        sess.run([tf.assign(V_layer.kernel, V)])

        # Extract filename to save the model from the callbacks
        for cb in kwargs['callbacks']:
            if isinstance(cb, callbacks.ModelCheckpoint):
                model_fullpathname = cb.filepath
                break
        else:
            if 'model_fullpathname' in kwargs:
                model_fullpathname = kwargs['model_fullpathname']
            else:
                raise ValueError("Can't save the model, "
                                 "ModelCheckpoint callback is missing")

        print('Saving the closed form fit model under %s' % model_fullpathname)
        self.save(model_fullpathname)

        return None


def get_model(input_dim, categories_output_dim, attributes_output_dim,
              class_descriptions, attributes_groups_ranges_ids):
    # Build model:
    print('Building ES-ZSL model')
    inputs = Input(shape=(input_dim,))
    h = inputs

    V_layer = Dense(attributes_output_dim, activation='linear',
                    use_bias=False,
                    # trainable=False,
                    trainable=True,
                    name='semantic_embedding_layer',
                    kernel_initializer=initializers.Orthogonal(
                        gain=common_args.orth_init_gain),
                    )

    att_predictions = V_layer(h)

    def init_S_train(*args): return class_descriptions['train']

    def init_S_val(*args): return class_descriptions['val']

    S_train_layer = semantic_transfer_Layer(categories_output_dim,
                                            init_S_train(),
                                            attributes_groups_ranges_ids,
                                            trainable=False,
                                            ZS_type = 'ESZSL',
                                            f_build=build_transfer_attributes_to_classes_ESZSL,
                                            name='ZS_train_layer')
    S_val_layer = semantic_transfer_Layer(categories_output_dim,
                                          init_S_val(),
                                          attributes_groups_ranges_ids,
                                          trainable=False,
                                          ZS_type = 'ESZSL',
                                          f_build=build_transfer_attributes_to_classes_ESZSL,
                                          name='ZS_val_layer')

    ctg_S_train_predictions = S_train_layer(att_predictions)
    ctg_S_val_predictions = S_val_layer(att_predictions)

    predictions = [ctg_S_train_predictions, ctg_S_val_predictions,
                   att_predictions]
    model = ESZSL_Model(inputs=inputs, outputs=predictions)
    model.inputs = inputs
    model.predictions = predictions

    model.output_layers = [S_train_layer, S_val_layer]
    model.output_layers_dict = dict(S_train_layer=S_train_layer,
                                    S_val_layer=S_val_layer,
                                    V_layer=V_layer)

    # An option to train ESZSL with gradient steps.
    # Not fully implemented here.
    # In an early version, we tried it. If you also like to try it, note that
    # we had to use a learning rate decay schedule in order to converge to the
    # optimum of ESZSL.
    model.loss_weights = [1., 0., 0.]
    model.monitor, model.mon_mode = 'val_ZS_val_layer_val_acc', 'max'

    def eszsl_train_loss(y_true, y_pred):
        return eszsl_loss(y_true, y_pred,
                          inputs, V_layer.kernel,
                          S_train_layer.class_descriptions.T)

    def zero_loss(y_true, y_pred):
        return K.constant(0.)

    model.loss_list = [eszsl_train_loss, zero_loss, zero_loss]

    return model


def sparse_categorical_SE(y_true, y_pred):
    """Categorical squared error with integer targets.
    it casts the integer ground truth labels to
    one-hot vectors \in {-1,1}
    # Arguments
        y_true: An integer tensor.
        y_pred: A tensor resulting from y_pred
    # Returns
        Output tensor.
    """
    output_shape = y_pred.get_shape()
    num_classes = int(output_shape[1])
    # Represent as one-hot
    y_true = K.cast(K.flatten(y_true), 'int64')
    y_true = K.one_hot(y_true, num_classes)

    if common_args.ES_use_mean:
        res = K.mean(K.square(y_true - y_pred))
    else:
        res = K.sum(K.square(y_true - y_pred))
    return res


def eszsl_loss(y_true, y_pred, inputs, V_kernel, S_kernel):
    XT = inputs
    V = V_kernel
    S = K.constant(S_kernel)
    gamma = common_args.ES_gamma
    ld = common_args.ES_lambda
    beta = gamma * ld
    if common_args.ES_use_mean:
        N = tf.to_float(tf.shape(XT)[0])
        beta = N * gamma * ld

    loss = sparse_categorical_SE(y_true, y_pred)
    loss += gamma * K.sum(K.square(K.dot(V, S)))
    loss += ld * K.sum(K.square(K.dot(XT, V)))
    loss += beta * K.sum(K.square(V))

    return loss


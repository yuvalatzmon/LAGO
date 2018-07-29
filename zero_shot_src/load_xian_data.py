"""
A python module to load zero-shot dataset of Xian (CVPR 2017).

Note that it requires altering the original directory structure of Xian.
See README.md for details.

Author: Yuval Atzmon
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd
import numpy as np
import scipy.io

from zero_shot_src import definitions

# Get values for the relevant command line arguments
common_args, unknown_args = definitions.get_common_commandline_args()



import os
from matplotlib import mlab  # MATLAB-like API with numpy


class XianDataset(definitions.ZSLData):
    def __init__(self, val_fold=0):
        self._df_class_descriptions_by_attributes = None
        self._data_dir = os.path.expanduser(common_args.data_dir)
        self._get_dir = self._init_dirs()

        data, self._att = self._load_raw_xian2017()
        self._all_X = data['features'].T
        self._all_Y = data['labels']

        self._classname_to_id = self._att['classname_to_id']

        # ( train / val ) split id
        self._val_fold = val_fold

        # With this flag we use the *test set* to judge the performance
        # Use the test set split by :
        # (1) Replacing train indices with (train bitwiseOR val) indices
        # (2) Replacing validation set indices with the test set indices
        self._use_trainval_set = common_args.use_trainval_set

    def _init_dirs(self):
        get_dir = dict(meta=os.path.join(self._data_dir, 'meta'),
                       xian_raw_data=os.path.join(self._data_dir, 'xian2017'), )
        return get_dir

    def get_data(self):
        data_dir = self._get_dir['xian_raw_data']
        val_fold_id = 1 + self._val_fold

        train_class_names = pd.read_csv(
            os.path.join(data_dir, 'trainclasses%d.txt' % val_fold_id),
            names=[]).index.tolist()
        val_class_names = pd.read_csv(
            os.path.join(data_dir, 'valclasses%d.txt' % val_fold_id),
            names=[]).index.tolist()
        test_class_names = pd.read_csv(
            os.path.join(data_dir, 'testclasses.txt'), names=[]).index.tolist()

        def class_names_to_ids(class_names_list):
            return set([self._classname_to_id[class_name] for class_name in
                        class_names_list])

        train_classes_ids = class_names_to_ids(train_class_names)
        val_classes_ids = class_names_to_ids(val_class_names)
        test_classes_ids = class_names_to_ids(test_class_names)

        #### Sanity check
        tvc_names = pd.read_csv(os.path.join(data_dir, 'trainvalclasses.txt'),
                                names=[]).index.tolist()
        assert (class_names_to_ids(tvc_names) == train_classes_ids.union(
            val_classes_ids))

        ### Sanity check end

        def get_boolean_indices(set_ids):
            return np.array(list(label in set_ids for label in self._all_Y))

        ix_train = get_boolean_indices(train_classes_ids)
        ix_val = get_boolean_indices(val_classes_ids)
        ix_test = get_boolean_indices(test_classes_ids)

        ### Sanity check: No overlap between sets
        assert (np.dot(ix_train, ix_val) == 0)
        assert (np.dot(ix_train, ix_test) == 0)
        assert (np.dot(ix_val, ix_test) == 0)

        # Sanity Check: Verify correspondence to train_loc, val_loc, ..
        if self._val_fold == 0:
            assert ((mlab.find(ix_train) + 1 == self._att['train_loc']).all())
            assert ((mlab.find(ix_val) + 1 == self._att['val_loc']).all())

        assert ((mlab.find(ix_test) + 1 == self._att['test_unseen_loc']).all())
        ### End sanity checks

        # With this flag as True, we use the *test set* to judge the performance
        # Therefore, here we do the following:
        # (1) Join the train+val sets, to be the train set
        # (2) Replace the validation set indices to be the test set indices,
        #     because the evaluations are always performed on what are set to be
        #     the validation set indices. With this setting we will run the
        #     evaluations on the test set.
        if self._use_trainval_set:
            # Replacing train indices with (train bitwiseOR val) indices
            ix_train = np.bitwise_or(ix_train, ix_val)
            # Replacing validation set indices with the test set indices
            ix_val = ix_test
            # Replacing test indices with empty set, because now we use
            ix_test = False * ix_test  #

            ### Sanity check: No overlap between sets
            assert (np.dot(ix_train, ix_val) == 0)
            assert (np.dot(ix_train, ix_test) == 0)
            assert (np.dot(ix_val, ix_test) == 0)
            ### End Sanity Check

        self._get_official_class_descriptions_by_attributes()

        X_train = self._all_X[ix_train, :]
        Y_train = np.array(self._all_Y)[ix_train]

        X_val = self._all_X[ix_val, :]
        Y_val = np.array(self._all_Y)[ix_val]

        data = dict(X_train=X_train, Y_train=Y_train,
                    X_val=X_val, Y_val=Y_val,
                    df_class_descriptions_by_attributes =
                    self._df_class_descriptions_by_attributes,
                    attributes_name=self._attributes_name,
                    )

        return data

        pass

    def _load_raw_xian2017(self):
        """
        load data as in Xian 2017 (images as ResNet101 vectors, different labels indexing of classes and unique dataset splits)
        :return:
            df_xian ?
        """
        """ 
            resNet101.mat includes the following fields:
            -features: columns correspond to image instances
            -labels: label number of a class is its row number in allclasses.txt
            -image_files: image sources  


            att_splits.mat includes the following fields:
            -att: columns correpond to class attributes vectors, following the classes order in allclasses.txt 
            -trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
            -test_seen_loc: instances indexes of test set features for seen classes
            -test_unseen_loc: instances indexes of test set features for unseen classes
            -train_loc: instances indexes of train set features (subset of trainval_loc)
            -val_loc: instances indexes of val set features (subset of trainval_loc)        
        """

        data_dir = self._get_dir['xian_raw_data']
        data, att = get_xian2017_data(data_dir)
        return data, att

    def _prepare_classes(self):
        """
        Returns a dataframe with meta-data with the following columns:
        class_id (index) | name | clean_name

        for clean_name column: removing numbers, lower case, replace '_' with ' ', remove trailing spaces
        """
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir, 'classes.txt')
        classes_df = pd.read_csv(fname, sep='[\s]',
                                 names=['class_id', 'name'],
                                 engine='python')
        self._classes_df = classes_df.set_index('class_id')

        return self._classes_df.name.tolist()

    def _load_attribute_names(self):
        # Load attribute names. Format is: id   'group_name:attribute_name'
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir,
                             'attribute_names_with_semantic_group.txt')
        df_attributes_list = \
            pd.read_csv(fname, delim_whitespace=True,
                        names=['attribute_id',
                               'attribute_name']).set_index('attribute_id')
        return df_attributes_list

    def _get_official_class_descriptions_by_attributes(self):
        data_dir = self._get_dir['meta']
        class_names_and_order = self._prepare_classes()


        # Load class descriptions
        fname = os.path.join(data_dir, 'class_descriptions_by_attributes.txt')
        df_class_descriptions_by_attributes = pd.read_csv(fname, header=None,
                                                          delim_whitespace=True,
                                                          error_bad_lines=False)

        df_attributes_list = self._load_attribute_names()

        # casting from percent to [0,1]
        df_class_descriptions_by_attributes /= 100.

        # Set its columns to attribute names
        df_class_descriptions_by_attributes.columns = \
            df_attributes_list.attribute_name.tolist()

        # Setting class id according to Xian order
        df_class_descriptions_by_attributes.index = \
            [self._classname_to_id[class_name]
             for class_name in class_names_and_order]

        # Sort according to Xian order
        df_class_descriptions_by_attributes = \
            df_class_descriptions_by_attributes.sort_index(axis=0)

        # xian_class_names_and_order = self._att['allclasses_names']
        # df_class_descriptions_by_attributes = \
        #     df_class_descriptions_by_attributes.loc[xian_class_names_and_order]


        ### Sanity check:
        # Make sure that when L2 normalizing official class results with Xian provided description

        # Extract only matrix values
        official_values = \
            df_class_descriptions_by_attributes.copy().values.T
        # L2 norm
        official_values_l2 = official_values / np.linalg.norm(official_values,
                                                              axis=0),
        # Compare to xian provided description
        check_official_class_description_l2_norm_equals_xian2017 = \
            np.isclose( official_values_l2, self._att['att'], rtol=1e-4).all()
        assert (check_official_class_description_l2_norm_equals_xian2017)
        ### End sanity check

        # If use_xian_normed_class_description=True, then replace official
        # values with Xian L2 normalized class description.
        #
        # NOTE: This is only provided to support other ZSL methods within this
        # framework. LAGO can not allow using Xian (CVPR 2017) class description,
        # because this description is a **L2 normalized** version of the mean
        # attribute values. Such normalization removes the probabilistic meaning
        # of the attribute-class description, which is a key ingredient of LAGO.
        if common_args.use_xian_normed_class_description:
            df_class_descriptions_by_attributes.iloc[:, :] = np.array(self._att['att']).T

        # Sorting class description and attributes by attribute names,
        # in order to cluster them by semantic group names.
        # (because a group name is the prefix for each attribute name)
        if common_args.sort_attr_by_names:
            df_class_descriptions_by_attributes = \
                df_class_descriptions_by_attributes.sort_index(axis=1)
        self._attributes_name = df_class_descriptions_by_attributes.columns

        self._df_class_descriptions_by_attributes = \
            df_class_descriptions_by_attributes


class AWA2_Xian(XianDataset):
    pass


class CUB_Xian(XianDataset):
    pass


class SUN_Xian(XianDataset):
    def _prepare_classes(self):
        return self._att['allclasses_names']

    def _load_attribute_names(self):
        # Load attribute names. Format is: 'group_name:attribute_name'
        data_dir = self._get_dir['meta']
        fname = os.path.join(data_dir,
                             'attribute_names_with_semantic_group.txt')
        df_attributes_list = \
            pd.read_csv(fname, delim_whitespace=True,
                        names=['attribute_name'])

        df_attributes_list.index += 1
        df_attributes_list.index.name = 'attribute_id'
        return df_attributes_list



def get_xian2017_data(data_dir):
    """
    load data as in Xian 2017 (images as ResNet101 vectors, different labels indexing of classes and unique dataset splits)
    :return:
        data, att (dictionaries)
    """

    """ From Xian2017 README:
        resNet101.mat includes the following fields:
        -features: columns correspond to image instances
        -labels: label number of a class is its row number in allclasses.txt
        -image_files: image sources


        att_splits.mat includes the following fields:
        -att: columns correpond to class attributes vectors, following the classes order in allclasses.txt
        -trainval_loc: instances indexes of train+val set features (for only seen classes) in resNet101.mat
        -test_seen_loc: instances indexes of test set features for seen classes
        -test_unseen_loc: instances indexes of test set features for unseen classes
        -train_loc: instances indexes of train set features (subset of trainval_loc)
        -val_loc: instances indexes of val set features (subset of trainval_loc)
    """
    att_file = 'att_splits.mat'
    feat_file = 'res101.mat'

    att_mat = scipy.io.loadmat(os.path.join(data_dir, att_file))
    data_mat = scipy.io.loadmat(os.path.join(data_dir, feat_file))

    data = _data_mat_to_py(data_mat)

    att = _att_mat_to_py(att_mat)

    # Add a mapping from classname to (xian) class_id
    id_to_classname = {(k + 1): v for k, v in
                       enumerate(att['allclasses_names'])}
    if 'CUB' in data_dir:
        assert (
            (np.array([id_to_classname[k] for k in data['labels']]) == np.array(
                [k.split('/')[0] for k in data['image_files']])).all())
    if 'SUN' in data_dir:
        assert (
            (np.array([id_to_classname[k] for k in data['labels']]) == np.array(
                ['_'.join(k.split('/')[1:-1]) for k in
                 data['image_files']])).all())
    att['id_to_classname'] = id_to_classname
    att['classname_to_id'] = {v: (k + 1) for k, v in
                              enumerate(att['allclasses_names'])}

    """ Properties of CUB att (for reference): 
        print [len(att[k]) for k in 
                ['train_loc', 'val_loc', 'trainval_loc', 'test_seen_loc', 'test_unseen_loc', 'att', ]]
        >>> [5875, 2946, 7057, 1764, 2967, 312]
        print 5875 + 2946
        >>> 8821
        print 7057 + 1764
        >>> 8821
        print 7057 + 1764 + 2967
        >>> 11788
        print len(set(att['train_loc']).intersection(att['trainval_loc']))
        >>> 4702
        print len(set(att['val_loc']).intersection(att['trainval_loc']))
        >>> 2355
        print 4702 + 2355
        >>> 7057      
        print len(set(att['test_seen_loc']).intersection(att['train_loc'] + att['val_loc']))
        >>> 1764
    """

    return data, att


def _att_mat_to_py(att_mat):
    att_py = {}
    att_py['allclasses_names'] = [val[0][0] for val in
                                  att_mat['allclasses_names']]
    att_py['train_loc'] = att_mat['train_loc'].astype(int).flatten().tolist()
    att_py['trainval_loc'] = att_mat['trainval_loc'].astype(
        int).flatten().tolist()
    att_py['val_loc'] = att_mat['val_loc'].astype(int).flatten().tolist()
    att_py['test_seen_loc'] = att_mat['test_seen_loc'].astype(
        int).flatten().tolist()
    att_py['test_unseen_loc'] = att_mat['test_unseen_loc'].astype(
        int).flatten().tolist()
    att_py['att'] = att_mat['att']

    return att_py


def _data_mat_to_py(data_mat):
    data_py = {}
    data_py['features'] = data_mat['features']
    if 'image_files' in data_mat:
        data_py['image_files'] = [
            fname[0][0].split('images/')[1].split('.jpg')[0] for fname in
            data_mat['image_files']]
    data_py['labels'] = data_mat['labels'].astype(int).flatten().tolist()
    return data_py



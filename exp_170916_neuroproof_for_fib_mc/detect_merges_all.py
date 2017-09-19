import os
import cPickle as pickle
import numpy as np
import vigra

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

import logging

from multicut_src import ExperimentSettings

from pipeline import find_false_merges

from init_trainsets import meta_folder, project_folder, train_ids
from init_testsets import experiment_ids

if __name__ == '__main__':

    from init_trainsets import trainset_names
    from init_testsets import testset_names, result_keys
    from run_mc_testsets import rf_cache_folder
    # from run_mc_trainsets import betas
    betas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # handler = logging.FileHandler(os.path.join(project_folder, 'test.log'))
    # handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)
    # h_stream = logging.StreamHandler()
    # h_stream.setLevel(logging.INFO)
    # h_stream.setFormatter(formatter)
    # logger.addHandler(h_stream)

    all_train_segs = []
    all_trainset_names = []
    for train_id in train_ids:
        all_trainset_names.append(trainset_names[train_id])
        trainset_name = trainset_names[train_id]

        trainset_filepath = os.path.join(project_folder, trainset_name, 'result.h5')

        all_train_segs.append(
            [trainset_filepath] * 9
        )

    all_train_keys = [['beta_{}'.format(beta) for beta in betas]] * len(train_ids)

    # # Training segmentations
    # all_train_segs = [
    #     [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
    #     [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
    #     [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
    #     [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
    #     [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
    #     [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9
    # ]
    # all_train_keys = [
    #     ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
    #     ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
    #     ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
    #     ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
    #     ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
    #     ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7']
    # ]

    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().n_threads = 40
    ExperimentSettings().n_trees = 500
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().path_features = ['path_features',
                                          'lengths',
                                          'multicuts',
                                          'cut_features',
                                          'cut_features_whole_plane']
    # ExperimentSettings().path_features = ['multicuts',
    #                                       'cut_features',
    #                                       'cut_features_whole_plane']
    # ExperimentSettings().path_features = ['lengths',
    #                                       'multicuts',
    #                                       'cut_features']
    # ExperimentSettings().path_features = ['path_features',
    #                                       'lengths']
    ExperimentSettings().use_probs_map_for_cut_features = True

    for ds_id in experiment_ids:
        ds_name = testset_names[ds_id]

        # train_segs_paths = np.delete(all_train_segs, ds_id, axis=0).tolist()
        # train_segs_keys = np.delete(all_train_keys, ds_id, axis=0).tolist()

        test_seg_path = os.path.join(project_folder, ds_name, 'result.h5')
        test_seg_key = result_keys[ds_id]

        # logger.info('Starting find_false_merges...')

        print 'ds_name = {}'.format(ds_name)
        print 'all_trainset_names = {}'.format(all_trainset_names)
        print 'meta_folder = {}'.format(meta_folder)
        print 'test_seg_path = {}'.format(test_seg_path)
        print 'test_seg_key = {}'.format(test_seg_key)
        print 'all_train_segs = {}'.format(all_train_segs)
        print 'all_train_segs.shape = {}'.format(np.array(all_train_segs).shape)
        print 'all_train_keys = {}'.format(all_train_keys)
        print 'all_train_keys.shape = {}'.format(np.array(all_train_keys).shape)

        find_false_merges(
            ds_name,
            all_trainset_names,
            meta_folder,
            test_seg_path, test_seg_key,
            all_train_segs, all_train_keys
        )


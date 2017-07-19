import os
import cPickle as pickle
import numpy as np
import vigra

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import find_false_merges

from init_datasets import meta_folder, project_folder, source_folder, result_keys, experiment_ids

if __name__ == '__main__':

    from init_datasets import ds_names
    from run_mc_all import rf_cache_folder

    # Training segmentations
    all_train_segs = [
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9
    ]
    all_train_keys = [
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7']
    ]


    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().n_threads = 30
    ExperimentSettings().n_trees = 500
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().verbose = True
    ExperimentSettings().num_it = 3000  # default = 3000
    ExperimentSettings().num_it_stop = 20  # default = 20
    # ExperimentSettings().path_features = ['path_features',
    #                                       'lengths',
    #                                       'multicuts',
    #                                       'cut_features']
    ExperimentSettings().path_features = ['multicuts']
    ExperimentSettings().use_probs_map_for_cut_features = True

    for ds_id in experiment_ids:
        ds_name = ds_names[ds_id]

        train_segs_paths = np.delete(all_train_segs, ds_id, axis=0).tolist()
        train_segs_keys = np.delete(all_train_keys, ds_id, axis=0).tolist()

        test_seg_path = os.path.join(project_folder, ds_name, 'result.h5')
        test_seg_key = result_keys[ds_id]

        # logger.info('Starting find_false_merges...')

        find_false_merges(
            ds_name,
            ds_names,
            meta_folder,
            test_seg_path, test_seg_key,
            train_segs_paths, train_segs_keys
        )
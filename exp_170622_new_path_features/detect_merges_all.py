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


    weight_filepaths = [
        os.path.join(meta_folder, ds_train_name, 'probs_to_energies_0_z_16.0_0.5_rawprobreg.h5')
        for ds_train_name in ds_names
    ]
    mc_weights_all = [
        vigra.readHDF5(train_weight_filepath, 'data')
        for train_weight_filepath in weight_filepaths
    ]

    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().n_threads = 30
    ExperimentSettings().n_trees = 500
    ExperimentSettings().rf_cache_folder = rf_cache_folder

    for ds_id in experiment_ids:
        ds_name = ds_names[ds_id]

        train_segs_paths = np.delete(all_train_segs, ds_id, axis=0).tolist()
        train_segs_keys = np.delete(all_train_keys, ds_id, axis=0).tolist()
        train_mc_weights = np.delete(mc_weights_all, ds_id, axis=0).tolist()

        test_seg_path = os.path.join(project_folder, ds_name, 'result.h5')
        test_seg_key = result_keys[ds_id]
        test_mc_weights = mc_weights_all[ds_id]

        find_false_merges(
            ds_name,
            ds_names,
            meta_folder,
            test_seg_path, test_seg_key,
            train_segs_paths, train_segs_keys,
            test_mc_weights, train_mc_weights
        )
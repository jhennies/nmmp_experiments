import os
import cPickle as pickle

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import DataSet

from multicut_src import compute_false_merges
from multicut_src import ExperimentSettings
from multicut_src import load_dataset

def find_false_merges(
        ds_test_name,
        meta_folder, rf_cache_folder,
        test_paths_cache_folder, train_paths_cache_folder,
        test_seg_path, test_seg_key,
        train_segs_paths, train_segs_keys,
        train_raw_sources, train_raw_sources_keys,
        train_probs_sources, train_probs_sources_keys,
        train_gt_sources, train_gt_sources_keys,
        ds_train_names
):

    ds_test = load_dataset(meta_folder, ds_test_name)

    trainsets = []
    for id_source, raw_source in enumerate(train_raw_sources):
        trainsets.append(
            DataSet(
                meta_folder, 'ds_train_{}'.format(ds_train_names[id_source])
            )
        )
        trainsets[-1].add_raw(raw_source, train_raw_sources_keys[id_source])
        trainsets[-1].add_input(train_probs_sources[id_source], train_probs_sources_keys[id_source])
        trainsets[-1].add_gt(train_gt_sources[id_source], train_gt_sources_keys[id_source])

    compute_false_merges(
        trainsets, ds_test,
        train_segs_paths, train_segs_keys,
        test_seg_path, test_seg_key,
        rf_cache_folder,
        test_paths_cache_folder,
        train_paths_cache_folder
    )


if __name__ == '__main__':

    from init_exp_splB_z1 import meta_folder, project_folder, source_folder, experiment_folder
    from init_exp_splB_z1 import test_name, train_name
    from run_mc_splB_z1 import rf_cache_folder

    # Load train datasets: for each source
    train_raw_sources = [
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_raw_sources_keys = [
        'z/0/raw',
        'z/1/raw',
        'z/0/raw',
        'z/0/raw',
        'z/1/raw'
    ]
    train_probs_sources = [
        source_folder + 'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_probs_sources_keys = [
        'z/0/data',
        'z/1/data',
        'z/0/data',
        'z/0/data',
        'z/1/data'
    ]
    gtruths_paths = [
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    gtruths_keys = [
        'z/0/neuron_ids',
        'z/1/neuron_ids',
        'z/0/neuron_ids',
        'z/0/neuron_ids',
        'z/1/neuron_ids'
    ]
    ds_names = [
        'splA_z0',
        'splA_z1',
        'splB_z0',
        'splC_z0',
        'splC_z1'
    ]

    # Training segmentations
    train_segs = [
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9,
        [source_folder + 'cremi.splC.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9
    ]
    train_keys = [
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7']
    ]

    # The test segmentation
    test_seg = experiment_folder + 'result.h5'
    test_seg_key = 'z/1/data'

    # Path folders
    test_paths_cache_folder = os.path.join(meta_folder, 'path_data')
    train_paths_cache_folder = os.path.join(project_folder, 'train_paths_cache')

    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().n_threads = 30
    ExperimentSettings().n_trees = 500

    find_false_merges(
        test_name, meta_folder,
        rf_cache_folder,
        test_paths_cache_folder, train_paths_cache_folder,
        test_seg, test_seg_key,
        train_segs, train_keys,
        train_raw_sources, train_raw_sources_keys,
        train_probs_sources, train_probs_sources_keys,
        gtruths_paths, gtruths_keys,
        ds_names
    )
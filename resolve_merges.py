
from multicut_src import compute_false_merges, ComputeFalseMergesParams
from multicut_src import MetaSet
from multicut_src import DataSet
# from multicut_src import PipelineParameters
from multicut_src import ExperimentSettings
# from find_false_merges_src import load_false_merges
# from find_false_merges_src import FeatureImages
# from find_false_merges_src import FeatureImageParams, SegFeatureImageParams
# from find_false_merges_src import PathFeatureParams, PathsParams, ClassifierParams

import numpy as np
import vigra
import os

# Load a multicut segmentation (test set)
mc_source_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
mc_source_file = 'cremi.splB.train.mcseg_betas.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
mc_source_hdf5path = 'z/1/beta_0.5'
mc_seg_test = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/result.h5'
mc_seg_test_key = 'z/1/test'

# Load a previously initialized test dataset
cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/'
meta = MetaSet(cache_folder)
meta.load()
ds_test = meta.get_dataset('ds_test')

# Multicut segmentation paths (training data for false merge candidate detection)
mc_segs_train_paths = [
    ['/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.train.mcseg_betas.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'] * 2
]
mc_segs_train_keys = [
    ['z/0/beta_0.5', 'z/0/beta_0.45'] #, 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7']
] * len(mc_segs_train_paths)

gtruths_paths = [
    '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
]
gtruths_keys = [
    'z/0/neuron_ids'
]

# TODO: Load train datasets: for each source
train_raw_sources = [
    '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
]
train_raw_sources_keys = [
    'z/0/raw'
]
train_probs_sources = [
    '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/cremi.splB.train.probs.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
]
train_probs_sources_keys = [
    'z/0/data'
]
trainsets = []
for id_source, raw_source in enumerate(train_raw_sources):
    trainsets.append(
        DataSet(
            cache_folder, 'ds_train_{}'.format(id_source)
        )
    )
    trainsets[-1].add_raw(raw_source, train_raw_sources_keys[id_source])
    trainsets[-1].add_input(train_probs_sources[id_source], train_probs_sources_keys[id_source])

# TODO: Add ground into dataset

# Merge candidate detection parameters
from false_merges import RemoveSmallObjectsParams
# FIXME: This parameter stucture doesn't work: Use simpler structure without sub-class
params = ComputeFalseMergesParams(remove_small_objects=RemoveSmallObjectsParams(max_threads=20))

# TODO: Compute the false merge
false_merge_ids, false_paths, path_classifier = compute_false_merges(
    trainsets,  # list of datasets with training data
    ds_test,  # one dataset -> predict the false merged objects
    mc_segs_train_paths,  # list with paths to segmentations (len(mc_segs_train) == len(trainsets))
    mc_segs_train_keys,
    mc_seg_test,
    mc_seg_test_key,
    gtruths_paths,
    gtruths_keys,
    params=ComputeFalseMergesParams()
)

# # TODO: Or load false merges
# false_merge_ids, false_paths, path_classifier = load_false_merges()
# TODO: Development: Some dummy ids
# import random
mc_seg_labels = np.unique(mc_seg)
# false_merge_ids = random.sample(mc_seg_labels, 10)
false_merge_ids = mc_seg_labels[20:30]

# TODO: Get feature images
feature_image_path = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/find_merges_cache/intermed/'
feature_image_files = [
    'test_segfeats.h5',
    'rawfeats.h5',
    'probsfeats.h5'
]
source_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
source_files = [
    'cremi.splB.train.mcseg_betas.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5',
    'cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5',
    'cremi.splB.train.probs.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
]
feature_images = {
    'segmentation': FeatureImages(
        filepath=feature_image_path + feature_image_files[0],
        source_filepath='/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/result.h5',
        source_internal_path='z/1/test/',
        internal_path='z/1/beta_0.5/',
        params=SegFeatureImageParams()
    ),
    'rawdata': FeatureImages(
        filepath=feature_image_path + feature_image_files[1],
        source_filepath=source_path + source_files[1],
        source_internal_path='z/1/raw',
        internal_path='z/1/',
        params=FeatureImageParams()
    ),
    'probs': FeatureImages(
        filepath=feature_image_path + feature_image_files[2],
        source_filepath=source_path + source_files[2],
        source_internal_path='z/1/data',
        internal_path='z/1/',
        params=FeatureImageParams()
    )
}

# TODO: Get edge probabilies of previous multicut
# TODO: Am I supposed to load them from here?
edge_probs = vigra.readHDF5("/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/rf_cache/pred_ds_train_0/prediction_ds_train_0_ds_test_0_('raw', 'prob', 'reg', 'topo')_10.0_False_False_500_0.4_0.6_False_True", "data")

# TODO: experiment parameters
mc_params = ExperimentSettings()
mc_params.set_rfcache(os.path.join(cache_folder, "resolve_merges_rf_cache"))
mc_params.set_ntrees(500)
mc_params.set_anisotropy(10.)
mc_params.set_use2d(False)
mc_params.set_nthreads(20)

# Path feature parameters
pf_params = PathFeatureParams(
    feat_list_file=feature_image_path + 'featlist.pkl',
    experiment_key='z_train0_predict1',
    max_threads=32
)

# Path computation parameters
path_params = PathsParams()
# Random forest parameters
classifier_params = ClassifierParams(
    classifier_file='/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/find_merges_cache/randomforest.pkl',
    classifier_key='z_train0_predict1'
)

# TODO: resolve false merges using a second Multi-cut
resolve_merges_with_lifted_edges(
    ds_test, false_merge_ids, false_paths, path_classifier,
    feature_images, mc_seg_test, edge_probs, mc_params,
    pf_params, path_params, classifier_params
)

# # TODO: Load the initial Multi-cut computed in run_mc
# cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/'
# meta = MetaSet(cache_folder)
#
# meta.load()
#
# ds_train = meta.get_dataset('ds_train')
# ds_test = meta.get_dataset('ds_test')
#
# mc_seg_test = vigra.readHDF5(cache_folder + 'result.h5', 'z/1/test')
# mc_seg_train = vigra.readHDF5(cache_folder + 'result.h5', 'z/0/train')
#
# # Set parameters
# params = PipelineParameters()
#
# # # TODO: Compute the false merges
# # false_merge_ids, false_paths, path_classifier = compute_false_merges(
# #     ds_train, ds_test,
# #     mc_seg_train, mc_seg_test,
# #     params
# # )
#
# # TODO: Or load false merges
# false_merge_ids, false_paths, path_classifier = load_false_merges()
#
# # TODO:
#
# # Some dummy ids:
# import random
# mc_seg_labels = np.unique(mc_seg)
# false_merge_ids = random.sample(mc_seg_labels, 10)
#
# false_paths = []
#
# # TODO: resolve false merges using a second Multi-cut
# resolve_merges_with_lifted_edges(ds_test, false_merge_ids, false_paths)


import os
import vigra
import numpy as np
import cPickle as pickle

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

# hack to get in meta
# import sys
# sys.path.append('..')
# from init_exp import meta
from multicut_src import MetaSet
from multicut_src import DataSet

# from multicut_src import shortest_paths, path_feature_aggregator
from multicut_src import compute_false_merges, resolve_merges_with_lifted_edges
from multicut_src import project_resolved_objects_to_segmentation
from multicut_src import ExperimentSettings


cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_test_pipeline_update/cache/'
source_folder = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'


def find_false_merges(ds_str):

    meta = MetaSet(cache_folder)
    meta.load()
    ds = meta.get_dataset(ds_str)

    # Load train datasets: for each source
    train_raw_sources = [
        source_folder + 'cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
    ]
    train_raw_sources_keys = [
        'z/0/raw'
    ]
    train_probs_sources = [
        source_folder + 'cremi.splB.train.probs.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
    ]
    train_probs_sources_keys = [
        'z/0/data'
    ]
    gtruths_paths = [
        source_folder + 'cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
    ]
    gtruths_keys = [
        'z/0/neuron_ids'
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
        trainsets[-1].add_gt(gtruths_paths[id_source], gtruths_keys[id_source])

    train_segs = [
        [source_folder + 'cremi.splB.train.mcseg_betas.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'] * 9
    ]
    test_seg = cache_folder + '../result.h5'

    train_keys = [
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7']
    ] * 1
    test_key = 'z/1/test'
    rf_save_folder = cache_folder + 'rf_cache/path_rfs'

    paths_save_folder = cache_folder + 'path_data/'

    params = ExperimentSettings()
    params.set_anisotropy(10.)

    paths, false_merge_probs, path_to_objs = compute_false_merges(
        trainsets,
        ds,
        train_segs,
        train_keys,
        test_seg,
        test_key,
        rf_save_folder,
        paths_save_folder=paths_save_folder,
        params=params
    )
    print "Succesfully computed false mereg probabilities for", len(false_merge_probs), "paths"
    # import os
    # os.mkdir('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/')
    # with open('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/paths.pkl', 'w') as f:
    #     pickle.dump(paths, f)
    with open(cache_folder + 'path_data/false_paths_predictions.pkl', 'w') as f:
        pickle.dump(false_merge_probs, f)
    # with open('/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/path_data/paths_to_objs.pkl', 'w') as f:
    #     pickle.dump(path_to_objs, f)


def resolve_false_merges(exp_params):
    meta = MetaSet(cache_folder)
    meta.load()
    ds = meta.get_dataset('splB_z1')
    with open(cache_folder + 'path_data/path_splB_z1.pkl') as f:
        path_data = pickle.load(f)
    paths = path_data['paths']
    paths_to_objs = path_data['paths_to_objs']
    with open(cache_folder + '/path_data/false_paths_predictions.pkl') as f:
        false_merge_probs = pickle.load(f)

    # unmerge the 2 objs with highest false merge probability paths
    sorted_indices = np.argsort(false_merge_probs)[::-1]
    false_merge_ids = [paths_to_objs[sorted_indices[0]]]
    false_paths     = {false_merge_ids[0] : [paths[sorted_indices[0]]] }
    sorted_id = 1
    while false_merge_ids[-1] == false_merge_ids[0]:
        next_index = sorted_indices[sorted_id]
        next_id = paths_to_objs[next_index]
        false_merge_ids.append(next_id)
        if next_id in false_paths:
            false_paths[next_id].append(paths[next_index])
        else:
            false_paths[next_id] = [paths[next_index]]
        sorted_id += 1

    rf_path = cache_folder + 'rf_cache/path_rfs/rf_merges_ds_train_0.pkl'
    test_seg = cache_folder + '../result.h5'
    # mc_weights_path = '/home/constantin/Work/home_hdd/cache/cremi/sample_A_train/probs_to_energies_0_-8166828587302537792.h5'

    mc_seg = vigra.readHDF5(test_seg, 'z/1/test')
    # TODO change here
    mc_weights = vigra.readHDF5(cache_folder + "splB_z1/probs_to_energies_0_-110420280210600738_rawprobregtopo.h5", "data")

    with open(rf_path) as f:
        path_rf = pickle.load(f)

    export_paths_path = cache_folder + 'path_data/'

    new_node_labels = resolve_merges_with_lifted_edges(
        ds, 0,
        false_paths, path_rf,
        mc_seg, mc_weights,
        exp_params,
        export_paths_path=export_paths_path
    )
    with open(cache_folder + 'path_data/new_noes.pkl', 'w') as f:
        pickle.dump(new_node_labels, f)


def project_new_segmentation():
    meta = MetaSet(cache_folder)
    meta.load()
    ds = meta.get_dataset('splB_z1')
    test_seg = cache_folder + '../result.h5'
    mc_seg = vigra.readHDF5(test_seg, 'z/1/test')
    with open(cache_folder + 'path_data/new_noes.pkl') as f:
        new_node_labels = pickle.load(f)
    new_seg = project_resolved_objects_to_segmentation(
            ds, 0,
            mc_seg, new_node_labels)
    print new_seg.shape
    vigra.writeHDF5(
        new_seg,
        cache_folder + '../result_resolved.h5',
        'z/1/test'
    )


def resolve_false_merges_threshold_test_settings(mc_params, meta):

    test_set_name = 'splB_z1'
    weight_file_name = 'probs_to_energies_0_-110420280210600738_rawprobregtopo.h5'

    from evaluation import resolve_merges_threshold_test
    resolve_merges_threshold_test(
        meta, test_set_name,
        mc_params, cache_folder,
        weight_file_name,
        min_prob_thresh=0.5
    )


if __name__ == '__main__':

    # # 1.) find false merge objects
    # find_false_merges('splB_z1')

    # 2.) resolve the objs classified as false merges
    # parameters for the Multicut
    meta = MetaSet(cache_folder)
    mc_params = ExperimentSettings()
    rfcache = os.path.join(meta.meta_folder, "rf_cache")
    mc_params.set_rfcache(rfcache)

    mc_params.set_anisotropy(10.)
    mc_params.set_use2d(False)

    mc_params.set_nthreads(30)

    mc_params.set_ntrees(500)

    mc_params.set_solver("multicut_fusionmoves")
    # mc_params.set_verbose(True)
    mc_params.set_weighting_scheme("z")

    mc_params.set_lifted_neighborhood(3)

    mc_params.min_nh_range = 5
    # mc_params.max_sample_size = 20
    mc_params.max_sample_size = 10

    # resolve_false_merges(mc_params)
    resolve_false_merges_threshold_test_settings(mc_params, meta)

    # # 3.) project the resolved result to segmentation
    # project_new_segmentation()

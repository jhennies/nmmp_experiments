
import numpy as np
import os
import cPickle as pickle
import vigra

from multicut_src import ExperimentSettings, load_dataset
from multicut_src import resolve_merges_with_lifted_edges_global, resolve_merges_with_lifted_edges

def resolve_false_merges_global(
        ds_name, meta_folder, paths_cache_folder,
        pre_seg_filepath, pre_seg_key,
        weight_filepath, lifted_filepath,
        rf_cache_name,
        min_prob_thresh, max_prob_thresh,
        exclude_objs_with_larger_thresh
):

    ds = load_dataset(meta_folder, ds_name)
    seg_id = 0

    path_data_filepath = os.path.join(paths_cache_folder, 'paths_ds_{}.h5'.format(ds_name))
    # with open(os.path.join(paths_cache_folder, 'paths_ds_{}.pkl'.format(ds_name))) as f:
    #     path_data = pickle.load(f)
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(paths_cache_folder, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    # Find objects where probability >= min_prob_thresh and <= max_prob_thresh
    objs_with_prob_greater_thresh = np.unique(
        np.array(paths_to_objs)[
            np.logical_and(
                false_merge_probs >= min_prob_thresh,
                false_merge_probs <= max_prob_thresh
            )
        ]
    )
    if exclude_objs_with_larger_thresh:
        objs_to_exclude = np.unique(
            np.array(paths_to_objs)[
                false_merge_probs > max_prob_thresh
            ]
        )
        objs_with_prob_greater_thresh = np.setdiff1d(objs_with_prob_greater_thresh, objs_to_exclude)

    # Extract all paths for each of the found objects
    false_paths = {}
    for obj in objs_with_prob_greater_thresh:
        # print paths_to_objs == obj
        false_paths[obj] = np.array(paths)[paths_to_objs == obj]

    rf_filepath = os.path.join(rf_cache_folder, rf_cache_name, 'rf.pkl')
    with open(rf_filepath) as f:
        path_rf = pickle.load(f)

    mc_segmentation = vigra.readHDF5(pre_seg_filepath, pre_seg_key)
    mc_weights_all = vigra.readHDF5(weight_filepath, "data")
    lifted_weights_all = vigra.readHDF5(lifted_filepath, "data")

    resolve_merges_with_lifted_edges(
        ds, seg_id,
        false_paths,
        path_rf,
        mc_segmentation,
        mc_weights_all,
        paths_cache_folder=paths_cache_folder,
        lifted_weights_all=lifted_weights_all
    )

if __name__ == '__main__':

    from init_exp_splB_z1 import meta_folder, experiment_folder
    from init_exp_splB_z1 import test_name
    from run_mc_splB_z1 import rf_cache_folder
    from detect_merges_splB_z1 import test_paths_cache_folder

    # These are the parameters as used for the initial mc
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = 30
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    # Parameters for the resolving algorithm
    ExperimentSettings().min_nh_range = 5
    ExperimentSettings().max_sample_size = 20

    rf_cache_name = 'rf_merges_ds_train_splA_z0_ds_train_splA_z1_ds_train_splB_z0_ds_train_splC_z0_ds_train_splC_z1/'
    min_prob_thresh = 0.3
    max_prob_thresh = 1.
    exclude_objs_with_larger_thresh = False
    pre_seg_filepath = os.path.join(experiment_folder, 'result.h5')
    pre_seg_key = 'z/1/data'
    weight_filepath = os.path.join(meta_folder, test_name,
                                   'probs_to_energies_0_z_16.0_0.5_rawprobreg.h5')
    lifted_filepath = os.path.join(meta_folder, test_name,
                                   'lifted_probs_to_energies_0_3_0.5_2.0.h5')

    resolve_false_merges_global(
        test_name, meta_folder, test_paths_cache_folder,
        pre_seg_filepath, pre_seg_key,
        weight_filepath, lifted_filepath,
        rf_cache_name,
        min_prob_thresh, max_prob_thresh,
        exclude_objs_with_larger_thresh
    )
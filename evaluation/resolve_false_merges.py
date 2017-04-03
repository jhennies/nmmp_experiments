
from multicut_src import MetaSet
from multicut_src import resolve_merges_with_lifted_edges
from multicut_src import project_resolved_objects_to_segmentation
import pickle
import vigra
import numpy as np


def resolve_merges(
        ds, seg_id,
        false_paths,
        path_rf,
        pre_seg_path, pre_seg_key,
        weight_file_path,
        exp_params, cache_folder
):
    # rf_path = cache_folder + 'rf_cache/path_rfs/rf_merges_ds_train_0.pkl'
    # test_seg = cache_folder + '../result.h5'
    # mc_weights_path = '/home/constantin/Work/home_hdd/cache/cremi/sample_A_train/probs_to_energies_0_-8166828587302537792.h5'

    mc_seg = vigra.readHDF5(pre_seg_path, pre_seg_key)
    mc_weights = vigra.readHDF5(weight_file_path, "data")

    # with open(rf_path) as f:
    #     path_rf = pickle.load(f)

    export_paths_path = cache_folder + 'path_data/'

    new_node_labels = resolve_merges_with_lifted_edges(
        ds, seg_id,
        false_paths, path_rf,
        mc_seg, mc_weights,
        exp_params,
        export_paths_path=export_paths_path
    )
    with open(cache_folder + 'path_data/new_noes.pkl', 'w') as f:
        pickle.dump(new_node_labels, f)


def project_new_segmentation(
        ds, seg_id,
        cache_folder,
        pre_seg_path, pre_seg_key
):

    # Load previous segmentation
    mc_seg = vigra.readHDF5(pre_seg_path, pre_seg_key)

    # Load resolving result
    with open(cache_folder + 'path_data/new_noes.pkl') as f:
        new_node_labels = pickle.load(f)

    new_seg = project_resolved_objects_to_segmentation(
            ds, seg_id,
            mc_seg, new_node_labels)

    return new_seg


def resolve_merges_threshold_test(
        meta, test_set_name,
        exp_params, cache_folder,
        weight_file_name,
        min_prob_thresh=0.5
):

    # TODO: First resolve all those objects that have probability >= min_prob_thresh

    seg_id = 0
    meta.load()
    ds = meta.get_dataset(test_set_name)
    with open(cache_folder + 'path_data/path_{}.pkl'.format(test_set_name)) as f:
        path_data = pickle.load(f)
    paths = path_data['paths']
    paths_to_objs = path_data['paths_to_objs']
    with open(cache_folder + '/path_data/false_paths_predictions.pkl') as f:
        false_merge_probs = pickle.load(f)

    # Find objects where probability >= min_prob_thresh
    objs_with_prob_greater_thresh = np.unique(
        np.array(paths_to_objs)[false_merge_probs >= min_prob_thresh]
    )
    # Extract all paths for each of the found objects
    false_paths = {}
    for obj in objs_with_prob_greater_thresh:
        # print paths_to_objs == obj
        false_paths[obj] = np.array(paths)[paths_to_objs == obj]

    # Get the random forest classifier
    rf_path = cache_folder + 'rf_cache/path_rfs/rf_merges_ds_train_0_ds_train_1_ds_train_2_ds_train_3_ds_train_4.pkl'
    # TODO: Generate name automatically
    with open(rf_path) as f:
        path_rf = pickle.load(f)

    # Set the source of the previous segmentation
    pre_seg_path = cache_folder + '../result.h5'
    pre_seg_key = 'z/1/test'

    # Set the weight file path
    weight_file_path = cache_folder + '/' + test_set_name + '/' + weight_file_name

    resolve_merges(
        ds, seg_id,
        false_paths,
        path_rf,
        pre_seg_path, pre_seg_key,
        weight_file_path,
        exp_params, cache_folder
    )

    # TODO: Project resolved result to segmentation
    # TODO: Merge small segments (Make sure the small segments are merged to the object we corrected)
    new_seg = project_new_segmentation(
        ds, seg_id,
        cache_folder,
        pre_seg_path, pre_seg_key
    )

    # Write new result
    print new_seg.shape
    vigra.writeHDF5(
        new_seg,
        cache_folder + '../result_resolved_thresh_{}.h5'.format(min_prob_thresh),
        'z/1/test'
    )

    # TODO: Loop over possible thresholds and create respective segmentation
    # Take the objects with prob > thresh from resolved_seg and the others from original_seg
    # FIXME a little dirty with the 1.01, but this makes sure the 1 is in the list
    thresh_range = np.arange(min_prob_thresh + 0.1, 1.01, 0.1)

    old_seg = vigra.readHDF5(pre_seg_path, pre_seg_key)

    from copy import deepcopy

    for thresh in thresh_range:

        print 'thresh = {}'.format(thresh)

        new_seg_thresh = deepcopy(new_seg)

        for obj in objs_with_prob_greater_thresh:

            if max(false_merge_probs[np.array(paths_to_objs) == obj]) < thresh:

                # Restore the original segmentation by using an appended label
                new_seg_thresh[old_seg == obj] = np.amax(new_seg_thresh) + 1

        new_seg_thresh, _, _ = vigra.analysis.relabelConsecutive(new_seg_thresh, keep_zeros=0)

        # Write new result
        vigra.writeHDF5(
            new_seg_thresh,
            cache_folder + '../result_resolved_thresh_{}.h5'.format(thresh),
            'z/1/test'
        )


if __name__ == '__main__':
    pass

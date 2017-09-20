
import os

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings, load_dataset

from pipeline import resolve_false_merges, project_new_result, project_resolved_objects_to_segmentation

if __name__ == '__main__':

    from init_trainsets import meta_folder, project_folder, trainset_names, train_ids
    from init_testsets import testset_names, result_keys, experiment_ids
    from run_mc_trainsets import rf_cache_folder

    # These are the parameters as used for the initial mc
    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().use_2d = False
    ExperimentSettings().use_2rfs = False
    ExperimentSettings().n_threads = 40
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'all'
    ExperimentSettings().lifted_neighborhood = 3

    # Parameters for the resolving algorithm
    ExperimentSettings().min_nh_range = 5
    ExperimentSettings().max_sample_size = 20
    ExperimentSettings().stacked_eccentricity_centers = False
    ExperimentSettings().path_features = ['path_features',
                                          'lengths',
                                          'multicuts',
                                          'cut_features',
                                          'cut_features_whole_plane']

    # Parameters deciding which objects to resolve
    min_prob_thresh = 0.3
    max_prob_thresh = 1.
    exclude_objs_with_larger_thresh = False

    all_trainset_names = []
    for train_id in train_ids:
        all_trainset_names.append(trainset_names[train_id])

    for ds_id in experiment_ids:
        ds_name = testset_names[ds_id]

        result_key = result_keys[ds_id]

        test_seg_path = os.path.join(project_folder, ds_name, 'result.h5')
        test_seg_key = result_keys[ds_id]

        # Local resolving ---------------------
        # TODO: Change here when adding result
        new_nodes_filepath = os.path.join(meta_folder, ds_name, 'new_ones_local_t{}.pkl'.format(min_prob_thresh))
        # TODO: Change here when adding result
        result_filepath = os.path.join(project_folder, ds_name, 'result_resolved_local_t{}.h5'.format(min_prob_thresh))

        print 'ds_name = {}'.format(ds_name)
        print 'all_trainset_names = {}'.format(all_trainset_names)
        print 'meta_folder = {}'.format(meta_folder)
        print 'test_seg_path = {}'.format(test_seg_path)
        print 'test_seg_key = {}'.format(test_seg_key)
        print 'new_nodes_filepath = {}'.format(new_nodes_filepath)
        print 'result_filepath = {}'.format(result_filepath)

        if not os.path.isfile(result_filepath):

            resolve_false_merges(
                ds_name, all_trainset_names,
                meta_folder, rf_cache_folder,
                new_nodes_filepath,
                test_seg_path, test_seg_key,
                min_prob_thresh, max_prob_thresh,
                exclude_objs_with_larger_thresh,
                global_resolve=False
            )

            project_resolved_objects_to_segmentation(
                meta_folder, ds_name,
                test_seg_path, test_seg_key,
                new_nodes_filepath,
                result_filepath, test_seg_key
            )

        else:

            print 'Nothing to do'

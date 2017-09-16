
import vigra
import cPickle as pickle
import os
import re
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from multicut_src import load_dataset

import sys
sys.path.append('/home/jhennies/src/cremi_python/cremi_python')
from cremi import Volume, NeuronIds


def compute_path_error_rates(
        paths_to_objs, paths,
        gt,
        false_merge_probs,
        thresh_range=[0.5]
):

    result_obj = {}
    result_path = {}

    for thresh in thresh_range:

        print 'THRESHOLD = {}'.format(thresh)
        # Get the predicted classes
        predicted = false_merge_probs >= thresh

        # Get the real classes
        reference = np.array([(gt[tuple(p[0])], gt[tuple(p[-1])]) for p in paths])
        reference = np.logical_not(np.equal(reference[:, 0], reference[:, 1]))

        # Turn the predicted and reference path classes to an object-wise classification,
        # i.e., determine prediction on an object level
        objs = np.unique(paths_to_objs)

        # Find objects where probability >= min_prob_thresh
        objs_with_prob_greater_thresh = np.unique(
            np.array(paths_to_objs)[predicted]
        )
        predicted_obj = np.zeros(objs.shape, dtype=np.bool)
        indices = np.nonzero(np.in1d(objs, objs_with_prob_greater_thresh))[0]
        predicted_obj[indices] = True

        # Determine reference classes on an object level
        objs_with_merged_path = np.unique(
            np.array(paths_to_objs)[reference]
        )
        reference_obj = np.zeros(objs.shape, dtype=np.bool)
        indices = np.nonzero(np.in1d(objs, objs_with_merged_path))[0]
        reference_obj[indices] = True

        from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

        def new_eval(result, truth):
            return {
                'precision': precision_score(truth, result, average='binary', pos_label=1),
                'recall': recall_score(truth, result, average='binary', pos_label=1),
                'f1': f1_score(truth, result, average='binary', pos_label=1),
                'accuracy': accuracy_score(truth, result)
            }

        # TODO: Also evaluate path-wise?
        # EVALUATION: Object level

        try:
            ref_class, ref_class_counts = np.unique(reference, return_counts=True)
            pred_class, pred_class_counts = np.unique(predicted, return_counts=True)
            print '    Paths (predicted | truth):'
            for cl_id, cl in enumerate(ref_class):
                print '        {}: {} | {}'.format(cl, pred_class_counts[cl_id], ref_class_counts[cl_id])

            ref_class_obj, ref_class_counts_obj = np.unique(reference_obj, return_counts=True)
            pred_class_obj, pred_class_counts_obj = np.unique(predicted_obj, return_counts=True)
            print '    Objects (predicted | truth):'
            for cl_id, cl in enumerate(ref_class_obj):
                print '        {}: {} | {}'.format(cl, pred_class_counts_obj[cl_id], ref_class_counts_obj[cl_id])
        except:
            pass

        result_obj[thresh] = new_eval(predicted_obj, reference_obj)
        result_path[thresh] = new_eval(predicted, reference)

    return result_path, result_obj


def path_eval_on_ds(
        ds_name,
        thresh_range,
        project_folder,
        run_name=''
):

    # from evaluation import compute_path_error_rates

    print '\nEvaluating {}'.format(ds_name)
    print '--------------------'

    meta_folder = os.path.join(project_folder, 'cache/')

    path_data_path = os.path.join(meta_folder, ds_name, 'path_data')

    path_data_filepath = os.path.join(path_data_path, 'paths_ds_{}.h5'.format(ds_name))

    # Load dataset
    ds = load_dataset(meta_folder, ds_name)

    gt = ds.gt()

    # Load paths
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(path_data_path, run_name, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    print 'Number of paths = {}'.format(len(paths_to_objs))
    print 'Number of objects = {}'.format(len(np.unique(paths_to_objs)))

    # Determine path error rates
    result_path, result_obj = compute_path_error_rates(
        paths_to_objs, paths, gt, false_merge_probs, thresh_range=thresh_range
    )

    return result_path, result_obj


def all_ds_path_eval(
        project_folder,
        thresh_range,
        ds_names,
        measures=None,
        run_name=''):

    if measures is None:
        measures = ['precision', 'recall', 'accuracy', 'f1']

    results_path = {}
    results_obj = {}
    for measure in measures:
        results_path[measure] = []
        results_obj[measure] = []

    for idx, ds_name in enumerate(ds_names):
        result_path, result_obj = path_eval_on_ds(
            ds_name,
            thresh_range,
            project_folder,
            run_name=run_name
        )

        sorted_keys = sorted(result_path.keys())
        for key in results_path.keys():
            results_path[key].append(np.array([result_path[k][key] for k in sorted_keys])[:, None])
            results_obj[key].append(np.array([result_obj[k][key] for k in sorted_keys])[:, None])

    return results_path, results_obj


def roi_and_rand_general(
        ds_name,
        project_folder,
        result_file,
        result_key,
        caching=False,
        debug=False,
        gt=None,
        compute_rand=True
):
    print '\nEvaluating {}'.format(ds_name)
    print 'Result file: {}'.format(result_file)

    experiment_folder = os.path.join(project_folder, ds_name)
    meta_folder = os.path.join(project_folder, 'cache')

    if caching:
        cache_filepath = os.path.join(
            experiment_folder,
            re.sub('.h5$', '', result_file) + '_roi_and_rand_cache.pkl'
        )
    else:
        cache_filepath = None

    if caching and os.path.isfile(cache_filepath):
        with open(cache_filepath, mode='r') as f:
            voi_split, voi_merge, adapted_rand = pickle.load(f)

    else:

        if gt is not None:
            pass

        else:
            # Load dataset
            ds = load_dataset(meta_folder, ds_name)

            if not debug:
                gt = ds.gt()

        print 'gt.shape = {}'.format(gt.shape)

        if not debug:
            vol_gt = Volume(gt)
            neuron_ids_evaluation = NeuronIds(vol_gt)

        mc_result_filepath = os.path.join(experiment_folder, result_file)

        if not debug:
            # Evaluate baseline
            mc_result = vigra.readHDF5(mc_result_filepath, result_key)
            print 'mc_result.shape = {}'.format(mc_result.shape)
            vol_mc_result = Volume(mc_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_mc_result)
            if compute_rand:
                adapted_rand = neuron_ids_evaluation.adapted_rand(vol_mc_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            if compute_rand:
                adapted_rand = 0.23

        if caching:
            with open(cache_filepath, mode='w') as f:
                if compute_rand:
                    pickle.dump((voi_split, voi_merge, adapted_rand), f)
                else:
                    pickle.dump((voi_split, voi_merge), f)

    print "\tvoi split   : " + str(voi_split)
    print "\tvoi merge   : " + str(voi_merge)
    if compute_rand:
        print "\tadapted RAND: " + str(adapted_rand)
    else:
        adapted_rand = 0

    return voi_split, voi_merge, adapted_rand


if __name__ == '__main__':

    pass
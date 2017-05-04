import sys
sys.path.append('..')
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from evaluation import resolve_merges_error_rate_path_level
from evaluation import compute_path_error_rates

import cPickle as pickle
import numpy as np
import vigra
import os

# TODO: Change here
from init_exp_splB_z1 import meta_folder, source_folder, experiment_folder
from init_exp_splB_z1 import test_name
from run_mc_splB_z1 import result_key

path_data_path = os.path.join(meta_folder, 'path_data')
path_data_filepath = os.path.join(path_data_path, 'paths_ds_{}.h5'.format(test_name))
seg_file = os.path.join(experiment_folder, 'result.h5')
seg_key = result_key
# TODO Change here when switching scope
resolved_file = os.path.join(experiment_folder, 'result_resolved_local.h5')
resolved_key = result_key
# TODO Change here when switching sample
gt_file = os.path.join(source_folder, 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5')
# TODO Change here when switching half
gt_key = 'z/1/neuron_ids'

# thresh_range = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
thresh_range = [0.3]

resolved_only=False

# Load paths
paths = vigra.readHDF5(path_data_filepath, 'all_paths')
if paths.size:
    paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
with open(os.path.join(path_data_path, 'false_paths_predictions.pkl')) as f:
    false_merge_probs = pickle.load(f)

# Load images
resolved = vigra.readHDF5(resolved_file, resolved_key)
gt = vigra.readHDF5(gt_file, gt_key)
seg = vigra.readHDF5(seg_file, seg_key)

# Determine merge error rate (path level)
errors_seg, errors_rsd, errors_to_obj = resolve_merges_error_rate_path_level(
    paths, paths_to_objs, resolved, gt, seg,
    false_merge_probs,
    thresh_range=thresh_range,
    resolved_only=resolved_only
)

# Determine path error rates
result_path, result_obj = compute_path_error_rates(
    paths_to_objs, paths, gt, false_merge_probs, thresh_range=thresh_range
)

for thresh in thresh_range:

    all_errors_rsd = np.concatenate(errors_rsd[thresh])
    all_errors_seg = np.concatenate(errors_seg[thresh])

    # Like this we can find the erroneous and correct pairs with information on
    #   the error type:
    #   3: gt == True, rs == True -> correct merge
    #   2: gt == True, rs == False -> false split
    #   1: gt == False, rs == True -> false merge
    #   0: gt == False, rs == False -> correct split
    # The original performance with respect to merged obj is in gt_equal:
    #   True -> correct merge
    #   False -> false merge

    error_types, error_counts = np.unique(all_errors_rsd, return_counts=True)
    orig_error_types, orig_error_counts = np.unique(all_errors_seg, return_counts=True)

    obj_mask_with_all_true_seg = np.array([x.all() for x in errors_seg[thresh]])
    obj_mask_with_all_true_rsd = np.array([(x != 1).all() for x in errors_rsd[thresh]])

    false_merge_seg = sum(np.logical_not(obj_mask_with_all_true_seg))
    false_merge_rsd = sum(np.logical_not(obj_mask_with_all_true_rsd))

    errors_seg_with_all_true = np.array(errors_seg[thresh])[obj_mask_with_all_true_seg]
    no_merge_in_seg = len(errors_seg_with_all_true)

    obj_mask_at_least_one_false_split = np.array([(x == 2).any() for x in errors_rsd[thresh]])
    no_merge_in_seg_false_split_resolved = sum(
        np.logical_and(
            obj_mask_at_least_one_false_split, obj_mask_with_all_true_seg
        )
    )

    false_merge_seg_false_split_resolved = sum(
        np.logical_and(
            obj_mask_at_least_one_false_split, np.logical_not(obj_mask_with_all_true_seg)
        )
    )

    number_of_objs = len(errors_seg[thresh])

    print '================================\nThreshold = {}'.format(thresh)
    print '--------------------------------'
    print 'Path evaluation: '
    print '    Recall:    {}'.format(result_path[thresh]['recall'])
    print '    Precision: {}'.format(result_path[thresh]['precision'])
    print '    F1:        {}'.format(result_path[thresh]['f1'])
    print 'Object evalutation: '
    print '    Recall:    {}'.format(result_obj[thresh]['recall'])
    print '    Precision: {}'.format(result_obj[thresh]['precision'])
    print '    F1:        {}'.format(result_obj[thresh]['f1'])
    print '--------------------------------'
    print 'PATH LEVEL'
    print 'Original segmentation:'
    print '    Correct merges: {}'.format(orig_error_counts[orig_error_types == True])
    print '    False merges:   {}'.format(orig_error_counts[orig_error_types == False])
    print 'Resolved segmentation:'
    print '    Correct merges: {}'.format(error_counts[error_types == 3])
    print '    Correct splits: {}'.format(error_counts[error_types == 0])
    print '    False merges:   {}'.format(error_counts[error_types == 1])
    print '    False splits:   {}'.format(error_counts[error_types == 2])
    print '--------------------------------'
    print 'OBJECT LEVEL ({} objects with...)'.format(number_of_objs)
    print '... at least one false merged path in seg:       {}'.format(false_merge_seg)
    print '    ... at least one false split in resolved     {}'.format(false_merge_seg_false_split_resolved)
    print '... no false merged paths in seg:                {}'.format(no_merge_in_seg)
    print '    ... at least one false split in resolved:    {}'.format(no_merge_in_seg_false_split_resolved)
    print '... at least one false merged path in resolved:  {}'.format(false_merge_rsd)



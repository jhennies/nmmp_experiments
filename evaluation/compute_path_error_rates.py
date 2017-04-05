
import pickle
import numpy as np
import vigra
from copy import deepcopy

# results_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_test_pipeline_update/'
# results_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_test_pipeline_update/'
results_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/'
paths_folder = results_folder + 'cache/path_data/'
# source_folder = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
# Load paths of false merge computation
with open(paths_folder + 'false_paths_predictions.pkl', mode='r') as f:
    false_merge_probs = pickle.load(f)

with open(paths_folder + 'path_splB_z1.pkl', mode='r') as f:
    path_data = pickle.load(f)

paths = path_data['paths']
paths_to_objs = path_data['paths_to_objs']

# Load GT image to determine real classes
gt = vigra.readHDF5(
    source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
    'z/1/neuron_ids'
)

min_prob_thresh = 0.5
thresh_range = np.arange(min_prob_thresh, 1.0, 0.1)
thresh_range = [0.5]

# Load the original segmentation
original_seg = vigra.readHDF5(
    results_folder + 'result.h5',
    'z/1/test'
)

# Load the resolved segmentation
# We only have to load the resolved segmentation for the smallest threshold, from this
#   we can derive all the others
resolved_seg = vigra.readHDF5(
    results_folder + 'result_resolved_thresh_{}.h5'.format(min_prob_thresh),
    'z/1/test'
)

# Evaluation: Do this for different probs-thresholds

# a) Determine values along the path to extract the real class of the object
# b) Compare end points of paths to determine real class of the object
# FIXME For now going for (b)

store_eval = {}

for thresh in thresh_range:

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

    result_obj = new_eval(predicted_obj, reference_obj)
    result_path = new_eval(predicted, reference)

    print '================================\nThreshold = {}'.format(thresh)
    print '--------------------------------'
    print 'Path evaluation: '
    print '    Recall:    {}'.format(result_path['recall'])
    print '    Precision: {}'.format(result_path['precision'])
    print '    F1:        {}'.format(result_path['f1'])
    print 'Object evalutation: '
    print '    Recall:    {}'.format(result_obj['recall'])
    print '    Precision: {}'.format(result_obj['precision'])
    print '    F1:        {}'.format(result_obj['f1'])

    # Evaluation of resolved paths
    resolved_count = len(objs_with_prob_greater_thresh)

    # How many of the resolved objects were merged?
    merged_objs = np.intersect1d(objs_with_prob_greater_thresh, objs_with_merged_path)
    merged = len(merged_objs)
    # And not merged
    not_merged_objs = np.setdiff1d(objs_with_prob_greater_thresh, objs_with_merged_path)
    not_merged = len(not_merged_objs)

    merged_objs = []
    merged_split_objs = []
    merged_split_correct_objs = []
    merged_not_split_objs = []

    not_merged_objs = []
    not_merged_split_objs = []
    not_merged_not_split_objs = []

    # TODO: Create a general object mask here for speed-up
    general_obj_mask_z = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(original_seg[:, :, z])[:, :, None] for z in xrange(original_seg.shape[2])],
        axis=2
    )
    general_obj_mask_y = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(original_seg[:, y, :])[:, None, :] for y in xrange(original_seg.shape[1])],
        axis=1
    )
    general_obj_mask = deepcopy(original_seg)
    general_obj_mask[general_obj_mask_y == 1] = 0
    general_obj_mask[general_obj_mask_z == 1] = 0
    eroded_general_obj_mask = vigra.filters.discErosion(general_obj_mask.astype(np.uint8), 10)
    # FIXME hack to make the discErosion capable of uint>8
    general_obj_mask[eroded_general_obj_mask == 0] = 0

    resolved_obj_mask_z = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(resolved_seg[:, :, z])[:, :, None] for z in xrange(original_seg.shape[2])],
        axis=2
    )
    resolved_obj_mask_y = np.concatenate(
        [vigra.analysis.regionImageToEdgeImage(resolved_seg[:, y, :])[:, None, :] for y in xrange(original_seg.shape[1])],
        axis=1
    )
    resolved_obj_mask = deepcopy(original_seg)
    resolved_obj_mask[general_obj_mask_y == 1] = 0
    resolved_obj_mask[general_obj_mask_z == 1] = 0
    eroded_resolved_obj_mask = vigra.filters.discErosion(resolved_obj_mask.astype(np.uint8), 10)
    # FIXME hack to make the discErosion capable of uint>8
    resolved_obj_mask[eroded_resolved_obj_mask == 0] = 0


    # Iterate over the resolved objects
    for obj in objs_with_prob_greater_thresh:

        print 'Checking object {}'.format(obj)

        # # Get the object from the segmentation
        # labels_in_resolved_seg = resolved_seg[original_seg == obj]
        # # Get the respective gt
        # labels_in_gt = gt[original_seg == obj]

        # Get the object mask
        obj_mask = original_seg == obj
        eroded_mask = general_obj_mask == obj
        # eroded_mask = vigra.filters.discErosion(obj_mask.astype(np.uint8), 10)

        # FIXME Look over this with Anna !!!

        # labels_in_gt_unique, labels_in_gt_counts = np.unique(
        #     gt[original_seg == obj], return_counts=True
        # )

        # sum_of_pixels = np.sum(labels_in_gt_counts)
        # if np.sum(labels_in_gt_counts > (sum_of_pixels * 0.05)) > 1:
        if len(np.unique(gt[eroded_mask == 1])) > 2:
            merged_objs.append(obj)
            is_merge = True
        else:
            is_merge = False
            not_merged_objs.append(obj)

        # labels_in_resolved_unique, labels_in_resolved_counts = np.unique(
        #     resolved_seg[original_seg == obj], return_counts=True
        # )

        # def is_correct(labels, gt, obj_mask):
        #
        #     labels[np.logical_not(obj_mask)] = 0
        #
        #     correct = True
        #     lbls, counts = np.unique(labels, return_counts=True)
        #
        #     for lbl in lbls[1:]:
        #         # mask = labels == lbl
        #         #
        #         # erd_mask = vigra.filters.discErosion(mask.astype(np.uint8), 10)
        #         erd_mask = labels == lbl
        #
        #         if np.amax(erd_mask):
        #             if len(np.unique(gt[erd_mask > 0])) > 2:
        #                 correct = False
        #
        #     return correct

        # sum_of_pixels = np.sum(labels_in_resolved_counts)
        # if np.sum(labels_in_resolved_counts > (sum_of_pixels * 0.05)) > 1:
        if len(np.unique(resolved_seg[eroded_mask == 1])) > 2:

            if is_merge:
                # This is correct
                merged_split_objs.append(obj)
                # if is_correct(deepcopy(resolved_obj_mask), gt, obj_mask):
                #     merged_split_correct_objs.append(obj)

            else:
                not_merged_split_objs.append(obj)
        else:
            if is_merge:
                merged_not_split_objs.append(obj)
            else:
                # This is correct
                not_merged_not_split_objs.append(obj)

    merged = len(merged_objs)
    not_merged = len(not_merged_objs)
    merged_split = len(merged_split_objs)
    not_merged_split = len(not_merged_split_objs)
    merged_not_split = len(merged_not_split_objs)
    not_merged_not_split = len(not_merged_not_split_objs)
    merged_split_correct = len(merged_split_correct_objs)

    store_eval[thresh] = {
        'merged_split': merged_split_objs,
        'merged_not_split': merged_not_split_objs,
        'not_merged_split': not_merged_split_objs,
        'not_merged_not_split': not_merged_not_split_objs
    }

    print '--------------------------------'
    print 'Evaluation of {} resolved objects'.format(resolved_count)
    print '    Merged:          {}'.format(merged)
    print '        Split:       {}'.format(merged_split)
    print '            Correct: {}'.format(merged_split_correct)
    print '        Not split:   {}'.format(merged_not_split)
    print '    Not merged:      {}'.format(not_merged)
    print '        Split:       {}'.format(not_merged_split)
    print '        Not split:   {}'.format(not_merged_not_split)


    #
    # from alex_evaluation import comparison
    #
    # correct_obj = comparison(original_seg, gt, resolved_seg)
    # correct = len(correct_obj)
    #
    # print '--------------------------------'
    # print 'Correctly segmented objects: {}'.format(correct)



with open(results_folder + 'evaluation.pkl', mode='w') as f:
    pickle.dump(store_eval, f)
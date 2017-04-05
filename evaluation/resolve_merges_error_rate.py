import vigra
import numpy as np
import pickle

def resolve_merges_error_rate(paths, paths_to_objs, resolved, gt):

    all_errors_rsd = {}
    all_errors_seg = {}

    for obj in np.unique(paths_to_objs):

        # print obj

        mask = np.array(paths_to_objs) == obj
        paths_obj = np.array(paths)[mask]
        # The end points of the paths as list of tuples
        end_points = [[x[0], x[-1]] for x in paths_obj]

        # # Reshape that array to get list of all end points
        # end_points = np.reshape(end_points, (len(end_points) * 2, 3))
        # # Remove the duplicate entries
        # a = np.ascontiguousarray(end_points).view(np.dtype((np.void, end_points.dtype.itemsize * end_points.shape[1])))
        # _, idx = np.unique(a, return_index=True)
        # end_points = end_points[idx]
        # # Create all possible combinations ...

        # The end points extracted like this are already all possible combinations,
        # assuming we computed all of the possible paths
        pairs = end_points

        # Labels according to the coordinate pairs
        gt_pairs = np.array([[gt[tuple(x[0])], gt[tuple(x[1])]] for x in pairs])
        rs_pairs = np.array([[resolved[tuple(x[0])], resolved[tuple(x[1])]] for x in pairs])
        gt_equal = np.equal(gt_pairs[:, 0], gt_pairs[:, 1])
        rs_equal = np.equal(rs_pairs[:, 0], rs_pairs[:, 1])

        errors = gt_equal.astype(np.uint8) * 2 + rs_equal.astype(np.uint8)
        # Like this we can find the erroneous and correct pairs with information on
        #   the error type:
        #   3: gt == True, rs == True -> correct merge
        #   2: gt == True, rs == False -> false split
        #   1: gt == False, rs == True -> false merge
        #   0: gt == False, rs == False -> correct split
        # The original performance with respect to merged obj is in gt_equal:
        #   True -> correct merge
        #   False -> false merge

        all_errors_rsd[obj] = errors
        all_errors_seg[obj] = gt_equal

    return all_errors_seg, all_errors_rsd


if __name__ == '__main__':

    cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170331_splB_z1_defcor/cache/'
    source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

    path_data_file = cache_folder + 'path_data/path_splB_z1.pkl'
    seg_file = cache_folder + '../result.h5'
    seg_key = 'z/1/test'
    resolved_file = cache_folder + '../result_resolved_thresh_0.5.h5'
    resolved_key = seg_key
    gt_file = source_folder + 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    gt_key = 'z/1/neuron_ids'

    # Load paths
    with open(path_data_file, mode='r') as f:
        path_data = pickle.load(f)
    paths = path_data['paths']
    paths_to_objs = path_data['paths_to_objs']

    # Load images
    # seg = vigra.readHDF5(seg_file, seg_key)
    resolved = vigra.readHDF5(resolved_file, resolved_key)
    gt = vigra.readHDF5(gt_file, gt_key)

    # Determine merge error rate
    errors_seg, errors_rsd = resolve_merges_error_rate(paths, paths_to_objs, resolved, gt)

    all_errors_rsd = np.concatenate(errors_rsd.values())
    all_errors_seg = np.concatenate(errors_seg.values())

    error_types, error_counts = np.unique(all_errors_rsd, return_counts=True)

    # Like this we can find the erroneous and correct pairs with information on
    #   the error type:
    #   3: gt == True, rs == True -> correct merge
    #   2: gt == True, rs == False -> false split
    #   1: gt == False, rs == True -> false merge
    #   0: gt == False, rs == False -> correct split
    # The original performance with respect to merged obj is in gt_equal:
    #   True -> correct merge
    #   False -> false merge

    print 'EVALUATION'
    print '----------'
    print 'Correct merges: {}'.format(error_counts[error_types == 3])
    print 'Correct splits: {}'.format(error_counts[error_types == 0])
    print 'False merges:   {}'.format(error_counts[error_types == 1])
    print 'False splits:   {}'.format(error_counts[error_types == 2])




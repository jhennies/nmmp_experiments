
import os
import vigra
import numpy as np

def count_paths(test_spl, spls, train_path_cache, detailed=False):

    print '\nEvaluating for test sample {}'.format(test_spl)
    print '----------------------------------'

    classes = []
    class_counts = []
    num_paths = []
    num_objs = []

    path_count_all = []
    path_count_0 = []
    true_path_count_all = []
    true_path_count_0 = []
    false_path_count_all = []
    false_path_count_0 = []
    obj_count_all = []
    obj_count_0 = []
    true_obj_count_all = []
    true_obj_count_0 = []
    false_obj_count_all = []
    false_obj_count_0 = []

    for spl in spls:

        if spl == test_spl:
            continue

        for idx in xrange(0, 9):

            path_file = os.path.join(train_path_cache,
                                     'paths_ds_ds_train_{}_seg_{}.h5'.format(spl, idx))

            paths_to_objs = np.array(vigra.readHDF5(path_file, 'paths_to_objs'))
            path_classes = np.array(vigra.readHDF5(path_file, 'path_classes'))

            if paths_to_objs.any():

                us, cs = np.unique(path_classes, return_counts=True)
                if us.any():
                    trues = cs[us]
                    falses = cs[np.logical_not(us)]
                else:
                    trues = np.array([])
                    falses = np.array([])

                u_obj_true = np.unique(paths_to_objs[path_classes])
                u_obj_false = np.unique(paths_to_objs[np.logical_not(path_classes)])

                if idx == 0:
                    path_count_0.append(len(paths_to_objs))
                    if trues.any():
                        true_path_count_0.append(trues[0])
                    if falses.any():
                        false_path_count_0.append(falses[0])
                    obj_count_0.append(len(np.unique(paths_to_objs)))
                    true_obj_count_0.append(len(u_obj_true))
                    false_obj_count_0.append(len(u_obj_false))

                path_count_all.append(len(paths_to_objs))
                if trues.any():
                    true_path_count_all.append(trues[0])
                if falses.any():
                    false_path_count_all.append(falses[0])
                obj_count_all.append(len(np.unique(paths_to_objs)))
                true_obj_count_all.append(len(u_obj_true))
                false_obj_count_all.append(len(u_obj_false))

    if detailed:
        # print 'All paths:           {:5d}; {}'.format(np.sum(path_count_all), path_count_all)
        # print 'Paths in seg0:       {:5d}; {}'.format(np.sum(path_count_0), path_count_0)
        print 'All true paths:      {:5d}; {}'.format(np.sum(true_path_count_all), true_path_count_all)
        print 'True paths in seg0:  {:5d}; {}'.format(np.sum(true_path_count_0), true_path_count_0)
        print 'All false paths:     {:5d}; {}'.format(np.sum(false_path_count_all), false_path_count_all)
        print 'False paths in seg0: {:5d}; {}'.format(np.sum(false_path_count_0), false_path_count_0)

        print ''

        # print 'All objs:            {:5d}; {}'.format(np.sum(obj_count_all), obj_count_all)
        # print 'Objs in seg0:        {:5d}; {}'.format(np.sum(obj_count_0), obj_count_0)
        print 'All true objs:       {:5d}; {}'.format(np.sum(true_obj_count_all), true_obj_count_all)
        print 'True objs in seg0:   {:5d}; {}'.format(np.sum(true_obj_count_0), true_obj_count_0)
        print 'All false objs:      {:5d}; {}'.format(np.sum(false_obj_count_all), false_obj_count_all)
        print 'False objs in seg0:  {:5d}; {}'.format(np.sum(false_obj_count_0), false_obj_count_0)

    else:
        # print 'All paths:           {:5d}; {}'.format(np.sum(path_count_all), path_count_all)
        # print 'Paths in seg0:       {:5d}; {}'.format(np.sum(path_count_0), path_count_0)
        print 'All true paths:      {:5d}'.format(np.sum(true_path_count_all))
        print 'All false paths:     {:5d}'.format(np.sum(false_path_count_all))
        print 'True paths in seg0:  {:5d}'.format(np.sum(true_path_count_0))
        print 'False paths in seg0: {:5d}'.format(np.sum(false_path_count_0))


        print ''

        # print 'All objs:            {:5d}; {}'.format(np.sum(obj_count_all), obj_count_all)
        # print 'Objs in seg0:        {:5d}; {}'.format(np.sum(obj_count_0), obj_count_0)
        print 'All true objs:       {:5d}'.format(np.sum(true_obj_count_all))
        print 'All false objs:      {:5d}'.format(np.sum(false_obj_count_all))
        print 'True objs in seg0:   {:5d}'.format(np.sum(true_obj_count_0))
        print 'False objs in seg0:  {:5d}'.format(np.sum(false_obj_count_0))


if __name__ == '__main__':

    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'
    train_path_cache = os.path.join(project_folder, 'train_paths_cache/')

    samples = ['splA_z0', 'splA_z1', 'splB_z0', 'splB_z1', 'splC_z0', 'splC_z1']

    for sample in samples:
        count_paths(sample, samples,
                    train_path_cache)

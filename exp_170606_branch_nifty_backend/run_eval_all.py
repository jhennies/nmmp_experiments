def run_roi_and_rand_general():

    from eval_all import roi_and_rand_general

    project_folder = '/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170606_branch_nifty_backend/'
    # project_folder = '/media/hdb/jhennies/neuraldata/results/multicut_workflow/170530_new_baseline/'
    # project_folder = '/media/julian/Daten/datasets/results/multicut_workflow/170530_new_baseline/'
    # result_file = 'result_resolved_local.h5'
    # result_files = ['result.h5', 'result_resolved_local.h5']
    result_files = ['result.h5', 'result_resolved_local.h5']
    source_folder = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/'
    # source_folder = '/media/julian/Daten/datasets/cremi_2016/170321_resolve_false_merges/'

    # samples = ['A', 'A', 'B', 'B', 'C', 'C']
    # halves = [0, 1] * 4
    # defect_corrects = [False, False, True, True, True, True]

    samples = ['A', 'A']
    halves = [0, 1]
    defect_corrects = [False, False]

    for idx, sample in enumerate(samples):
        half = halves[idx]
        defect_correct = defect_corrects[idx]
        for result_file in result_files:
            roi_and_rand_general(sample, half, defect_correct, project_folder,
                                 source_folder, result_file, caching=True, debug=False)


def run_eval_obj_measures_readable():

    from eval_all import eval_obj_measures_readable

    samples = ['A', 'A', 'B', 'B', 'C', 'C']
    halves = [0, 1] * 4
    defect_corrects = [False, False, True, True, True, True]

    for spl_id, spl in enumerate(samples):

        half = halves[spl_id]
        defect_correct = defect_corrects[spl_id]

        print '\nEvaluating spl{}_z{}'.format(spl, half)

        project_folder = '/home/julian/ssh_data/neuraldata/results/multicut_workflow/170530_new_baseline/'
        seg_file = 'result.h5'
        seg_key = 'z/{}/data'.format(half)
        resolved_files = ['result_resolved_local.h5']
        resolved_key = seg_key

        result = eval_obj_measures_readable(
            spl, half,
            project_folder,
            seg_file, seg_key,
            resolved_files, resolved_key,
            thresh_range=None,
            resolved_only=False,
            defect_correct=defect_correct
        )
        result = result[0][0]

        # print result

        print 'Objects:            {}'.format(result[0])
        print 'TP:                 {}'.format(result[1])
        print '    Fully resolved: {}'.format(result[3])
        print '    Falsely split:  {}'.format(result[4])
        print 'FP:                 {}'.format(result[2])
        print '    Falsely split:  {}'.format(result[5])


def run_plot_all_sample_path_eval_split_samples():

    from eval_all import plot_all_sample_path_eval_split_samples
    samples = ['A', 'A', 'B', 'B', 'C', 'C']
    halves = [0, 1] * 4
    defect_corrects = [False, False, True, True, True, True]

    project_folder = '/home/julian/ssh_data/neuraldata/results/multicut_workflow/170530_new_baseline/'

    thresh_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Plot sample path evaluation
    for i in xrange(0, 6, 2):
        plot_all_sample_path_eval_split_samples(project_folder, thresh_range,
                                  halves[i: i+2],
                                  defect_corrects[i: i+2],
                                  samples[i: i+2],
                                  measures=['f1', 'precision', 'recall'])
    # plot_all_sample_path_eval_split_samples(
    #     project_folder,
    #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     halves, defect_corrects,
    #     samples, measures=['f1', 'precision', 'recall']
    # )


if __name__ == '__main__':
    run_roi_and_rand_general()
def run_roi_and_rand_general():

    from eval_all import roi_and_rand_general

    project_folder = '/media/hdb/jhennies/neuraldata/results/multicut_workflow/170530_new_baseline/'
    # project_folder = '/media/julian/Daten/datasets/results/multicut_workflow/170530_new_baseline/'
    # result_file = 'result_resolved_local.h5'
    result_files = ['result.h5', 'result_resolved_local.h5']
    source_folder = '/media/hdb/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
    # source_folder = '/media/julian/Daten/datasets/cremi_2016/170321_resolve_false_merges/'

    samples = ['A', 'A', 'B', 'B', 'C', 'C']
    halves = [0, 1] * 4
    defect_corrects = [False, False, True, True, True, True]

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


if __name__ == '__main__':
    pass

import vigra
import cPickle as pickle
import os
import re

import sys
sys.path.append('/home/jhennies/src/cremi_python/cremi_python')
from cremi import Volume, NeuronIds

def roi_and_rand_general(
        sample, half, defect_correct, project_folder,
        source_folder,
        result_file, caching=False, debug=False
):
    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print 'Result file: {}'.format(result_file)

    experiment_folder = os.path.join(project_folder, 'spl{}_z{}/'.format(sample, half))

    if caching:
        cache_filepath = os.path.join(
            experiment_folder,
            re.sub('.h5$', '', result_file) + '_roi_and_rand_cache.pkl'
        )
    else:
        cache_filepath = None

    if caching and os.path.isfile(cache_filepath):
        with open(cache_filepath, mode='r') as f:
            return pickle.load(f)

    else:

        if defect_correct:
            defect_correct_str = '_defect_correct'
        else:
            defect_correct_str = ''

        mc_result_key = 'z/{}/data'.format(half)

        # # Load stuff
        # source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

        # # TODO: Change here when switching sample
        # ref_result_filepath = os.path.join(source_folder, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(sample))
        # # TODO: Change here when switching half
        # ref_result_key = 'z/{}/beta_0.5'.format(half)

        gt_filepath = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defect_correct_str))
        gt_key = 'z/{}/neuron_ids'.format(half)

        # ref_result, _, _ = vigra.analysis.relabelConsecutive(vigra.readHDF5(ref_result_filepath, ref_result_key), start_label=1, keep_zeros=True)
        if not debug:
            gt = vigra.readHDF5(gt_filepath, gt_key)
            vol_gt = Volume(gt)
            neuron_ids_evaluation = NeuronIds(vol_gt)

        mc_result_filepath = os.path.join(experiment_folder, result_file)

        if not debug:
            # Evaluate baseline
            mc_result = vigra.readHDF5(mc_result_filepath, mc_result_key)
            vol_mc_result = Volume(mc_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_mc_result)
            adapted_rand = neuron_ids_evaluation.adapted_rand(vol_mc_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            adapted_rand = 0.23

        print "\tvoi split   : " + str(voi_split)
        print "\tvoi merge   : " + str(voi_merge)
        print "\tadapted RAND: " + str(adapted_rand)

        if caching:
            with open(cache_filepath, mode='w') as f:
                pickle.dump((voi_split, voi_merge, adapted_rand), f)

        return (voi_split, voi_merge, adapted_rand)

if __name__ == '__main__':

    project_folder = '/media/hdb/jhennies/neuraldata/results/multicut_workflow/170530_new_baseline/'
    result_file = 'result.h5'
    source_folder = '/media/hdb/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

    samples = ['A', 'A', 'B', 'B']
    halves = [0, 1, 0, 1]
    defect_corrects = [False, False, True, True]

    for idx, sample in enumerate(samples):
        half = halves[idx]
        defect_correct = defect_corrects[idx]
        roi_and_rand_general(sample, half, defect_correct, project_folder,
                             source_folder, result_file, caching=False, debug=False)
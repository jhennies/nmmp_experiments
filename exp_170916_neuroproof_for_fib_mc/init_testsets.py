
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

# The following locations should be importable by downstream scripts
testset_source_folder = '/export/home/jhennies/sshs/fib_25/fib25/170809_chunked_fib25/normalized/'
project_folder = '/mnt/ssd/jhennies/results/multicut_workflow/170916_neuroproof_for_fib_mc/'

testset_names = ['fib_8_5_6', 'fib_8_5_7', 'fib_7_5_6', 'fib_7_5_7']
experiment_ids = [0, 1, 2, 3]
mc_trainset_names = ['neuroproof_train']
path_trainset_names = ['neuroproof_test']

result_keys = ['data'] * 4

meta_folder = os.path.join(project_folder, 'cache')

from pipeline import init_datasets

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    make_cutouts = [True] * 4

    raw_files = [
        'fib25_raw_chunks_selected.h5'
    ] * 4
    raw_names = ['data_8_5_6', 'data_8_5_7', 'data_7_5_6', 'data_7_5_7']
    probs_files = [
        'fib25_membrane_predictions_chunks_selected.h5'
    ] * 4
    probs_names = ['data_8_5_6', 'data_8_5_7', 'data_7_5_6', 'data_7_5_7']
    seg_files = [
        'fib25_vanilla_watershed_relabeled_chunks_selected.h5'
    ] * 4
    seg_names = ['data_8_5_6', 'data_8_5_7', 'data_7_5_6', 'data_7_5_7']
    gt_files = [
        'fib25_gt_chunks_selected.h5'
    ] * 4
    gt_names = ['data_8_5_6', 'data_8_5_7', 'data_7_5_6', 'data_7_5_7']

    # Init train sets
    init_datasets(
        meta_folder, testset_names,
        testset_source_folder, raw_files, raw_names,
        testset_source_folder, probs_files, probs_names,
        testset_source_folder, seg_files, seg_names,
        testset_source_folder, gt_files, gt_names,
        make_cutouts=make_cutouts
    )

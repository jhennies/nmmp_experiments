
import os

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

# The following locations should be importable by downstream scripts
testset_source_folder = '/mnt/localdata1/jhennies/neuraldata/fib25/overlap_50/'
project_folder = '/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170919_neuroproof_for_fib_overlap/'

testset_names = ['fib_5_6_7', 'fib_5_6_8', 'fib_6_6_7', 'fib_6_6_8']
experiment_ids = [0]
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
        'fib25_raw_chunks_selected_olap50.h5'
    ] * 4
    raw_names = ['data_5_6_7', 'data_5_6_8', 'data_6_6_7', 'data_6_6_8']
    probs_files = [
        'fib25_membrane_predictions_chunks_selected_olap50.h5'
    ] * 4
    probs_names = ['data_5_6_7', 'data_5_6_8', 'data_6_6_7', 'data_6_6_8']
    seg_files = [
        'fib25_vanilla_watershed_relabeled_chunks_selected_olap50.h5'
    ] * 4
    seg_names = ['data_5_6_7', 'data_5_6_8', 'data_6_6_7', 'data_6_6_8']
    gt_files = [
        'fib25_gt_chunks_selected_olap50.h5'
    ] * 4
    gt_names = ['data_5_6_7', 'data_5_6_8', 'data_6_6_7', 'data_6_6_8']

    # Init train sets
    init_datasets(
        meta_folder, testset_names,
        testset_source_folder, raw_files, raw_names,
        testset_source_folder, probs_files, probs_names,
        testset_source_folder, seg_files, seg_names,
        testset_source_folder, gt_files, gt_names,
        make_cutouts=make_cutouts
    )

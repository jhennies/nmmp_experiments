
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

# The following locations should be importable by downstream scripts
# TODO Change here when switching server
source_folder = '/mnt/data/Neuro/neuroproof_data/'
project_folder = '/mnt/localdata0/jhennies/results/multicut_workflow/170727_neuroproof/'

ds_names = ['train_mc', 'train_path_classifier']
experiment_ids = [1]
# experiment_ids = [0, 1, 3, 4, 5]
result_keys = ['data'] * 2

meta_folder = os.path.join(project_folder, 'cache')

from pipeline import init_datasets

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    raw_files = [
        'raw_train.h5',
        'raw_test.h5'
    ]
    raw_names = ['data'] * 2
    probs_files = [
        'probabilities_train.h5',
        'probabilities_test.h5'
    ]
    probs_names = ['data'] * 2
    seg_files = [
        'overseg_train.h5',
        'overseg_test.h5'
    ]
    seg_names = ['data'] * 2
    gt_files = [
        'gt_train.h5',
        'gt_test.h5'
    ]
    gt_names = ['data'] * 2

    # Init train sets
    init_datasets(
        meta_folder, ds_names,
        source_folder, raw_files, raw_names,
        source_folder, probs_files, probs_names,
        source_folder, seg_files, seg_names,
        source_folder, gt_files, gt_names
    )
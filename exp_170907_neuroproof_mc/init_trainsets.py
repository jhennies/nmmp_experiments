
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

# The following locations should be importable by downstream scripts
# TODO Change here when switching server
trainset_source_folder = '/mnt/data/Neuro/neuroproof_data/'
project_folder = '/mnt/localdata0/jhennies/results/multicut_workflow/170907_neuroproof_mc/'

# neuroproof_train is the 256 block used for training the multicuts
#   On this one no multicut is computed -> just training
# neuroproot_test is the 512 block used to train the path classifier
#   All the different beta segmentations with conventional multicut are computed on this dataset
trainset_names = ['neuroproof_train', 'neuroproof_test']

# Select neuroproof_test for multicut computation
train_ids = [1]

# This has to have the same shape as ds_names but the first entry will never be read in this setting
result_keys = ['', 'beta']

meta_folder = os.path.join(project_folder, 'cache')

from pipeline import init_datasets

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    make_cutouts = [True, False]

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
        meta_folder, trainset_names,
        trainset_source_folder, raw_files, raw_names,
        trainset_source_folder, probs_files, probs_names,
        trainset_source_folder, seg_files, seg_names,
        trainset_source_folder, gt_files, gt_names,
        make_cutouts=make_cutouts
    )
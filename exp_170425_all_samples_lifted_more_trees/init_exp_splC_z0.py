
import os

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from pipeline import init_dataset

# The following locations should be importable by downstream scripts
# TODO Change here when switching server
source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170425_all_samples_lifted_more_trees/'
# TODO Change here
experiment_folder = project_folder + 'splC_z0/'
meta_folder = experiment_folder + 'cache/'
# TODO Change here
test_name = 'splC_z0'
train_name = 'splC_z1'

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    # TODO Change here for changing sample
    raw_path = source_folder
    raw_file = 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    probs_path = source_folder
    probs_file = 'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    seg_path = source_folder
    seg_file = 'cremi.splC.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'
    gt_path = source_folder
    gt_file = 'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'

    # Init test set
    # TODO Change here for changing half
    init_dataset(
        meta_folder, test_name,
        raw_path + raw_file, 'z/0/raw',
        probs_path + probs_file, 'z/0/data',
        seg_path + seg_file, 'z/0/labels'
    )

    # Init train set
    # TODO Change here for changing half
    init_dataset(
        meta_folder, train_name,
        raw_path + raw_file, 'z/1/raw',
        probs_path + probs_file, 'z/1/data',
        seg_path + seg_file, 'z/1/labels',
        gt_filepath=gt_path + gt_file, gt_name='z/1/neuron_ids',
        make_cutouts=True
    )

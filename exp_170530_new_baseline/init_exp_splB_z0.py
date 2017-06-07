
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

from pipeline import init_dataset, init_train_sets

# The following locations should be importable by downstream scripts
# TODO Change here when switching server
source_folder = '/media/hdb/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
project_folder = '/media/hdb/jhennies/neuraldata/results/multicut_workflow/170530_new_baseline/'
# TODO Change here
experiment_folder = os.path.join(project_folder, 'splB_z0/')
meta_folder = os.path.join(experiment_folder, 'cache/')
train_folder = os.path.join(project_folder, 'train_sets_cache/')
# TODO Change here
test_name = 'splB_z0'
# These include all names, the test set is excluded in run_lifted_mc
train_names = ['splA_z0', 'splA_z1', 'splB_z0', 'splB_z1', 'splC_z0', 'splC_z1']

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    # TODO Change here for changing sample
    raw_path = source_folder
    raw_file = 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    probs_path = source_folder
    probs_file = 'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    seg_path = source_folder
    seg_file = 'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'

    # Init test set
    # TODO Change here for changing half
    init_dataset(
        meta_folder, test_name,
        raw_path + raw_file, 'z/0/raw',
        probs_path + probs_file, 'z/0/data',
        seg_path + seg_file, 'z/0/labels'
    )

    train_raw_files = [
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_raw_names = ['z/0/raw', 'z/1/raw'] * 3
    train_probs_files = [
        'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.probs.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_probs_names = ['z/0/data', 'z/1/data'] * 3
    train_seg_files = [
        'cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    train_seg_names = ['z/0/labels', 'z/1/labels'] * 3
    gt_path = source_folder
    gt_files = [
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5',
        'cremi.splC.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    ]
    gt_names = ['z/0/neuron_ids', 'z/1/neuron_ids'] * 3

    # Init train sets
    init_train_sets(
        train_folder, train_names,
        raw_path, train_raw_files, train_raw_names,
        probs_path, train_probs_files, train_probs_names,
        seg_path, train_seg_files, train_seg_names,
        gt_path, gt_files, gt_names
    )


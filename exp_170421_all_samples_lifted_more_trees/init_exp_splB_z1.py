
import os
from multicut_src import DataSet

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')


def init_dataset(
        meta_folder, name,
        raw_filepath, raw_name,
        probs_filepath, probs_name,
        seg_filepath, seg_name,
        gt_filepath=None, gt_name=None,
        make_cutouts=False
):

    # Init the dataset
    ds = DataSet(meta_folder, name)

    # Add data
    ds.add_raw(raw_filepath, raw_name)
    ds.add_input(probs_filepath, probs_name)
    ds.add_seg(seg_filepath, seg_name)
    if gt_filepath is not None:
        ds.add_gt(gt_filepath, gt_name)

    # add cutouts for lifted multicut training
    if make_cutouts:
        shape = ds.shape
        z_offset = 10
        ds.make_cutout([0, 0, 0], [shape[0], shape[1], z_offset])
        ds.make_cutout([0, 0, z_offset], [shape[0], shape[1], shape[2] - z_offset])
        ds.make_cutout([0, 0, shape[2] - z_offset], [shape[0], shape[1], shape[2]])

# The following locations should be importable by downstream scripts
source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170421_all_samples_lifted_more_trees/'
experiment_folder = project_folder + 'splB_z1/'
meta_folder = project_folder + 'splB_z1/cache/'
test_name = 'splB_z1'
train_name = 'splB_z0'

if __name__ == '__main__':

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    if not os.path.exists(meta_folder):
        os.mkdir(meta_folder)

    raw_path = source_folder
    raw_file = 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'
    probs_path = source_folder
    probs_file = 'cremi.splB.train.probs_defect_correct.crop.axes_xyz.split_z.h5'
    seg_path = source_folder
    seg_file = 'cremi.splB.train.wsdt_relabel_defect_correct.crop.axes_xyz.split_z.h5'
    gt_path = source_folder
    gt_file = 'cremi.splB.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'

    # Init test set
    init_dataset(
        meta_folder, test_name,
        raw_path + raw_file, 'z/1/raw',
        probs_path + probs_file, 'z/1/data',
        seg_path + seg_file, 'z/1/labels'
    )

    # Init train set
    init_dataset(
        meta_folder, train_name,
        raw_path + raw_file, 'z/0/raw',
        probs_path + probs_file, 'z/0/data',
        seg_path + seg_file, 'z/0/labels',
        gt_filepath=gt_path + gt_file, gt_name='z/0/neuron_ids',
        make_cutouts=True
    )

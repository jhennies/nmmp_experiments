
import os
import vigra
import cPickle as pickle
import numpy as np

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import DataSet
from multicut_src import lifted_multicut_workflow, multicut_workflow
from multicut_src import load_dataset
from multicut_src import compute_false_merges
from multicut_src import resolve_merges_with_lifted_edges_global, resolve_merges_with_lifted_edges
from multicut_src import RandomForest
from multicut_src import ExperimentSettings
from multicut_src import merge_small_segments

import logging
logger = logging.getLogger(__name__)

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


def init_train_sets(
        meta_folder, names,
        raw_path, raw_files, raw_names,
        probs_path, probs_files, probs_names,
        seg_path, seg_files, seg_names,
        gt_path, gt_files, gt_names
):

    for idx, train_name in enumerate(names):

        if not os.path.exists(os.path.join(meta_folder, train_name)):

            print 'Training set {} is being created ...'.format(train_name)

            raw_file = raw_files[idx]
            probs_file = probs_files[idx]
            seg_file = seg_files[idx]
            gt_file = gt_files[idx]
            raw_name = raw_names[idx]
            probs_name = probs_names[idx]
            seg_name = seg_names[idx]
            gt_name = gt_names[idx]

            init_dataset(
                meta_folder, train_name,
                raw_path + raw_file, raw_name,
                probs_path + probs_file, probs_name,
                seg_path + seg_file, seg_name,
                gt_filepath=gt_path + gt_file, gt_name=gt_name,
                make_cutouts=True
            )

        else:

            print 'Training set {} exists, nothing to do.'.format(train_name)


def run_mc(
        meta_folder,
        train_folder,
        ds_train_names,
        ds_test_name,
        save_path,
        results_name,
        pre_save_path=None
):
    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    merge_segments = True
    compute_mc = True
    # if os.path.isfile(save_path):  # Nothing to do
    #     compute_mc = False
    #     merge_segments = False
    # else:                          # Final result needs to be computed
    #     if pre_save_path is not None:
    #         if os.path.isfile(pre_save_path):
    #             compute_mc = False

    if compute_mc:

        seg_id = 0

        feature_list = ['raw', 'prob', 'reg']

        ds_train = [load_dataset(train_folder, name) for name in ds_train_names if name != ds_test_name]
        ds_test = load_dataset(meta_folder, ds_test_name)

        mc_nodes, _, _, _ = multicut_workflow(
            ds_train,
            ds_test,
            seg_id, seg_id,
            feature_list
        )

        segmentation = ds_test.project_mc_result(seg_id, mc_nodes)

        # Save in case sth in the following function goes wrong
        if pre_save_path is not None:
            vigra.writeHDF5(segmentation, pre_save_path, results_name, compression='gzip')

    if merge_segments:

        if not compute_mc:
            assert pre_save_path is not None, 'Investigate code, this must not happen!'
            segmentation = vigra.readHDF5(pre_save_path, results_name)

        # # Relabel with connected components
        # segmentation = vigra.analysis.labelVolume(segmentation.astype('uint32'))
        # Merge small segments
        segmentation = merge_small_segments(segmentation.astype('uint32'), 100)

        # Store the final result
        vigra.writeHDF5(segmentation, save_path, results_name, compression = 'gzip')


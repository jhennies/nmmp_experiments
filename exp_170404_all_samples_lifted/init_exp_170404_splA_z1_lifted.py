
# import multicut_src
# import numpy as np
# import vigra

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import MetaSet
from multicut_src import DataSet

import os

# Cache folder for training and test datasets
result_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170404_all_samples_lifted/170404_splA_z1_lifted/'
cache_folder = result_folder + 'cache/'
os.mkdir(result_folder)

# Other sources
raw_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
raw_file = 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5'
raw_key_train = 'z/0/raw'
raw_key_test = 'z/1/raw'

probs_path = raw_path
probs_file = 'cremi.splA.train.probs.crop.axes_xyz.split_z.h5'
probs_key_train = 'z/0/data'
probs_key_test = 'z/1/data'

seg_path = raw_path
seg_file = 'cremi.splA.train.wsdt_relabel.crop.axes_xyz.split_z.h5'
seg_key_train = 'z/0/labels'
seg_key_test = 'z/1/labels'

gt_path = raw_path
gt_file = 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5'
gt_key_train = 'z/0/neuron_ids'
gt_key_test = 'z/1/neuron_ids'


# Create datasets
meta = MetaSet(cache_folder)
ds_train = DataSet(cache_folder, 'splA_z0')
ds_test = DataSet(cache_folder, 'splA_z1')

# Add raw data
ds_train.add_raw(raw_path + raw_file, raw_key_train)
ds_test.add_raw(raw_path + raw_file, raw_key_test)
# Add probabilities
ds_train.add_input(probs_path + probs_file, probs_key_train)
ds_test.add_input(probs_path + probs_file, probs_key_test)
# Add segmentation (Superpixels)
ds_train.add_seg(seg_path + seg_file, seg_key_train)
ds_test.add_seg(seg_path + seg_file, seg_key_test)
# Add ground truth
ds_train.add_gt(gt_path + gt_file, gt_key_train)

# add cutouts for lifted multicut training
shape = ds_train.shape
z_offset = 10
ds_train.make_cutout([0, shape[0], 0, shape[1], 0, z_offset])
ds_train.make_cutout([0, shape[0], 0, shape[1], z_offset, shape[2] - z_offset])
ds_train.make_cutout([0, shape[0], 0, shape[1], shape[2] - z_offset, shape[2]])

# Add datasets
meta.add_dataset('splA_z0', ds_train)
meta.add_dataset('splA_z1', ds_test)

meta.save()




# import multicut_src
# import numpy as np
# import vigra

from multicut_src import MetaSet
from multicut_src import DataSet

# Cache folder for training and test datasets
cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170224_test/cache/'

# Other sources
raw_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
raw_file = 'cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
raw_key_train = 'z/0/raw'
raw_key_test = 'z/1/raw'

probs_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
probs_file = 'cremi.splB.train.probs.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
probs_key_train = 'z/0/data'
probs_key_test = 'z/1/data'

seg_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
seg_file = 'cremi.splB.train.mcseg_betas.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
seg_key_train = 'z/0/beta_0.3'
seg_key_test = 'z/1/beta_0.3'

gt_path = '/mnt/localdata01/jhennies/neuraldata/cremi_2016/resolve_merges/'
gt_file = 'cremi.splB.raw_neurons.crop.axes_xyz.crop_x100-612_y100-612.split_z.h5'
gt_key_train = 'z/0/neuron_ids'
gt_key_test = 'z/1/neuron_ids'

# Create datasets
meta = MetaSet(cache_folder)
ds_train = DataSet(cache_folder, 'ds_train')
ds_test = DataSet(cache_folder, 'ds_test')

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

# Add datasets
meta.add_dataset('ds_train', ds_train)
meta.add_dataset('ds_test', ds_test)

meta.save()



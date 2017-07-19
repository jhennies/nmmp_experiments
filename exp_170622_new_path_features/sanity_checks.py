
import cPickle as pickle
import vigra
import numpy as np
import os

experiment = '170622_new_path_features'
sample = 'A'
half = 0
defect_correct = ''

paths_path = '/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/{}/cache/spl{}_z{}/path_data/'.format(
    experiment,
    sample,
    half
)

path_prediction_filepath = os.path.join(paths_path, 'false_paths_predictions.pkl')
paths_filepath = os.path.join(paths_path, 'paths_ds_spl{}_z{}.h5'.format(sample, half))

with open(path_prediction_filepath, mode='r') as f:
    path_prediction = pickle.load(f)

all_paths = vigra.readHDF5(paths_filepath, 'all_paths')
paths_to_objs = vigra.readHDF5(paths_filepath, 'paths_to_objs')

if all_paths.size:
    all_paths = np.array([path.reshape((len(path) / 3, 3)) for path in all_paths])

# Get raw data
source_path = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges'
raw_filepath = os.path.join(source_path, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defect_correct))
raw = vigra.readHDF5(raw_filepath, 'z/{}/raw'.format(half))

paths_im = np.zeros(raw.shape, dtype='uint32')

for path_id, path in enumerate(all_paths):
    paths_im[path[:, 0], path[:, 1], path[:, 2]] = 1

paths_im_filepath = os.path.join(paths_path, 'paths_im.h5')

vigra.writeHDF5(paths_im, paths_im_filepath, 'data')



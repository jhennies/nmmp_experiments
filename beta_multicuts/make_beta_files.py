import vigra
import os
import numpy as np

import h5py

experiment_folder = '/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170721_mc_for_betas/'
samples = ['A', 'A', 'B', 'B', 'C', 'C']
halves = [0, 1] * 3

from run_mc_all import betas

def print_item(name, node):
    print name

target_path = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/'


for spl_id, spl in enumerate(samples):

    half = halves[spl_id]

    print '========='
    print spl, half
    print '---------'

    with h5py.File(os.path.join(experiment_folder, 'spl{}_z{}'.format(spl, half), 'result.h5'), 'r') as f:
        f.visititems(print_item)

    result_filepath = os.path.join(experiment_folder, 'spl{}_z{}'.format(spl, half), 'result.h5')

    mc_segs = {}
    for beta in betas:

        print beta

        im = vigra.readHDF5(result_filepath, 'z/{}/{}'.format(half, beta))

        target_filepath = os.path.join(target_path, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(spl))
        target_key = 'z/{}/beta_{}'.format(half, beta)
        vigra.writeHDF5(im, target_filepath, target_key, compression='gzip')


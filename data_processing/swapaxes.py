import vigra
import numpy as np
import os

source_folder = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
target_folder = '/mnt/ssd/jhennies/neuraldata/cremi_2016/170606_resolve_false_merges/'

samples = ['A', 'A', 'B', 'B', 'C', 'C']
halves = [0, 1] * 3
defect_corrects = [False, False, True, True, True, True]

for spl_id, sample in enumerate(samples):

    half = halves[spl_id]
    defect_correct = defect_corrects[spl_id]

    print 'Working on sample{}_z{}'.format(sample, half)

    if defect_correct:
        defcor = '_defect_correct'
    else:
        defcor = ''

    betas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

    mc_seg_file_s = os.path.join(source_folder, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(sample))
    mc_seg_file_t = os.path.join(target_folder, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(sample))

    for beta in betas:
        im = vigra.readHDF5(mc_seg_file_s, 'z/{}/beta_{}'.format(half, beta))
        im = np.swapaxes(im, 0, 2)
        vigra.writeHDF5(im, mc_seg_file_t, 'z/{}/beta_{}'.format(half, beta), compression = 'gzip')

    probs_file_s = os.path.join(source_folder, 'cremi.spl{}.train.probs{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))
    probs_file_t = os.path.join(target_folder, 'cremi.spl{}.train.probs{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))

    im = vigra.readHDF5(probs_file_s, 'z/{}/data'.format(half))
    im = np.swapaxes(im, 0, 2)
    vigra.writeHDF5(im, probs_file_t, 'z/{}/data'.format(half), compression = 'gzip')

    raw_file_s = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))
    raw_file_t = os.path.join(target_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))

    im = vigra.readHDF5(raw_file_s, 'z/{}/raw'.format(half))
    im = np.swapaxes(im, 0, 2)
    vigra.writeHDF5(im, raw_file_t, 'z/{}/raw'.format(half), compression = 'gzip')

    im = vigra.readHDF5(raw_file_s, 'z/{}/neuron_ids'.format(half))
    im = np.swapaxes(im, 0, 2)
    vigra.writeHDF5(im, raw_file_t, 'z/{}/neuron_ids'.format(half), compression = 'gzip')

    seg_file_s = os.path.join(source_folder, 'cremi.spl{}.train.wsdt_relabel{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))
    seg_file_t = os.path.join(target_folder, 'cremi.spl{}.train.wsdt_relabel{}.crop.axes_xyz.split_z.h5'.format(sample, defcor))

    im = vigra.readHDF5(seg_file_s, 'z/{}/labels'.format(half))
    im = np.swapaxes(im, 0, 2)
    vigra.writeHDF5(im, seg_file_t, 'z/{}/labels'.format(half), compression = 'gzip')






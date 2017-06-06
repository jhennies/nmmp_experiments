
import volumina_viewer

import vigra
import os

sample = 'B'
half = 0
defect_correct = '_defect_correct'
# defect_correct = ''

source_folder = '/media/julian/Daten/datasets/cremi_2016/170321_resolve_false_merges/'
result_folder = '/media/julian/Daten/datasets/results/multicut_workflow/170530_new_baseline/'

seg = vigra.readHDF5(os.path.join(result_folder, 'spl{}_z{}/result.h5'.format(sample, half)),
                     'z/{}/data'.format(half))

raw = vigra.readHDF5(
    os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defect_correct)),
    'z/{}/raw'.format(half)
)

print seg.shape
print raw.shape

volumina_viewer.volumina_n_layer([raw.astype('float32'), seg], ['raw', 'seg'])
# volumina_viewer.volumina_flexible_layer([im], ['RandomColors'])

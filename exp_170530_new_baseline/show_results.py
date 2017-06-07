
import volumina_viewer

import vigra
import os


def result_and_raw():
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


def check_paths():

    sample = 'B'
    half = 0
    defect_correct = '_defect_correct'

    source_folder = '/media/julian/Daten/datasets/cremi_2016/170321_resolve_false_merges/'
    result_folder = '/home/julian/ssh_data/neuraldata/results/multicut_workflow/170530_new_baseline/'

    paths_folder = os.path.join(
        result_folder,
        'cache',
        'spl{}_z{}'.format(sample, half),
        'path_data'
    )

    paths_file = os.path.join(paths_folder, 'paths_ds_spl{}_z{}.h5'.format(sample, half))

    # import h5py
    #
    # def print_it(name):
    #     dset = f[name]
    #     print(dset)
    #     # print(type(dset))
    #
    # with h5py.File(paths_file, 'r') as f:
    #     f.visit(print_it)

    all_paths = vigra.readHDF5(paths_file, 'all_paths')
    paths_to_objs = vigra.readHDF5(paths_file, 'paths_to_objs')

    # Randomly select object


    # Display object and all paths within




if __name__ == '__main__':
    check_paths()
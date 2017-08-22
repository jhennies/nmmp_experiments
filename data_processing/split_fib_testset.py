
import h5py
import numpy as np
import os
import vigra


def split_data(filepath, datakey, target_filepath, section_shape):

    section_shape = np.array(section_shape)

    # Load data as file object
    f_in = h5py.File(filepath, 'r')

    # Determine the number of sections in each dimension
    total_shape = np.array(f_in[datakey].shape)
    section_counts = (total_shape / section_shape).astype('uint32')

    for x in xrange(0, section_counts[0]):
        print '{}: x = {}/{}'.format(filepath, x, section_counts[0])
        for y in xrange(0, section_counts[1]):
            for z in xrange(0, section_counts[2]):

                # print '{}, {}, {}'.format(x, y, z)

                # Load a data section into memory
                position_from = np.array([x, y, z]) * section_shape
                position_to = np.array([x + 1, y + 1, z + 1]) * section_shape
                section_s = np.s_[
                            position_from[0]:position_to[0],
                            position_from[1]:position_to[1],
                            position_from[2]:position_to[2]
                            ]
                section_data = f_in[datakey][section_s]

                # Store the data section to a new file
                #    Generate datakey name
                target_datakey = '{}_{}_{}_{}'.format(datakey, x, y, z)
                #    Write data
                # with h5py.File(target_filepath, 'w') as f_out:
                #     f_out.create_dataset(target_datakey, data=section_data, compression='gzip')
                vigra.writeHDF5(section_data, target_filepath, target_datakey, compression='gzip')


def split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list):

    section_shape = np.array(section_shape)

    # Load data as file object
    f_in = h5py.File(filepath, 'r')

    # Determine the number of sections in each dimension
    total_shape = np.array(f_in[datakey].shape)
    section_counts = (total_shape / section_shape).astype('uint32')

    for chunk in chunk_list:

        print '{}: chunk = {}'.format(filepath, chunk)

        x = chunk[0]
        y = chunk[1]
        z = chunk[2]

        # print '{}, {}, {}'.format(x, y, z)

        # Load a data section into memory
        position_from = np.array([x, y, z]) * section_shape
        position_to = np.array([x + 1, y + 1, z + 1]) * section_shape
        section_s = np.s_[
                    position_from[0]:position_to[0],
                    position_from[1]:position_to[1],
                    position_from[2]:position_to[2]
                    ]
        section_data = f_in[datakey][section_s]

        # Store the data section to a new file
        #    Generate datakey name
        target_datakey = '{}_{}_{}_{}'.format(datakey, x, y, z)
        #    Write data
        # with h5py.File(target_filepath, 'w') as f_out:
        #     f_out.create_dataset(target_datakey, data=section_data, compression='gzip')
        vigra.writeHDF5(section_data, target_filepath, target_datakey, compression='gzip')


if __name__ == '__main__':

    # FIXME: Do this only for the ones I select!
    chunk_list = [
        [8, 5, 6],
        [8, 5, 7],
        [13, 5, 5],
        [13, 5, 6]
    ]

    # # Raw
    # filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-raw.h5'
    # datakey = 'data'
    # target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25_raw_chunks_selected.h5'
    # section_shape = (512, 512, 512)
    #
    # split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list)

    # GT
    filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25-gt.h5'
    datakey = 'data'
    target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25_gt_chunks_selected.h5'
    section_shape = (512, 512, 512)

    split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list)

    # # Probs
    # filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-cantor-pmap.h5'
    # datakey = 'data'
    # target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25_cantor_pmap_chunks.h5'
    # section_shape = (512, 512, 512)
    #
    # split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list)
    #
    # # Superpixels
    # filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-vanilla-watershed-relabeled.h5'
    # datakey = 'data'
    # target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25_vanilla_watershed_relabeled_chunks.h5'
    # section_shape = (512, 512, 512)
    #
    # split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list)

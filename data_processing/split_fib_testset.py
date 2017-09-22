
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


def split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list,
                               relabel=False, normalize=False, conncomp=False, overlap=0):

    section_shape = np.array(section_shape)

    # Load data as file object
    f_in = h5py.File(filepath, 'r')

    # # Determine the number of sections in each dimension
    # total_shape = np.array(f_in[datakey].shape)
    # section_counts = (total_shape / section_shape).astype('uint32')

    for chunk in chunk_list:

        print '{}: chunk = {}'.format(filepath, chunk)

        x = chunk[0]
        y = chunk[1]
        z = chunk[2]

        # print '{}, {}, {}'.format(x, y, z)

        # Load a data section into memory
        position_from = np.array([x, y, z]) * (section_shape - (overlap,)*3)
        position_to = np.array([x + 1, y + 1, z + 1]) * (section_shape - (overlap,)*3) + (overlap,)*3
        section_s = np.s_[
                    position_from[0]:position_to[0],
                    position_from[1]:position_to[1],
                    position_from[2]:position_to[2]
                    ]
        section_data = f_in[datakey][section_s]

        if conncomp:
            print 'Connected components...'
            section_data = vigra.analysis.labelMultiArray(section_data.astype('uint32'))

        if relabel:
            print 'Relabeling...'
            section_data = vigra.analysis.relabelConsecutive(
                section_data,
                start_label=0, keep_zeros=False
            )[0]

        if normalize:
            print 'Normalizing ...'
            print 'Old min: {}'.format(section_data.min())
            print 'Old max: {}'.format(section_data.max())
            section_data = section_data - section_data.min()
            section_data = section_data / float(section_data.max())
            print 'New min: {}'.format(section_data.min())
            print 'New max: {}'.format(section_data.max())
            section_data = section_data.astype('float32')

        # Store the data section to a new file
        #    Generate datakey name
        target_datakey = '{}_{}_{}_{}'.format(datakey, x, y, z)
        #    Write data
        with h5py.File(target_filepath, 'a') as f_out:
            f_out.create_dataset(target_datakey, data=section_data, compression='gzip')
            f_out[target_datakey].attrs.create('Location', [position_from, position_to])
        # vigra.writeHDF5(section_data, target_filepath, target_datakey, compression='gzip')


if __name__ == '__main__':

    # FIXME: Do this only for the ones I select!
    # chunk_list = [
    #     [8, 5, 6],
    #     [8, 5, 7],
    #     [13, 5, 5],
    #     [13, 5, 6]
    # ]

    # chunk_list = [
    #     [7, 5, 6],
    #     [7, 5, 7],
    #     [8, 5, 6],
    #     [8, 5, 7]
    # ]
    chunk_list = [
        [5, 6, 7],
        [5, 6, 8],
        [6, 6, 7],
        [6, 6, 8]
    ]
    # chunk_list = [
    #     [7, 5, 6]
    # ]

    section_shape = (512, 512, 512)
    overlap = 50

    # Raw
    filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-raw.h5'
    datakey = 'data'
    target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/overlap_50/fib25_raw_chunks_selected_olap50.h5'

    split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list,
                               overlap=overlap, normalize=True)

    # GT
    filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25-gt.h5'
    datakey = 'data'
    target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/overlap_50/fib25_gt_chunks_selected_olap50.h5'

    split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list,
                               overlap=overlap, conncomp=True, relabel=True)

    # Probs
    filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-membrane-predictions_squeezed.h5'
    datakey = 'data'
    target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/overlap_50/fib25_membrane_predictions_chunks_selected_olap50.h5'

    split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list,
                               overlap=overlap, normalize=True)

    # Superpixels
    filepath = '/net/hci-storage01/groupfolders/multbild/fib25_hdf5/fib25-vanilla-watershed-relabeled.h5'
    datakey = 'data'
    target_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/overlap_50/fib25_vanilla_watershed_relabeled_chunks_selected_olap50.h5'

    split_data_selected_chunks(filepath, datakey, target_filepath, section_shape, chunk_list,
                               overlap=overlap, conncomp=True, relabel=True)

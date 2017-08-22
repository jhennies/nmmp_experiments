
import numpy as np
import h5py
import vigra


def search_chunks(filepath, data_locations_filepath):

    chunk_numbers = []

    def count_chunks(name, node):

        chunk_numbers.append(np.array(name.split('_')[1:]).astype('uint32'))

    with h5py.File(filepath, 'r') as f:
        f.visititems(count_chunks)

    max_chunks = np.array(chunk_numbers).max(axis=0)
    data_locations = np.zeros(max_chunks + np.array([1, 1, 1]))

    def evaluate_chunks(name, node):
        node_data = np.array(node)
        coords = np.array(name.split('_')[1:]).astype('uint32')

        # u, c = np.unique(node_data, return_counts=True)
        node_data_max = node_data.max()
        if node_data_max:

            num_zeros = np.prod(node_data.shape) - len(node_data.nonzero()[0])

            print '{}: {}: {}'.format(name, node_data_max, num_zeros)

            if num_zeros < 10 and num_zeros > 0:
                data_locations[tuple(coords.tolist())] = 2

            elif num_zeros == 0:
                data_locations[tuple(coords.tolist())] = 3

            else:
                data_locations[tuple(coords.tolist())] = 1

    with h5py.File(filepath, 'r') as f:
        f.visititems(evaluate_chunks)

    vigra.writeHDF5(data_locations, data_locations_filepath, 'data', compression='gzip')


if __name__ == '__main__':

    filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/fib25_raw_chunks.h5'
    data_locations_filepath = '/mnt/localdata1/jhennies/neuraldata/fib25/data_locations.h5'
    search_chunks(filepath, data_locations_filepath)
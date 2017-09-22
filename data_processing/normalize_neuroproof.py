import numpy as np
import vigra


def normalize_data(
        filepath,
        datakey,
        target_filepath,
        compression=None
):

    data = vigra.readHDF5(filepath, datakey)

    print 'Normalizing ...'
    print 'Old min: {}'.format(data.min())
    print 'Old max: {}'.format(data.max())
    data = data - data.min()
    data = data / float(data.max())
    print 'New min: {}'.format(data.min())
    print 'New max: {}'.format(data.max())
    data = data.astype('float32')

    vigra.writeHDF5(data, target_filepath, datakey, compression=compression)


if __name__ == '__main__':

    # Raw test
    filepath = '/export/home/jhennies/sshs/neuroproof_data/raw_test.h5'
    datakey = 'data'
    target_filepath = '/export/home/jhennies/sshs/neuroproof_data_normalized/raw_test.h5'

    normalize_data(filepath, datakey, target_filepath)

    # Raw train
    filepath = '/export/home/jhennies/sshs/neuroproof_data/raw_train.h5'
    datakey = 'data'
    target_filepath = '/export/home/jhennies/sshs/neuroproof_data_normalized/raw_train.h5'

    normalize_data(filepath, datakey, target_filepath)

    # Probs test
    filepath = '/export/home/jhennies/sshs/neuroproof_data/probabilities_test.h5'
    datakey = 'data'
    target_filepath = '/export/home/jhennies/sshs/neuroproof_data_normalized/probabilities_test.h5'

    normalize_data(filepath, datakey, target_filepath)

    # Probs train
    filepath = '/export/home/jhennies/sshs/neuroproof_data/probabilities_train.h5'
    datakey = 'data'
    target_filepath = '/export/home/jhennies/sshs/neuroproof_data_normalized/probabilities_train.h5'

    normalize_data(filepath, datakey, target_filepath)

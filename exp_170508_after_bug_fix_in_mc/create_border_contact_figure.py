
from multicut_src.false_merges.compute_border_contacts import compute_border_contacts_old
from multicut_src import load_dataset
import os
import vigra

from multicut_src import ExperimentSettings

from pipeline import find_false_merges

# TODO Change here
from init_exp_splB_z0 import meta_folder, project_folder, source_folder, experiment_folder
from run_mc_splB_z0 import result_key

# Path folders
test_paths_cache_folder = os.path.join(meta_folder, 'path_data')
train_paths_cache_folder = os.path.join(project_folder, 'train_paths_cache')


if __name__ == '__main__':

    # TODO Change here
    from init_exp_splB_z0 import test_name
    from run_mc_splB_z0 import rf_cache_folder

    test_seg_filepath = experiment_folder + 'result.h5'
    seg = vigra.readHDF5(test_seg_filepath, 'z/0/data')[0: 300, 0:300, :]

    ds_test = load_dataset(meta_folder, test_name)

    from matplotlib import pyplot as plt

    dt = ds_test.inp(2)[0: 300, 0:300, :]

    compute_border_contacts_old(seg, dt)




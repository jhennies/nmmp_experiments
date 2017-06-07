
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import run_lifted_mc

# TODO Change here
from init_exp_splA_z1 import meta_folder, experiment_folder
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

# TODO Change here when switching half
result_key = 'z/1/data'


if __name__ == '__main__':

    # TODO Change here
    from init_exp_splA_z1 import test_name, train_names, train_folder

    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 10.
    ExperimentSettings().use_2d = False
    ExperimentSettings().n_threads = 30
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'z'
    ExperimentSettings().lifted_neighborhood = 3

    run_lifted_mc(
        meta_folder,
        train_folder,
        train_names,
        test_name,
        experiment_folder + 'result.h5',
        result_key,
        pre_save_path=experiment_folder + 'pre_result.h5'
    )

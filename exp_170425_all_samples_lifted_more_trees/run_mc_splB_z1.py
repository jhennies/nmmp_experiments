
import os

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import run_lifted_mc

from init_exp_splB_z1 import meta_folder, experiment_folder
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')


if __name__ == '__main__':

    from init_exp_splB_z1 import test_name, train_name

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
        train_name,
        test_name,
        experiment_folder + 'result.h5',
        'z/1/data'
    )
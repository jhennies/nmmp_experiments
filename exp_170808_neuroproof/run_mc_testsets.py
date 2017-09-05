
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import run_lifted_mc, run_mc

from init_testsets import meta_folder
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

# experiment_sets = ['splB_z0']
# betas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

if __name__ == '__main__':

    # TODO Change here
    from init_testsets import mc_trainset_names
    from init_testsets import testset_names, project_folder, result_keys, experiment_ids

    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().use_2d = False
    ExperimentSettings().use_2rfs = False
    ExperimentSettings().n_threads = 120
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'all'
    ExperimentSettings().lifted_neighborhood = 3
    ExperimentSettings().rf_batch_size = 5000000

    for ds_id in experiment_ids:

        result_key = result_keys[ds_id]
        testset_name = testset_names[ds_id]

        result_folder = os.path.join(project_folder, testset_name)
        if not os.path.exists(result_folder):
            os.mkdir(result_folder)

        run_lifted_mc(
            meta_folder,
            meta_folder,
            mc_trainset_names,
            testset_name,
            os.path.join(result_folder, 'result.h5'),
            result_key,
            pre_save_path=os.path.join(result_folder, 'pre_result.h5')
        )

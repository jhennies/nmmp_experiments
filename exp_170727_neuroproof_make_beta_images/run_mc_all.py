
import os

import sys
sys.path.append(
    '/home/jhennies/src/nature_methods_multicut_pipeline/nature_methods_multicut_pipeline/software/')

from multicut_src import ExperimentSettings

from pipeline import run_mc

from init_datasets import meta_folder
rf_cache_folder = os.path.join(meta_folder, 'rf_cache')

# experiment_sets = ['splB_z0']
betas = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

if __name__ == '__main__':

    # TODO Change here
    from init_datasets import ds_names, project_folder, result_keys, experiment_ids

    ExperimentSettings().rf_cache_folder = rf_cache_folder
    ExperimentSettings().anisotropy_factor = 1.
    ExperimentSettings().use_2d = False
    ExperimentSettings().use_2rfs = False
    ExperimentSettings().n_threads = 20
    ExperimentSettings().n_trees = 500
    ExperimentSettings().solver = 'multicut_fusionmoves'
    ExperimentSettings().verbose = True
    ExperimentSettings().weighting_scheme = 'all'
    ExperimentSettings().lifted_neighborhood = 3

    for ds_id in experiment_ids:

        for beta in betas:

            ExperimentSettings().beta_local = beta

            result_key = result_keys[ds_id]
            result_key = result_key.format(beta)
            ds_name = ds_names[ds_id]

            result_folder = os.path.join(project_folder, ds_name)
            if not os.path.exists(result_folder):
                os.mkdir(result_folder)

            run_mc(
                meta_folder,
                meta_folder,
                ds_names,
                ds_name,
                os.path.join(result_folder, 'result.h5'),
                result_key,
                pre_save_path=os.path.join(result_folder, 'pre_result.h5')
            )

import vigra
import os

from multicut_src import lifted_multicut_workflow
from multicut_src import ExperimentSettings, load_dataset


def run_lifted_mc(
        meta_folder,
        ds_train_name,
        ds_test_name,
        save_path,
        results_name
):
    assert os.path.exists(os.path.split(save_path)[0]), "Please choose an existing folder to save your results"

    seg_id = 0

    feature_list = ['raw', 'prob', 'reg']
    feature_list_lifted = ['cluster', 'reg']

    gamma = 2.

    ds_train = load_dataset(meta_folder, ds_train_name)
    ds_test = load_dataset(meta_folder, ds_test_name)

    mc_nodes, _, _, _ = lifted_multicut_workflow(
        ds_train, ds_test,
        seg_id, seg_id,
        feature_list, feature_list_lifted,
        gamma=gamma
    )

    segmentation = ds_test.project_mc_result(seg_id, mc_nodes)
    vigra.writeHDF5(segmentation, save_path, results_name, compression = 'gzip')


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
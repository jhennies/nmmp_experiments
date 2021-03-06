
from multicut_src import pre_compute_paths
from multicut_src import MetaSet
from multicut_src import DataSet
from multicut_src import ExperimentSettings

cache_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170404_all_samples_lifted/paths_cache/'
source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

def compute_train_paths():

    gtruths_paths = [
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
        source_folder + 'cremi.splA.train.raw_neurons.crop.axes_xyz.split_z.h5',
    ]
    gtruths_keys = [
        'z/0/neuron_ids',
        'z/1/neuron_ids',
    ]

    ds_names = [
        'splA_z0',
        'splA_z1'
    ]

    trainsets = []
    for id_source, gt_source in enumerate(gtruths_paths):
        trainsets.append(
            DataSet(
                cache_folder, 'ds_train_{}'.format(ds_names[id_source])
            )
        )
        # trainsets[-1].add_raw(raw_source, train_raw_sources_keys[id_source])
        # trainsets[-1].add_input(train_probs_sources[id_source], train_probs_sources_keys[id_source])
        trainsets[-1].add_gt(gt_source, gtruths_keys[id_source])

    mc_segs = [
        [source_folder + 'cremi.splA.train.mcseg_betas.crop.axes_xyz.split_z.h5'] * 9
    ] * 2

    mc_segs_keys = [
        ['z/0/beta_0.5', 'z/0/beta_0.45', 'z/0/beta_0.55', 'z/0/beta_0.4', 'z/0/beta_0.6', 'z/0/beta_0.35', 'z/0/beta_0.65', 'z/0/beta_0.3', 'z/0/beta_0.7'],
        ['z/1/beta_0.5', 'z/1/beta_0.45', 'z/1/beta_0.55', 'z/1/beta_0.4', 'z/1/beta_0.6', 'z/1/beta_0.35', 'z/1/beta_0.65', 'z/1/beta_0.3', 'z/1/beta_0.7']
    ]

    params = ExperimentSettings()
    params.set_anisotropy(10.)

    paths_save_folder = cache_folder + 'train/'

    pre_compute_paths(
        trainsets,
        mc_segs,
        mc_segs_keys,
        params,
        paths_save_folder
    )

if __name__ == '__main__':

    compute_train_paths()
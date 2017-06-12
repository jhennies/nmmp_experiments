from multicut_src.false_merges.compute_paths_and_features import multicut_path_features
from multicut_src import load_dataset

# from init_datasets import meta_folder
meta_folder = '/export/home/jhennies/ssh_results/neuraldata/results/multicut_workflow/170530_new_baseline/cache/'
# from init_datasets import source_folder
source_folder = '/export/home/jhennies/ssh_results/neuraldata/cremi_2016/170321_resolve_false_merges/'

import vigra
import os

ds = load_dataset(meta_folder, ds_name='splB_z0')
seg_id = 0

seg_filepath = os.path.join(source_folder, 'cremi.splB.train.mcseg_betas.crop.axes_xyz.split_z.h5')
mc_segmentation = vigra.readHDF5(seg_filepath, 'z/0/beta_0.5')

paths_to_objs = vigra.readHDF5(
    os.path.join(meta_folder, 'splB_z0/path_data/', 'paths_ds_splB_z0.h5'),
    'paths_to_objs'
)
all_paths = vigra.readHDF5(
    os.path.join(meta_folder, 'splB_z0/path_data/', 'paths_ds_splB_z0.h5'),
    'all_paths'
)


def paths_to_objs_to_objs_to_paths(paths_to_objs, paths):

    objs_to_paths = {}
    for obj_id, obj in enumerate(paths_to_objs):

        if obj in objs_to_paths.keys():
            objs_to_paths[obj][obj_id] = paths[obj_id]
        else:
            objs_to_paths[obj] = {obj_id: paths[obj_id]}

    return objs_to_paths


objs_to_paths = paths_to_objs_to_objs_to_paths(paths_to_objs, all_paths)

edge_probabilities = vigra.readHDF5(
    os.path.join(meta_folder, 'splB_z0/probs_to_energies_0_z_16.0_0.5_rawprobreg.h5'),
    'data'
)

mc_path_features = multicut_path_features(
    ds,
    seg_id,
    mc_segmentation,
    objs_to_paths,  # dict[merge_ids : dict[path_ids : paths]]
    edge_probabilities
)



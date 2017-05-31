import vigra
import os
from multicut_src import merge_small_segments

project_folder = '/media/julian/Daten/datasets/results/multicut_workflow/170530_new_baseline/'
# TODO Change here
experiment_folder = os.path.join(project_folder, 'splA_z0/')

# TODO Change here when switching half
result_key = 'z/0/data'

segmentation = vigra.readHDF5(experiment_folder + 'pre_result.h5', result_key)

save_path = experiment_folder + 'result.h5'

print segmentation.shape
print segmentation.dtype
print type(segmentation)

# Relabel with connected components
segmentation = vigra.analysis.labelVolume(segmentation.astype('uint32'))
# Merge small segments
segmentation = merge_small_segments(segmentation, 100)

# Store the final result
vigra.writeHDF5(segmentation, save_path, result_key, compression='gzip')
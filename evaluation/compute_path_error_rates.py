
import pickle
import numpy as np

results_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170329_test_pipeline_update/'
paths_folder = results_folder + 'cache/path_data/'

# Load paths of false merge computation
with open(paths_folder + 'false_paths_predictions.pkl', mode='r') as f:
    false_paths_predictions = pickle.load(f)

with open(paths_folder + 'path_splB_z1.pkl', mode='r') as f:
    path_data = pickle.load(f)

paths = path_data['paths']
paths_to_objs = path_data['paths_to_objs']

# TODO: Evaluation: Do this for different probs-thresholds
# Determine values along the path
pass

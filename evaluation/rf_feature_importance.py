
import cPickle as pickle
import os
import numpy as np
from copy import deepcopy

feature_lists = ['all_path_features', 'old_path_features', 'new_path_features', 'lengths']

# feature_list = feature_lists[1]
#
# with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z1_splB_z0_splB_z1_splC_z0_splC_z1/rf.pkl'.format(feature_list), 'r') as f:
#     rf_a0 = pickle.load(f)
# #
# # with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z0_splB_z0_splB_z1_splC_z0_splC_z1/rf.pkl'.format(feature_list), 'r') as f:
# #     rf_a1 = pickle.load(f)
# #
# # with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z0_splA_z1_splB_z1_splC_z0_splC_z1/rf.pkl'.format(feature_list), 'r') as f:
# #     rf_b0 = pickle.load(f)
# #
# # with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z0_splA_z1_splB_z0_splC_z0_splC_z1/rf.pkl'.format(feature_list), 'r') as f:
# #     rf_b1 = pickle.load(f)
# #
# # with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z0_splA_z1_splB_z0_splB_z1_splC_z1/rf.pkl'.format(feature_list), 'r') as f:
# #     rf_c0 = pickle.load(f)
# #
# # with open('/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/cache/rf_cache/rf_bkps/{}/rf_merges_splA_z0_splA_z1_splB_z0_splB_z1_splC_z0/rf.pkl'.format(feature_list), 'r') as f:
# #     rf_c1 = pickle.load(f)
#
# fis_a0 = rf_a0.feature_importances_
# # fis_a1 = rf_a1.feature_importances_
# # fis_b0 = rf_b0.feature_importances_
# # fis_b1 = rf_b1.feature_importances_
# # fis_c0 = rf_c0.feature_importances_
# # fis_c1 = rf_c1.feature_importances_
#
# # Print the feature ranking
# print("Feature ranking:")

experiment_folder = '/mnt/localdata1/jhennies/neuraldata/results/multicut_workflow/170717_new_path_features/'
samples = ['splA_z0', 'splA_z1', 'splB_z0', 'splB_z1', 'splC_z0', 'splC_z1']

for feature_list in feature_lists:

    print '==========================='
    print feature_list
    prediction_path = os.path.join(experiment_folder, 'cache/rf_cache/rf_bkps/{}/'.format(feature_list))

    for sample in samples:

        other_samples = deepcopy(samples)
        other_samples.remove(sample)

        rf_path = os.path.join(
            prediction_path,
            'rf_merges_' + '_'.join(other_samples),
            'rf.pkl'
        )

        # print 'Loading random forest for sample {} from:'.format(sample)
        # print rf_path

        with open(rf_path, 'r') as f:
            rf = pickle.load(f)

        importances = rf.feature_importances_

        # print 'The first ten most important features:'

        important_ids = np.argsort(importances)
        important_vals = np.sort(importances)

        # print important_vals[::-1][:10]
        # print important_ids[::-1][:20]
        print important_ids[:20]







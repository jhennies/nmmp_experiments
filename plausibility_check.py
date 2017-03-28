import pickle
import numpy as np

folder = '/media/julian/Daten/datasets/results/multicut_workflow/170324_test_recompute/cache/debug/'

with open(folder + 'features_resolve_87.0.pkl', mode='r') as f:
    frs = pickle.load(f)
with open(folder + 'features_test.pkl', mode='r') as f:
    fts = pickle.load(f)
# with open(folder + 'fs_12.0.pkl', mode='r') as f:
#     frp = pickle.load(f)


for idt, ft in enumerate(fts):
    for idr, fr in enumerate(frs):
        if np.array_equal(sorted(fr), sorted(ft)):
            print np.array_equal(fr, ft)
            print np.array_equal(sorted(fr), sorted(ft))
            print '====='

print '\nChecking more in detail...\n'

print "fts.shape = {}".format(np.array(fts).shape)
print "fts.shape = {}".format(np.array(frs).shape)

for idt, ft in enumerate(fts):
    for idr, fr in enumerate(frs):
        c = 0
        for i in ft:
            for j in fr:
                if i == j:
                    c += 1
        if c > 30:
            print 'Equality count for IDs {}/{} = {}'.format(idt, idr, c)
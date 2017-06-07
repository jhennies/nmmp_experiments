
import vigra
import numpy

def defect_correct(filepath, target_filepath, data_key, slicemap):

    volume = vigra.readHDF5(filepath, data_key)

    for convert, to in slicemap.iteritems():

        volume[:, :, convert] = volume[:, :, to]

    vigra.writeHDF5(volume, target_filepath, data_key)

spl = 'C'

gt_filepath = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.spl{}.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'.format(spl)
save_filepath = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/cremi.spl{}.train.raw_neurons_defect_correct.crop.axes_xyz.split_z.h5'.format(spl)

# # For B0
# slicemap = {
#     15: 14,
#     16: 17,
#     44: 43,
#     45: 46
# }
#
# defect_correct(gt_filepath, save_filepath, 'z/0/neuron_ids', slicemap)

# # For B1
# slicemap = {
#     15: 14
# }
#
# defect_correct(gt_filepath, save_filepath, 'z/1/neuron_ids', slicemap)

# # For C0
# slicemap = {
#     14: 13
# }
#
# defect_correct(gt_filepath, save_filepath, 'z/0/neuron_ids', slicemap)

# For C1
slicemap = {
    12: 11,
    24: 23
}

defect_correct(gt_filepath, save_filepath, 'z/1/neuron_ids', slicemap)
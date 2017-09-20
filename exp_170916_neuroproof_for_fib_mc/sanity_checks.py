import vigra
import numpy as np

import sys
sys.path.append(
    '/export/home/jhennies/src/nature_methods_multicut_pipeline_devel/nature_methods_multicut_pipeline/software/')

from multicut_src import load_dataset


def check_ds_for_non_continuous_objects(meta_folder, ds_name, check_seg=True, check_gt=False):

    ds = load_dataset(meta_folder, ds_name)

    def check_seg_for_non_continuous_objects(segmentation):

        print type(segmentation)

        segmentation_conn = vigra.analysis.labelMultiArray(segmentation)
        segmentation = vigra.analysis.relabelConsecutive(segmentation)[0]
        segmentation_conn = vigra.analysis.relabelConsecutive(segmentation_conn)[0]

        print type(segmentation)
        print type(segmentation_conn)

        print '{}, {}'.format(segmentation.max(), segmentation_conn.max())


    if check_seg:
        check_seg_for_non_continuous_objects(ds.seg(0))

    if check_gt:
        check_seg_for_non_continuous_objects(ds.gt())


if __name__ == '__main__':

    ds_name = 'fib_8_5_7'
    meta_folder = '/mnt/ssd/jhennies/results/multicut_workflow/170916_neuroproof_for_fib_mc/cache/'
    check_ds_for_non_continuous_objects(meta_folder,
                                        ds_name,
                                        check_seg=True,
                                        check_gt=True)

    ds_name = 'fib_7_5_6'
    meta_folder = '/mnt/ssd/jhennies/results/multicut_workflow/170916_neuroproof_for_fib_mc/cache/'
    check_ds_for_non_continuous_objects(meta_folder,
                                        ds_name,
                                        check_seg=True,
                                        check_gt=True)




import cPickle as pickle
import vigra
import numpy as np
import os
from evaluation import compute_path_error_rates
from matplotlib import pyplot as plt


def path_eval_on_sample(sample, half, defect_correct, project_folder, thresh_range):

    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print '--------------------'

    if defect_correct:
        defect_correct_str = '_defect_correct'
    else:
        defect_correct_str = ''

    # Load stuff
    source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
    # TODO: Change here
    experiment_folder = os.path.join(project_folder, 'spl{}_z{}/'.format(sample, half))
    meta_folder = os.path.join(experiment_folder, 'cache/')

    test_name = 'spl{}_z{}'.format(sample, half)

    path_data_path = os.path.join(meta_folder, 'path_data')
    path_data_filepath = os.path.join(path_data_path, 'paths_ds_{}.h5'.format(test_name))

    # TODO Change here when switching sample
    gt_file = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5').format(sample, defect_correct_str)
    # TODO Change here when switching half
    gt_key = 'z/{}/neuron_ids'.format(half)
    gt = vigra.readHDF5(gt_file, gt_key)

    # Load paths
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(path_data_path, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    print 'Number of paths = {}'.format(len(paths_to_objs))
    print 'Number of objects = {}'.format(len(np.unique(paths_to_objs)))

    # Determine path error rates
    result_path, result_obj = compute_path_error_rates(
        paths_to_objs, paths, gt, false_merge_probs, thresh_range=thresh_range
    )

    return result_path, result_obj


def all_sample_path_eval(project_folder, thresh_range, samples, halves, defect_corrects, measures=None):

    if measures is None:
        measures = ['precision', 'recall', 'accuracy', 'f1']

    results_path = {}
    results_obj = {}
    for measure in measures:
        results_path[measure] = []
        results_obj[measure] = []

    for idx, sample in enumerate(samples):
        result_path, result_obj = path_eval_on_sample(sample, halves[idx], defect_corrects[idx], project_folder,
                                                      thresh_range)

        sorted_keys = sorted(result_path.keys())
        for key in results_path.keys():
            results_path[key].append(np.array([result_path[k][key] for k in sorted_keys])[:, None])
            results_obj[key].append(np.array([result_obj[k][key] for k in sorted_keys])[:, None])

    return results_path, results_obj


from cremi import Volume
from cremi.evaluation import NeuronIds


def roi_and_rand(
        sample, half, defect_correct, project_folder,
        sources=['Baseline', 'Global', 'Local'], debug=False):

    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print '--------------------'

    if defect_correct:
        defect_correct_str = '_defect_correct'
    else:
        defect_correct_str = ''

    mc_result_key = 'z/{}/data'.format(half)

    # Load stuff
    source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'
    experiment_folder = os.path.join(project_folder, 'spl{}_z{}/'.format(sample, half))

    # # TODO: Change here when switching sample
    # ref_result_filepath = os.path.join(source_folder, 'cremi.spl{}.train.mcseg_betas.crop.axes_xyz.split_z.h5'.format(sample))
    # # TODO: Change here when switching half
    # ref_result_key = 'z/{}/beta_0.5'.format(half)

    gt_filepath = os.path.join(source_folder, 'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(sample, defect_correct_str))
    gt_key = 'z/{}/neuron_ids'.format(half)

    # ref_result, _, _ = vigra.analysis.relabelConsecutive(vigra.readHDF5(ref_result_filepath, ref_result_key), start_label=1, keep_zeros=True)
    if not debug:
        gt = vigra.readHDF5(gt_filepath, gt_key)
        vol_gt = Volume(gt)
        neuron_ids_evaluation = NeuronIds(vol_gt)

    #
    # # Evaluate reference (CREMI submission)
    # vol_ref_result = Volume(ref_result)
    # (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_ref_result)
    # adapted_rand = neuron_ids_evaluation.adapted_rand(vol_ref_result)
    #
    # print 'Reference (CREMI submission)'
    # print "\tvoi split   : " + str(voi_split)
    # print "\tvoi merge   : " + str(voi_merge)
    # print "\tadapted RAND: " + str(adapted_rand)

    result = {}

    if 'Baseline' in sources:

        mc_result_filepath = os.path.join(experiment_folder, 'result.h5')

        if not debug:
            # Evaluate baseline
            mc_result = vigra.readHDF5(mc_result_filepath, mc_result_key)
            vol_mc_result = Volume(mc_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_mc_result)
            adapted_rand = neuron_ids_evaluation.adapted_rand(vol_mc_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            adapted_rand = 0.23

        print 'Baseline'
        print "\tvoi split   : " + str(voi_split)
        print "\tvoi merge   : " + str(voi_merge)
        print "\tadapted RAND: " + str(adapted_rand)

        result['Baseline'] = np.array([voi_split, voi_merge, adapted_rand])

    if 'Global' in sources:

        glob_result_filepath = os.path.join(experiment_folder, 'result_resolved_global.h5')
        glob_result_key = mc_result_key

        if not debug:
            # Evaluate global resolving
            glob_result = vigra.readHDF5(glob_result_filepath, glob_result_key)
            vol_glob_result = Volume(glob_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_glob_result)
            adapted_rand = neuron_ids_evaluation.adapted_rand(vol_glob_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            adapted_rand = 0.23

        print 'Global'
        print "\tvoi split   : " + str(voi_split)
        print "\tvoi merge   : " + str(voi_merge)
        print "\tadapted RAND: " + str(adapted_rand)

        result['Global'] = np.array([voi_split, voi_merge, adapted_rand])

    if 'Local' in sources:

        loc_result_filepath = os.path.join(experiment_folder, 'result_resolved_local.h5')
        loc_result_key = mc_result_key

        if not debug:
            # Evaluate local resolving
            loc_result = vigra.readHDF5(loc_result_filepath, loc_result_key)
            vol_loc_result = Volume(loc_result)
            (voi_split, voi_merge) = neuron_ids_evaluation.voi(vol_loc_result)
            adapted_rand = neuron_ids_evaluation.adapted_rand(vol_loc_result)
        else:
            voi_split = 1.09
            voi_merge = 0.70
            adapted_rand = 0.23

        print 'Local'
        print "\tvoi split   : " + str(voi_split)
        print "\tvoi merge   : " + str(voi_merge)
        print "\tadapted RAND: " + str(adapted_rand)

        result['Local'] = np.array([voi_split, voi_merge, adapted_rand])

    return result


def all_sample_roi_and_rand(samples, halves, defect_corrects, project_folder,
                            sources=['Baseline', 'Global', 'Local'], cache_file=None,
                            debug=False):

    if cache_file is None or not os.path.isfile(cache_file):
        results = {}
        for source in sources:
            results[source] = []

        for idx, sample in enumerate(samples):

            half = halves[idx]
            defect_correct = defect_corrects[idx]

            result = roi_and_rand(sample, half, defect_correct, project_folder,
                                  sources=sources, debug=debug)

            for source in sources:
                results[source].append(result[source][:, None])

        for source in sources:
            results[source] = np.concatenate(results[source], axis=1)

        if cache_file is not None:
            with open(cache_file, mode='w') as f:
                pickle.dump(results, f)

    else:
        with open(cache_file, mode = 'r') as f:
            results = pickle.load(f)

    return results


def plot_roi_and_rand(project_folder, halves, defect_corrects, samples, pool_some=2,
                      sources=['Baseline', 'Global', 'Local'], cache_file=None,
                      debug=False):

    # Evaluate
    results = all_sample_roi_and_rand(samples, halves, defect_corrects, project_folder,
                                      sources=sources, cache_file=cache_file,
                                      debug=debug)

    print results['Baseline']
    print results['Global']
    print results['Local']

    import matplotlib
    font = {'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    results_to_plot = {}
    stds_to_plot = {}
    for k, v in results.iteritems():
        results_to_plot[k] = []
        stds_to_plot[k] = []
        for i in xrange(0, v.shape[1], pool_some):
            results_to_plot[k].append(np.mean(v[:, i:i+pool_some], axis=1))
            stds_to_plot[k].append(np.std(v[:, i:i+pool_some], axis=1))

    for i in xrange(0, len(results_to_plot[sources[0]])):

        # Plotting
        n_groups = 3

        fig, ax = plt.subplots()

        index = np.arange(n_groups)
        bar_width = 0.27

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        labels = ['$V^{\mathrm{split}}$', '$V^{\mathrm{merge}}$', 'RI']

        for j in xrange(0, 3):
            to_plot = [results_to_plot['Baseline'][i][j],
                       results_to_plot['Global'][i][j],
                       results_to_plot['Local'][i][j]]

            std = [stds_to_plot['Baseline'][i][j],
                   stds_to_plot['Global'][i][j],
                   stds_to_plot['Local'][i][j]]


            plt.bar(index + j*bar_width,
                             to_plot,
                             bar_width,
                             alpha=opacity,
                             # color='b',
                             yerr=std,
                             error_kw=error_config,
                             label=labels[j])

        # plt.xlabel('Group')
        plt.ylabel('Score')
        # plt.title('Scores by group and gender')
        plt.xticks(index + bar_width, sources)

        ax.legend(loc='center left', bbox_to_anchor=(0.76, 0.86))

        plt.tight_layout()

        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        # fig.savefig('/export/home/jhennies/Documents/samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()


def plot_roi_and_rand_group_errors(
        project_folder, halves, defect_corrects, samples,
        sources=['Baseline', 'Global', 'Local'], cache_file=None,
        debug=False
):

    def autolabel(rects, height_over_bar=0.01):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            # ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
            #         '%d' % float(int(height * 10)) / 10,
            #         ha='center', va='bottom')

            ax.text(rect.get_x() + rect.get_width() / 2., height_over_bar + height,
                    '{0:.2f}'.format(height),
                    ha='center', va='bottom', fontsize=10, rotation=90)

    # Evaluate
    results = all_sample_roi_and_rand(samples, halves, defect_corrects, project_folder,
                                      sources=sources, cache_file=cache_file,
                                      debug=debug)

    print results['Baseline']
    print results['Global']
    print results['Local']

    import matplotlib
    font = {'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    y_names = ['$V^{\mathrm{split}}$', '$V^{\mathrm{merge}}$', 'RI']

    for i in xrange(0, 3):

        # Plotting
        n_groups = len(samples)

        results_this = [results['Baseline'][i],
                        results['Global'][i],
                        results['Local'][i]]

        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

        index = np.arange(n_groups)
        bar_width = 0.27

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        labels = ['Baseline', 'Global', 'Local']

        for j in xrange(0, 3):
            to_plot = results_this[j]

            std = [0,] * len(to_plot)

            rects_j = plt.bar(index + j*bar_width,
                             to_plot,
                             bar_width,
                             alpha=opacity,
                             # color='b',
                             yerr=std,
                             error_kw=error_config,
                             label=labels[j])

            autolabel(rects_j, 0.02 * (np.max(results_this) - np.min(results_this)))

        ymin = np.min(results_this) - 0.2 * (np.max(results_this) - np.min(results_this))
        ymax = np.max(results_this) + 0.2 * (np.max(results_this) - np.min(results_this))

        axes = plt.gca()
        axes.set_ylim([ymin, ymax])

        # plt.xlabel('Group')
        plt.ylabel(y_names[i])
        # plt.title('Scores by group and gender')
        plt.xticks(index + bar_width, ['$A_0$', '$A_1$', '$B_0$', '$B_1$', '$C_0$', '$C_1$'])

        ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.8))

        plt.tight_layout()

        handles, labels = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        # fig.savefig('/export/home/jhennies/Documents/samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')

        # fig.set_size_inches(18.5, 10.5)

    plt.show()


def plot_all_sample_path_eval(project_folder, thresh_range, halves, defect_corrects, samples,
                              measures=['f1', 'precision', 'recall', 'accuracy']):

    results_path, results_obj = all_sample_path_eval(
        project_folder, thresh_range, samples, halves, defect_corrects,
        measures=measures
    )

    plt.figure()

    for key in results_path.keys():

        measures_path = np.concatenate(results_path[key], axis=1)

        means_path = np.mean(measures_path, axis=1)
        std_path = np.std(measures_path, axis=1)

        plt.errorbar(thresh_range, means_path, yerr=std_path, fmt='-o', label=key)

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()

    plt.figure()

    for key in results_obj.keys():

        measures_obj = np.concatenate(results_obj[key], axis=1)

        means_obj = np.mean(measures_obj, axis=1)
        std_obj = np.std(measures_obj, axis=1)

        plt.errorbar(thresh_range, means_obj, yerr=std_obj, fmt='-o', label=key)

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()

    plt.show()


def plot_all_sample_path_eval_split_samples(project_folder, thresh_range, halves, defect_corrects, samples,
                              measures=['f1', 'precision', 'recall', 'accuracy']):

    results_path, results_obj = all_sample_path_eval(
        project_folder, thresh_range, samples, halves, defect_corrects,
        measures=measures
    )

    plt.figure()

    colors = {'f1':'C0', 'precision':'C1', 'recall':'C2'}
    opacity = 1
    error_config = {'ecolor': '0.3'}

    for key, val in results_path.iteritems():

        # measures_path = np.concatenate(results_path[key], axis=1)

        # means_path = np.mean(measures_path, axis=1)
        # std_path = np.std(measures_path, axis=1)

        plt.errorbar(thresh_range, val[0], yerr=[0]*len(val[0]), fmt='-o', label=key,
                     color=colors[key], alpha=opacity)
        plt.errorbar(thresh_range, val[1], yerr=[0]*len(val[1]), fmt='--o',
                     color=colors[key], alpha=opacity)

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([-0.05, 1.05])

    plt.figure()

    for key, val in results_obj.iteritems():

        # measures_obj = np.concatenate(results_obj[key], axis=1)
        #
        # means_obj = np.mean(measures_obj, axis=1)
        # std_obj = np.std(measures_obj, axis=1)

        plt.errorbar(thresh_range, val[0], yerr=[0]*len(val[0]), fmt='-o', label=key, color=colors[key])
        plt.errorbar(thresh_range, val[1], yerr=[0]*len(val[1]), fmt='--o', color=colors[key])

    plt.xlabel('$t_m$')
    plt.title("")
    plt.legend()
    axes = plt.gca()
    axes.set_xlim([-0.05, 1.05])

    plt.show()


if __name__ == '__main__':

    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'
    thresh_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # This is for all samples
    halves = ['0', '1']*3
    defect_corrects = [False, False, True, True, True, True]
    samples = ['A', 'A', 'B', 'B', 'C', 'C']
    # halves = ['0', '1']
    # defect_corrects = [True, True]
    # samples = ['B', 'B']

    # plot_all_sample_path_eval(project_folder, thresh_range, halves, defect_corrects, samples)
    #
    # # Plot sample path evaluation
    # for i in xrange(0, 6, 2):
    #     plot_all_sample_path_eval(project_folder, thresh_range,
    #                               halves[i: i+2],
    #                               defect_corrects[i: i+2],
    #                               samples[i: i+2],
    #                               measures=['f1', 'precision', 'recall'])

    # Plot sample path evaluation
    for i in xrange(0, 6, 2):
        plot_all_sample_path_eval_split_samples(project_folder, thresh_range,
                                  halves[i: i+2],
                                  defect_corrects[i: i+2],
                                  samples[i: i+2],
                                  measures=['f1', 'precision', 'recall'])

    # plot_all_sample_path_eval(project_folder, thresh_range,
    #                           halves, defect_corrects, samples,
    #                           measures=['f1', 'precision', 'recall'])

    # plot_all_sample_path_eval_split_samples(project_folder, thresh_range,
    #                           halves, defect_corrects, samples,
    #                           measures=['f1', 'precision', 'recall'])



    # Plot roi and rand

    # cache_file = os.path.join(project_folder, 'all_samples_roi_and_rand_A_B_C.pkl')
    # # cache_file = os.path.join(project_folder, 'all_samples_roi_and_rand.pkl')
    # # plot_roi_and_rand(project_folder, halves, defect_corrects, samples, pool_some=2,
    # #                   cache_file=cache_file,
    # #                   debug=False)
    # plot_roi_and_rand_group_errors(project_folder, halves, defect_corrects, samples,
    #                   cache_file=cache_file,
    #                   debug=False)
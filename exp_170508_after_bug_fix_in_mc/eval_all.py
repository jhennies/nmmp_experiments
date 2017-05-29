
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


import re


def roi_and_rand_general(
        sample, half, defect_correct, project_folder,
        result_file, caching=False, debug=False
):
    print '\nEvaluating spl{}_z{}'.format(sample, half)
    print 'Result file: {}'.format(result_file)

    experiment_folder = os.path.join(project_folder, 'spl{}_z{}/'.format(sample, half))

    if caching:
        cache_filepath = os.path.join(
            experiment_folder,
            re.sub('.h5$', '', result_file) + '_roi_and_rand_cache.pkl'
        )
    else:
        cache_filepath = None

    if caching and os.path.isfile(cache_filepath):
        with open(cache_filepath, mode='r') as f:
            return pickle.load(f)

    else:

        if defect_correct:
            defect_correct_str = '_defect_correct'
        else:
            defect_correct_str = ''

        mc_result_key = 'z/{}/data'.format(half)

        # Load stuff
        source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

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

        mc_result_filepath = os.path.join(experiment_folder, result_file)

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

        print "\tvoi split   : " + str(voi_split)
        print "\tvoi merge   : " + str(voi_merge)
        print "\tadapted RAND: " + str(adapted_rand)

        if caching:
            with open(cache_filepath, mode='w') as f:
                pickle.dump((voi_split, voi_merge, adapted_rand), f)

        return (voi_split, voi_merge, adapted_rand)


def multiple_roi_and_rand_general(
        sample, half, defect_correct, project_folder,
        result_files, caching=True, debug=False
):

    result = []

    for result_file in result_files:

        if type(result_file) is list:

            result.append([])
            for variation_file in result_file:
                result[-1].append(
                    roi_and_rand_general(sample, half, defect_correct, project_folder,
                    variation_file, caching=caching, debug=debug
                )
            )

        else:

            result.append(
                roi_and_rand_general(
                    sample, half, defect_correct, project_folder,
                    result_file, caching=caching, debug=debug
                )
            )

    return result


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
        debug=False, ymins=None, ymaxs=None, num_digits_over_bar=3,
        y_names=['$V^{\mathrm{split}}$', '$V^{\mathrm{merge}}$', 'RI']
):

    # Evaluate
    results = all_sample_roi_and_rand(samples, halves, defect_corrects, project_folder,
                                      sources=sources, cache_file=cache_file,
                                      debug=debug)

    print results['Baseline']
    # print results['Global']
    print results['Local']

    import matplotlib
    font = {'weight': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)



    for i in xrange(0, 3):

        # Plotting
        n_groups = len(samples)

        # results_this = [results['Baseline'][i],
        #                 results['Global'][i],
        #                 results['Local'][i]]

        results_this = []
        for source in sources:
            results_this.append(results[source][i])

        fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

        index = np.arange(n_groups)
        bar_width = 0.27

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        labels = sources

        for j in xrange(0, len(sources)):
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

            autolabel(rects_j, ax, 0.02 * (np.max(results_this) - np.min(results_this)),
                      num_digits=num_digits_over_bar)

        if ymins is not None:
            ymin=ymins[i]
        else:
            ymin = np.min(results_this) - 0.2 * (np.max(results_this) - np.min(results_this))
        if ymaxs is not None:
            ymax = ymaxs[i]
        else:
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


def autolabel(rects, ax, height_over_bar=0.01, num_digits=2, font_size=10):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        # ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
        #         '%d' % float(int(height * 10)) / 10,
        #         ha='center', va='bottom')

        ax.text(rect.get_x() + rect.get_width() / 2., height_over_bar + height,
                ('{0:.' + str(num_digits) + 'f}').format(height),
                    ha='center', va='bottom', fontsize=font_size, rotation=90)


def plot_roi_and_rand_sample_count(
        project_folder, half, defect_correct, sample,
        result_files, caching=True,
        labels=None,
        fig_size=(4, 4),
        debug=False,
        show_baseline=False,
        x_label=None,
        num_digits_over_bar=2,
        y_names=['$V^{\mathrm{split}}$', '$V^{\mathrm{merge}}$', 'RI'],
        font_size=15,
        font_size_over_bars=12,
        font_size_x=15,
        font_size_y=15
):

    # TODO: Adapt this function to different path sampling (i.e. error bars)

    # Evaluate
    # results = all_sample_roi_and_rand(samples, halves, defect_corrects, project_folder,
    #                                   sources=sources, cache_file=cache_file,
    #                                   debug=debug)

    results = multiple_roi_and_rand_general(
        sample, half, defect_correct, project_folder, result_files, caching=caching, debug=debug
    )

    # Prepare the results:
    stds = []
    for idx, item in enumerate(results):
        if type(item) is tuple:
            stds.append((0, 0, 0))
        else:
            stds.append(tuple(np.std(np.array(item), axis=0).tolist()))
            results[idx] = tuple(np.mean(np.array(item), axis=0).tolist())

    import matplotlib
    font = {'weight': 'normal',
            'size': font_size}
    matplotlib.rc('font', **font)

    for i in xrange(0, len(results[0])):

        # Plotting
        n_groups = len(result_files)
        if show_baseline:
            n_groups -= 1

        fig, ax = plt.subplots(figsize=fig_size, dpi=80)

        index = np.arange(n_groups)
        bar_width = 0.8

        opacity = 0.4
        error_config = {'ecolor': '0.3'}

        if labels is None:
            labels = np.arange(n_groups)

        to_plot = [x[i] for x in results]
        std = [x[i] for x in stds]

        if show_baseline:
            baseline = to_plot[0]
            to_plot = to_plot[1:]
            std = std[1:]

        rects = plt.bar(index + bar_width,
                        to_plot,
                        bar_width,
                        yerr=std,
                        alpha=opacity,
                        label='')



        ymin = 0.7 * np.min(to_plot)
        # ymin = np.max(np.min(to_plot) - 0.8 * (np.max(to_plot) - np.min(to_plot)), 0)
        # ymax = np.max(to_plot) + 0.2 * (np.max(to_plot) - np.min(to_plot))

        # ymin = 0
        ymax = np.max(to_plot) + 0.2 * np.max(to_plot)

        # autolabel(rects, 0.02 * (np.max(to_plot) - np.min(to_plot)))
        autolabel(rects, ax, 0.05 * (ymax - ymin), num_digits=num_digits_over_bar,
                  font_size=font_size_over_bars)

        axes = plt.gca()
        axes.set_ylim([ymin, ymax])

        if x_label is not None:
            plt.xlabel(x_label)
        plt.ylabel(y_names[i])
        # plt.title('Scores by group and gender')
        plt.xticks(index + bar_width, labels, fontsize=font_size_x)
        plt.yticks(fontsize=font_size_y)

        # ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.8))

        plt.tight_layout()

        handles, lbls = ax.get_legend_handles_labels()
        # lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
        # fig.savefig('/export/home/jhennies/Documents/samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')

        if show_baseline:
            ax.axhline(y=baseline, xmin=0, xmax=3, c="blue", linewidth=0.5, zorder=0)

    # fig.set_size_inches(18.5, 10.5)

    plt.show()


from evaluation import resolve_merges_error_rate_path_level
from evaluation import compute_path_error_rates


def eval_obj_measures(
        spl, half,
        project_folder,
        seg_file, seg_key,
        resolved_file, resolved_key,
        thresh_range=None,
        resolved_only=False,
        defect_correct=False
):

    if thresh_range is None:
        thresh_range = [0.3]

    experiment_folder = os.path.join(
        project_folder,
        'spl{}_z{}/'.format(spl, half)
    )
    path_data_folder = os.path.join(
        experiment_folder,
        'cache/path_data/'
    )

    path_data_filepath = os.path.join(
        path_data_folder,
        'paths_ds_spl{}_z{}.h5'.format(spl, half)
    )

    if defect_correct:
        defect_correct_str = '_defect_correct'
    else:
        defect_correct_str = ''

    mc_result_key = 'z/{}/data'.format(half)

    # Load stuff
    source_folder = '/mnt/localdata02/jhennies/neuraldata/cremi_2016/170321_resolve_false_merges/'

    gt_filepath = os.path.join(
        source_folder,
        'cremi.spl{}.train.raw_neurons{}.crop.axes_xyz.split_z.h5'.format(spl, defect_correct_str))
    gt_key = 'z/{}/neuron_ids'.format(half)

    # Load paths
    paths = vigra.readHDF5(path_data_filepath, 'all_paths')
    if paths.size:
        paths = np.array([path.reshape((len(path) / 3, 3)) for path in paths])
    paths_to_objs = vigra.readHDF5(path_data_filepath, 'paths_to_objs')
    with open(os.path.join(path_data_folder, 'false_paths_predictions.pkl')) as f:
        false_merge_probs = pickle.load(f)

    # Load images
    resolved = vigra.readHDF5(os.path.join(experiment_folder, resolved_file), resolved_key)
    gt = vigra.readHDF5(gt_filepath, gt_key)
    seg = vigra.readHDF5(os.path.join(experiment_folder, seg_file), seg_key)

    # Determine merge error rate (path level)
    errors_seg, errors_rsd, errors_to_obj = resolve_merges_error_rate_path_level(
        paths, paths_to_objs, resolved, gt, seg,
        false_merge_probs,
        thresh_range=thresh_range,
        resolved_only=resolved_only
    )

    return errors_seg, errors_rsd, errors_to_obj


def eval_obj_measures_readable(
        spl, half,
        project_folder,
        seg_file, seg_key,
        resolved_files, resolved_key,
        thresh_range=None,
        resolved_only=False
):

    if thresh_range == None:
        thresh_range = [0.3]

    result = []

    files_shape = None
    if np.array(resolved_files).ndim > 1:
        files_shape = np.array(resolved_files).shape
        resolved_files = np.array(resolved_files).flatten()

    for resolved_file in resolved_files:

        errors_seg, errors_rsd, errors_to_obj = eval_obj_measures(
            spl, half,
            project_folder,
            seg_file, seg_key,
            resolved_file, resolved_key,
            thresh_range=thresh_range,
            resolved_only=resolved_only
        )

        result.append([])

        for thresh in thresh_range:
            print 'Evaluation for thresh = {}'.format(thresh)

            # TP
            obj_mask_with_all_true_seg = np.array([x.all() for x in errors_seg[thresh]])
            tp = sum(np.logical_not(obj_mask_with_all_true_seg))

            # TP & fully resolved
            obj_mask_with_all_true_rsd = np.array([(x != 1).all() for x in errors_rsd[thresh]])
            false_merge_rsd = sum(np.logical_not(obj_mask_with_all_true_rsd))
            tp_fully_resolved = tp - false_merge_rsd

            # TP & falsely split
            obj_mask_at_least_one_false_split = np.array([(x == 2).any() for x in errors_rsd[thresh]])
            tp_falsely_split = sum(
                np.logical_and(
                    obj_mask_at_least_one_false_split, np.logical_not(obj_mask_with_all_true_seg)
                )
            )

            # FP
            errors_seg_with_all_true = np.array(errors_seg[thresh])[obj_mask_with_all_true_seg]
            fp = len(errors_seg_with_all_true)

            # FP & falsely split
            obj_mask_at_least_one_false_split = np.array([(x == 2).any() for x in errors_rsd[thresh]])
            fp_falsely_split = sum(
                np.logical_and(
                    obj_mask_at_least_one_false_split, obj_mask_with_all_true_seg
                )
            )

            # Object count
            number_of_objs = len(errors_seg[thresh])

            result[-1].append((number_of_objs, tp, fp, tp_fully_resolved, tp_falsely_split, fp_falsely_split))

    if files_shape is not None:
        result = np.array(result)
        result = result.reshape(np.concatenate((files_shape, result.shape[1:])))

    return result


def run_eval_obj_measures_to_tex_table():

    spl = 'B'
    half = '0'
    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'
    seg_file = 'result.h5'
    seg_key = 'z/{}/data'.format(half)
    resolved_key = seg_key

    thresh_range = [0.3]
    xlabels = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000]

    result_files = [
        'result_resolved_local_gs_1000.h5',
        'result_resolved_local_gs_100.h5',
        'result_resolved_local_gs_10.h5',
        'result_resolved_local_gs_5.h5',
        'result_resolved_local_gs_2.h5',
        'result_resolved_local_gs_1.h5',
        'result_resolved_local_gs_0.5.h5',
        'result_resolved_local_gs_0.2.h5',
        'result_resolved_local_gs_0.1.h5',
        'result_resolved_local_gs_0.01.h5',
        'result_resolved_local_gs_0.001.h5',
    ]


    result = eval_obj_measures_readable(
        spl, half,
        project_folder,
        seg_file, seg_key,
        result_files, resolved_key,
        thresh_range=thresh_range,
        resolved_only=False
    )

    print result
    res = np.array(result).squeeze()[:, :, 3:6]

    import matplotlib
    font = {'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)

    labels = ['tp & fully resolved', 'tp & falsely split', 'fp & falsely split']
    for idx in xrange(0, 3):
        plt.plot(res[:, idx],'o-', label=labels[idx])
        plt.xlabel('$w_{path}$')
        plt.ylabel('object number')
        plt.title("")
        plt.legend()
        axes = plt.gca()
        plt.xticks(range(0, len(xlabels)), xlabels)
        plt.yticks(range(0, 24, 5))

    plt.show()


def run_eval_obj_measures_path_sample_count():

    spl = 'B'
    half = '0'
    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'
    seg_file = 'result.h5'
    seg_key = 'z/{}/data'.format(half)
    resolved_key = seg_key

    thresh_range = [0.3]

    inner = [0, 1, 2, 3, 4]
    outer = [0, 5, 10, 15, 20, 25, 30]
    result_files = []
    for i in outer:
        result_files.append([])
        for j in inner:
            result_files[-1].append('result_resolved_local_path_count_{}_{}.h5'.format(j, i))

    result = eval_obj_measures_readable(
        spl, half,
        project_folder,
        seg_file, seg_key,
        result_files, resolved_key,
        thresh_range=thresh_range,
        resolved_only=False
    )

    print result

    rslt = result.squeeze()[:, :, 3:6]

    import matplotlib
    font = {'weight': 'normal',
            'size': 13}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(7, 5), dpi=80)

    colors = {'f1':'C0', 'precision':'C1', 'recall':'C2'}
    opacity = 1
    error_config = {'ecolor': '0.3'}

    # for key, val in results_path.iteritems():

    # measures_path = np.concatenate(results_path[key], axis=1)

    # means_path = np.mean(measures_path, axis=1)
    # std_path = np.std(measures_path, axis=1)

    mean = np.mean(rslt, axis=1)
    std = np.std(rslt, axis=1)

    labels = ['tp & fully resolved', 'tp & falsely split', 'fp & falsely split']

    for i in xrange(0, mean.shape[1]):
        plt.errorbar(outer, mean[:, i], yerr=std[:, i], fmt='-o', label=labels[i])

    # plt.errorbar(thresh_range, val[1], yerr=[0]*len(val[1]), fmt='--o',
    #              color=colors[key], alpha=opacity)
    plt.legend()

    plt.xlabel('Samples')
    plt.ylabel('Object count')


    # plt.title("")

    axes = plt.gca()
    axes.set_ylim(0, 1.1*np.max(mean + std))
    ax.legend(loc='center left', bbox_to_anchor=(0.01, 0.2))

    plt.show()

def run_plot_roi_and_rand_sample_count():

    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'
    half = 0
    sample = 'B'
    defect_correct = True
    # result_files = [
    #     'result.h5',
    #     'result_resolved_local_path_count_0_0.h5'
    #     'result_resolved_local_path_count_0_5.h5',
    #     'result_resolved_local_path_count_0_10.h5',
    #     'result_resolved_local_path_count_0_15.h5',
    #     'result_resolved_local_path_count_0_20.h5',
    #     'result_resolved_local_path_count_0_25.h5'
    # ]

    inner = [0, 1, 2, 3, 4]
    outer = [0, 5, 10, 15, 20, 25, 30]
    result_files = ['result.h5']
    for i in outer:
        result_files.append([])
        for j in inner:
            result_files[-1].append('result_resolved_local_path_count_{}_{}.h5'.format(j, i))

    plot_roi_and_rand_sample_count(
        project_folder, half, defect_correct, sample, result_files,
        fig_size=(4, 4),
        labels=outer,
        caching=True, debug=False, show_baseline=True, x_label='Samples'
    )


def run_plot_roi_and_rand_grid_search():

    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'

    half = 1
    sample = 'B'
    defect_correct = True
    result_files = [
        'result.h5',
        'result_resolved_local_gs_1000.h5',
        'result_resolved_local_gs_100.h5',
        'result_resolved_local_gs_10.h5',
        'result_resolved_local_gs_5.h5',
        'result_resolved_local_gs_2.h5',
        'result_resolved_local_gs_1.h5',
        'result_resolved_local_gs_0.5.h5',
        'result_resolved_local_gs_0.2.h5',
        'result_resolved_local_gs_0.1.h5',
        'result_resolved_local_gs_0.01.h5',
        'result_resolved_local_gs_0.001.h5',
    ]

    plot_roi_and_rand_sample_count(
        project_folder, half, defect_correct, sample, result_files,
        fig_size=(8, 6),
        labels=[0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100, 1000],
        caching=True, debug=False, show_baseline=True, x_label='Weight',
        num_digits_over_bar=3, font_size_over_bars=15, font_size=15
    )


def run_plot_roi_and_group_errors():


    project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'

    thresh_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # This is for all samples
    halves = ['0', '1']*3
    defect_corrects = [False, False, True, True, True, True]
    samples = ['A', 'A', 'B', 'B', 'C', 'C']

    # Plot roi and rand

    # cache_file = os.path.join(project_folder, 'all_samples_roi_and_rand_A_B_C.pkl')
    cache_file = os.path.join(project_folder, 'all_samples_roi_and_rand_A_B_C_bl_loc.pkl')
    # cache_file = os.path.join(project_folder, 'all_samples_roi_and_rand.pkl')
    # plot_roi_and_rand(project_folder, halves, defect_corrects, samples, pool_some=2,
    #                   cache_file=cache_file,
    #                   debug=False)
    plot_roi_and_rand_group_errors(project_folder, halves, defect_corrects, samples,
                                   cache_file=cache_file,
                                   debug=False,
                                   ymins=[0.2, 0, 0], ymaxs=[1.6, 1.4, 0.4],
                                   y_names=['$V^{\mathrm{split}}$', '$V^{\mathrm{merge}}$', 'aRand'],
                                   sources=['Baseline', 'Local'])


if __name__ == '__main__':

    run_eval_obj_measures_path_sample_count()
    # run_eval_obj_measures_to_tex_table()

    # run_plot_roi_and_rand_sample_count()
    # run_plot_roi_and_rand_grid_search()
    # run_plot_roi_and_group_errors()

    # project_folder = '/mnt/localdata01/jhennies/neuraldata/results/multicut_workflow/170508_after_bug_fix_in_mc/'

    # thresh_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #
    # # This is for all samples
    # halves = ['0', '1']*3
    # defect_corrects = [False, False, True, True, True, True]
    # samples = ['A', 'A', 'B', 'B', 'C', 'C']
    # # halves = ['0', '1']
    # # defect_corrects = [True, True]
    # # samples = ['B', 'B']
    #
    # # plot_all_sample_path_eval(project_folder, thresh_range, halves, defect_corrects, samples)
    # #
    # # # Plot sample path evaluation
    # # for i in xrange(0, 6, 2):
    # #     plot_all_sample_path_eval(project_folder, thresh_range,
    # #                               halves[i: i+2],
    # #                               defect_corrects[i: i+2],
    # #                               samples[i: i+2],
    # #                               measures=['f1', 'precision', 'recall'])
    #
    # # Plot sample path evaluation
    # for i in xrange(0, 6, 2):
    #     plot_all_sample_path_eval_split_samples(project_folder, thresh_range,
    #                               halves[i: i+2],
    #                               defect_corrects[i: i+2],
    #                               samples[i: i+2],
    #                               measures=['f1', 'precision', 'recall'])

    # plot_all_sample_path_eval(project_folder, thresh_range,
    #                           halves, defect_corrects, samples,
    #                           measures=['f1', 'precision', 'recall'])

    # plot_all_sample_path_eval_split_samples(project_folder, thresh_range,
    #                           halves, defect_corrects, samples,
    #                           measures=['f1', 'precision', 'recall'])




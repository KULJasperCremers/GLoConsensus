import matplotlib.pyplot as plt
import numpy as np
import path as path_class


def plot_global_sm_and_column_warping_paths(
    timeseries_list,
    global_similarity_matrix,
    global_column_dict_path,
):
    fig, axs, _ = plot_sm(
        np.concatenate(timeseries_list),
        np.concatenate(timeseries_list),
        global_similarity_matrix,
    )
    paths = []
    for column in global_column_dict_path:
        column_paths = []
        for path in global_column_dict_path[column]:
            paths.append(path)
            column_paths.append(path)
        fig1, axs1, _ = plot_sm(
            np.concatenate(timeseries_list),
            np.concatenate(timeseries_list),
            global_similarity_matrix,
        )
        axs1 = plot_local_warping_paths(axs1, column_paths, lw=3)
        fig1.savefig(f'./plots/global_sm_with_colum_{column}_wps.png')
    axs = plot_local_warping_paths(axs, paths, lw=3)
    fig.savefig('./plots/global_sm_with_wps.png')
    # plt.close(fig)

    return fig, axs


def plot_sm(
    s1,
    s2,
    sm,
    path=None,
    figsize=(15, 15),
    colorbar=False,
    matshow_kwargs=None,
    ts_kwargs={'linewidth': 1.5, 'ls': '-'},
):
    from matplotlib import gridspec

    width_ratios = [0.9, 5]
    if colorbar:
        height_ratios = [0.8, 5, 0.15]
    else:
        height_ratios = width_ratios

    fig = plt.figure(figsize=figsize, frameon=True)
    gs = gridspec.GridSpec(
        2 + colorbar,
        2,
        wspace=5,
        hspace=5,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_prop_cycle(None)
    ax1.set_axis_off()
    ax1.plot(range(len(s2)), s2, **ts_kwargs)
    ax1.set_xlim([-0.5, len(s2) - 0.5])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_prop_cycle(None)
    ax2.set_axis_off()
    ax2.plot(-s1, range(len(s1), 0, -1), **ts_kwargs)
    ax2.set_ylim([0.5, len(s1) + 0.5])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_aspect(1)
    ax3.tick_params(
        axis='both',
        which='both',
        labeltop=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
    )

    kwargs = {} if matshow_kwargs is None else matshow_kwargs
    img = ax3.matshow(sm, **kwargs)

    cax = None
    if colorbar:
        cax = fig.add_subplot(gs[2, 1])
        fig.colorbar(img, cax=cax, orientation='horizontal')

    gs.tight_layout(fig)

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position(
        (ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])
    )  # adjust the time series on the left vertically
    if len(s1) < len(s2):
        ax3.set_position(
            (ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])
        )  # move the time series on the left and the distance matrix upwards
    if len(s1) > len(s2):
        ax3.set_position(
            (ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])
        )  # move the time series at the top and the distance matrix to the left
        ax1.set_position(
            (ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally
    if len(s1) == len(s2):
        ax1.set_position(
            (ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])
        )  # adjust the time series at the top horizontally

    ax = fig.axes
    return fig, ax, cax


def plot_local_warping_paths(axs, paths, **kwargs):
    for p in paths:
        if isinstance(p, path_class.Path):
            axs[3].plot(p.path[:, 1], p.path[:, 0], 'red', **kwargs)
        if isinstance(p, np.ndarray):
            axs[3].plot(p[:, 1], p[:, 0], 'red', **kwargs)

    return axs

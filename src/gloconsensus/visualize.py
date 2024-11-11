import matplotlib.pyplot as plt
import numpy as np


def plot_motif_set(
    timeseries_list, representative, motif_set, induced_paths, best_fitness
):
    series = np.concatenate(timeseries_list)
    begin, end = representative
    k = len(motif_set)
    fig, ax = plt.subplots(
        2,
        k + 1,
        figsize=(2 * (0.5 + k), 3),
        width_ratios=[0.5] + [1] * k,
        height_ratios=[0.5, 1],
    )
    ax[0, 0].set_axis_off()

    ax[1, 0].plot(-series[begin:end, :], range(end - begin), 0, -1)
    ax[1, 0].set_ylim([-0.5, end - begin + 0.5])
    ax[1, 0].set_axis_off()

    for i, path in enumerate(induced_paths):
        begin_motif, end_motif = path[0][0], path[-1][0]

        ax[0, i + 1].plot(series[begin_motif:end_motif, :])
        ax[0, i + 1].set_xlim([-0.5, end_motif - begin_motif + 0.5])
        ax[0, i + 1].set_axis_off()

        ax[1, i + 1].invert_yaxis()
        ax[1, i + 1].plot(
            path[:, 0] - begin_motif,
            path[:, 1] - begin,
            c='r',
            ls='-',
            marker='.',
            markersize=1,
        )
        ax[1, i + 1].set_ylim([end - begin, 0])
        ax[1, i + 1].set_xlim([0, end_motif - begin_motif])

    fig.suptitle(f'Best fitness: {best_fitness:.5f}')
    plt.tight_layout()
    plt.savefig(f'./plots/motif_set_{begin}_{end}_fitness_{best_fitness:.5f}.png')
    plt.close(fig)

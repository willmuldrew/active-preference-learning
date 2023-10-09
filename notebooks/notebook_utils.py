import math

import matplotlib.pyplot as plt


def key_to_color(key):
    return {
        "ENTROPY": 'orange',
        "CERTAINTY": 'green',
        "UNCERTAINTY": 'blue',
        "OFFLINE": 'black',
        "RANDOM": 'gray',
        "TAILS_UNCERTAINTY": 'pink',
        "MID_UNCERTAINTY": 'purple',
    }[key]


def key_to_nice_name(key):
    return {
        "ENTROPY": 'Active-Predictive-Entropy',
        "CERTAINTY": 'Active-R-Certainty',
        "UNCERTAINTY": 'Active-R-Uncertainty',
        "OFFLINE": 'Offline-Random',
        "RANDOM": 'Online-Random',
        "TAILS_UNCERTAINTY": 'pink',
        "MID_UNCERTAINTY": 'purple',
    }[key]


def plot_winrate_grid(df, df_agg, series: list[list[str]], output_path=None, title="title!", ncols=2, acq="acquire_pairs_function", win_rate="win_rate", ylim=(0.55, 0.9), figwidth=11):
    m_s = sorted(list(set(df.m)))
    nrows = math.ceil(len(series) / 3.0)

    fig, axs = plt.subplots(figsize=(figwidth, 4 * nrows), nrows=nrows, ncols=ncols)

    if nrows == 1:
        axs = [axs]
    axs = [ax for r in axs for ax in r]

    for i, ax in enumerate(axs):
        ax.title.set_text("(" + chr(ord('a') + i) + ")")

    for ax, keys in zip(axs, series):
        for key, group in df_agg.groupby(acq):
            if key in keys:
                group.plot('m', 'mean', yerr='stderr', alpha=0.5, label=key_to_nice_name(key), ax=ax, capsize=2.0,
                           color=key_to_color(key))
                df[df[acq] == key].plot.scatter(
                    color=key_to_color(key), ax=ax, marker='x', s=10, x='m', y=win_rate)

        ax.set_xticks(m_s)
        ax.set_xticklabels(m_s)
        ax.set_ylim(ylim[0], ylim[1])
        ax.tick_params(axis='x', which='minor', bottom=False)
        ax.set_ylabel("Win-rate (%)")
        ax.legend(loc='lower right')

        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_yticklabels([f"{int(x * 100)}" for x in ax.get_yticks()])

    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)

    plt.show()
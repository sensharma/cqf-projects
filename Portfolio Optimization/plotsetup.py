import matplotlib
from math import sqrt


def modify_image(fig_width=None, fig_height=None, columns=None):
    """
    Set up matplotlib's RC params for LaTeX plotting.
    """
    if columns is None:
        columns = 1

    if fig_width is None:
        fig_width = 8 if columns == 1 else 5.5

    if fig_height is None:
        golden_mean = (sqrt(5)-1.0)/2.0
        fig_height = fig_width*golden_mean if columns == 1 else 4

    MAX_HEIGHT_INCHES = 6.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large: {} so will reduce to {} inches.".format(fig_height, MAX_HEIGHT_INCHES))
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 14,
              'figure.titlesize': 16,
              'axes.titlesize': 16,
              'font.size': 14,
              'legend.fontsize': 10,
              'legend.frameon': True,
              'legend.edgecolor': 'gray',
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': True,
              'figure.figsize': [fig_width, fig_height],
              'font.family': 'serif'
    }
    matplotlib.rcParams.update(params)


def format_axes(ax):
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out')
    return ax
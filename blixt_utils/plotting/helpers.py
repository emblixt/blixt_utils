import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.colors as mcolors
from itertools import cycle

logger = logging.getLogger(__name__)

clrs = list(mcolors.BASE_COLORS.keys())
clrs.remove('w')
cclrs = cycle(clrs)  # "infinite" loop of the base colors

def next_color():
    return next(cclrs)

def axis_plot(ax, y, data, limits, styles, yticks=True, nxt=4, **kwargs):
    """
    Plot data in one subplot
    :param ax:
        matplotlib axes object
    :param y:
        numpy ndarray
        The depth data of length N
    :param data:
        list
        list of ndarrays, each of length N, which should be plotted in the same subplot
    :param limits:
        list
        list of lists, each with min, max value of respective curve, e.g.
        [[0, 150], [6, 16], ...]
    :param styles:
        list
        list of dictionaries that defines the plotting styles of the data
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :param yticks:
        bool
        if False the yticklabels are not shown
    :param nxt:
        int
        Number of gridlines in x direction
    :param kwargs:
    :return:
        list
        xlim, list of list, each with the limits (min, max) of the x axis
    """
    if data is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return

    if not (len(data) == len(limits) == len(styles)):
        raise IOError('Must be same number of items in data, limits and styles')

    # store the x axis limits that has been used in each axis
    xlims = []

    # set up multiple twin axes to allow for different scaling of each plot
    axes = []
    for i in range(len(data)):
        if i == 0:
            axes.append(ax)
        else:
            axes.append(axes[-1].twiny())
    # Adjust the positions according to the original axes
    for i, ax in enumerate(axes):
        if i == 0:
            pos = ax.get_position()
        else:
            ax.set_position(pos)

    # start plotting
    for i in range(len(data)):
        axes[i].plot(data[i], y, **styles[i])

    # set up the x range differently for each plot
    for i in range(len(data)):
        #axes[i].set_xlim(*limits[i])
        set_lim(axes[i], limits[i], 'x')
        #print(limits[i])
        xlims.append(axes[i].get_xlim())
        # Change major ticks to set up the grid as desired
        if nxt > 0:
            xlim = axes[i].get_xlim()
            x_int = np.abs(xlim[1] - xlim[0]) / (nxt + 1)
            axes[i].xaxis.set_major_locator(MultipleLocator(x_int))
            axes[i].get_xaxis().set_ticklabels([])
        else:
            axes[i].get_xaxis().set_ticks([])
        if i == 0:
            axes[i].set_ylim(ax.get_ylim()[::-1])
            if not yticks:
                axes[i].get_yaxis().set_ticklabels([])
        else:
            axes[i].tick_params(axis='x', length=0.)

    ax.grid(which='major', alpha=0.5)

    return xlims


def axis_log_plot(ax, y, data, limits, styles, yticks=True,  **kwargs):
    """
    Plot data in one subplot
    Similar to axis_plot, but uses the same (logarithmic) x axis for all data
    :param ax:
        matplotlib axes object
    :param y:
        numpy ndarray
        The depth data of length N
    :param data:
        list
        list of ndarrays, each of length N, which should be plotted in the same subplot
    :param limits:
        list
        min, max value of axis
        E.G. [0.2, 200]
    :param styles:
        list
        list of dictionaries that defines the plotting styles of the data
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :param yticks:
        bool
        if False the yticklabels are not shown
    :param nxt:
        int
        Number of gridlines in x direction
    :param kwargs:
    :return:
        list
        xlims, list of list, each with the limits (min, max) of the x axis
    """
    if not (len(data) == len(styles)):
        raise IOError('Must be same number of items in data and styles')

    # store the x axis limits that has been used in each axis
    xlims = []

    # start plotting
    for i in range(len(data)):
        ax.plot(data[i], y, **styles[i])

    ax.set_xscale('log')
    #ax.set_xlim(*limits)
    set_lim(ax, limits, 'x')
    xlims.append(ax.get_xlim())
    ax.get_xaxis().set_ticklabels([])
    ax.tick_params(axis='x', which='both', length=0)

    ax.set_ylim(ax.get_ylim()[::-1])
    if not yticks:
        ax.get_yaxis().set_ticklabels([])
        ax.tick_params(axis='y', length=0)

    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.2)

    return xlims


def annotate_plot(ax, y, pad=-30):
    if y is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return

    ax.plot(np.ones(len(y)), y, lw=0)
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.get_xaxis().set_ticks([])
    ax.tick_params(axis='y', direction='in', length=5., labelsize='smaller', right=True)
    yticks = [*ax.yaxis.get_major_ticks()]
    for tick in yticks:
        tick.set_pad(pad)


def header_plot(ax, limits, legends, styles, title=None):
    """
    Tries to create a "header" to a plot, similar to what is used in RokDoc and many CPI plots
    :param ax:
        matplotlib axes object
    :param limits:
        list
        list of lists, each with min, max value of respective curve, e.g.
        [[0, 150], [6, 16], ...]
        Should not be more than 4 items in this list
    :param legends:
        list
        list of strings, that should annotate the respective limits
    :param styles:
        list
        list of dicts which describes the line styles
        E.G. [{'lw':1, 'color':'k', 'ls':'-'}, {'lw':2, 'color':'r', 'ls':'-'}, ... ]
    :return:
    """
    if limits is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        if title is not None:
            ax.text(0.5, 0.1, title, ha='center')
        return

    if not (len(limits) == len(legends) == len(styles)):
        raise IOError('Must be same number of items in limits, legends and styles')

    # Sub divide plot in this number of horizontal parts
    n = 8
    for i in range(len(limits)):
        ax.plot([1, 2],  [n-1-2*i, n-1-2*i], **styles[i])
        ax.text(1-0.03, n-1-2*i, '{:.1f}'.format(limits[i][0]), ha='right', va='center', fontsize='smaller')
        ax.text(2+0.03, n-1-2*i, '{:.1f}'.format(limits[i][1]), ha='left', va='center', fontsize='smaller')
        ax.text(1.5, n-1-2*i+0.05, legends[i], ha='center', fontsize='smaller')

    ax.set_xlim(0.8, 2.3)
    ax.get_xaxis().set_ticks([])
    ax.set_ylim(0.5, 8)
    ax.get_yaxis().set_ticks([])


def wiggle_plot(ax, y, wiggle, zero_at=0., scaling=1., fill_pos_style='default',
                fill_neg_style='default', zero_style=None, **kwargs):
    """
    Draws a (seismic) wiggle plot centered at 'zero_at'
    :param ax:
        matplotlib Axis object
    :param y:
        numpy ndarray
        Depth variable
    :param wiggle:
        numpy ndarray
        seismic trace
    :param zero_at:
        float
        x value at which the wiggle should be centered
    :param scaling:
        float
        scale the data
    :param fill:
        str
        'neg': Fills the negative side of the wiggle
        'pos': Fills the positive side of the wiggle
        None : No fill
    :param zero_style:
        dict
        style keywords of the line marking zero
    :param kwargs:

    :return:
    """
    if y is None:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return

    #print(len(y), len(wiggle))
    lw = kwargs.pop('lw', 0.5)
    c = kwargs.pop('c', 'k')

    if fill_pos_style == 'default':
        fill_pos_style = {'color': 'r', 'alpha': 0.2, 'lw': 0.}
    elif fill_pos_style == 'pos-blue':
        fill_pos_style = {'color': 'b', 'alpha': 0.2, 'lw': 0.}
    if fill_neg_style == 'default':
        fill_neg_style = {'color': 'b', 'alpha': 0.2, 'lw': 0.}
    elif fill_neg_style == 'neg-red':
        fill_neg_style = {'color': 'r', 'alpha': 0.2, 'lw': 0.}
    if zero_style is None:
        zero_style = {'lw': 0.5, 'color': 'k', 'alpha': 0.2}
    # shift and scale the data so that it is centered at 'zero_at'
    wig = zero_at + wiggle*scaling
    ax.plot(wig, y, lw=lw, color=c, **kwargs)
    if fill_pos_style is not None:
        ax.fill_betweenx(y, wig, zero_at, wig > zero_at, **fill_pos_style)
        #pass
    if fill_neg_style is not None:
        ax.fill_betweenx(y, zero_at, wig, wig < zero_at, **fill_neg_style)
        #pass

    ax.axvline(zero_at, **zero_style)

    ax.set_ylim(ax.get_ylim()[::-1])

def set_lim(ax, limits, axis=None):
    """
    Convinience function to set the x axis limits. Interprets None as autoscale
    :param ax:
        matplotlib.axes
    :param limits:
        list
        list of  min, max value.
        If any is None, axis is autoscaled
    :param axis
        str
        'x' or 'y'
    :return:
    """
    if axis is None:
        return

    if None in limits:
        print('limits', limits)
        # first autoscale
        ax.autoscale(True, axis=axis)
        if axis == 'x':
            _lims = ax.get_xlim()
            print(_lims)
            ax.set_xlim(
                limits[0] if limits[0] is not None else _lims[0],
                limits[1] if limits[1] is not None else _lims[1]
            )
        else:
            _lims = ax.get_ylim()
            ax.set_ylim(
                limits[0] if limits[0] is not None else _lims[0],
                limits[1] if limits[1] is not None else _lims[1]
            )
    else:
        if axis == 'x':
            ax.set_xlim(*limits)
        else:
            ax.set_ylim(*limits)


def plot_ampspec(ax, freq, amp, f_peak, name=None):
    '''
    plot_ampspec (C) aadm 2016-2018
    Plots amplitude spectrum calculated with mpfullspec (aageofisica.py).

    INPUT
    ax, axis object. If None, creates a new figure
    freq: frequency
    amp: amplitude
    f_peak: average peak frequency
    '''
    db = 20 * np.log10(amp)
    db = db - np.amax(db)
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), facecolor='w')
    ax[0].plot(freq, amp, '-k', lw=2)
    ax[0].set_ylabel('Power')
    ax[1].plot(freq, db, '-k', lw=2)
    ax[1].set_ylabel('Power (dB)')
    for aa in ax:
        aa.set_xlabel('Frequency (Hz)')
        aa.set_xlim([0, np.amax(freq) / 1.5])
        aa.grid()
        aa.axvline(f_peak, color='r', ls='-')
        if name != None:
            aa.set_title(name, fontsize=16)


def plot_ampspecs(ax, freq_amp_list, names=None):
    '''Plots overlay of multiple amplitude spectras.

    A variation of:
    plot_ampspec2 (C) aadm 2016-2018
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/playing_with_seismic.ipynb
    which takes a list of multiple freqency-amplitude pairs (with an optional "average peak frequency")

    INPUT
        ax, axis object. If None, creates a new figure

        freq_amp_list: list of
        [frequency (np.ndarray), amplitude spectra (np.ndarray), optional average peak frequency (float)] lists

        names: list of strings, same length as freq_amp_list

    '''
    dbs = []  # list to hold dB values of the amplitude spectra
    for _item in freq_amp_list:
        _db = 20 * np.log10(_item[1])
        dbs.append(_db - np.amax(_db))

    one_axes = True
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), facecolor='w')
        one_axes = False

    labels = None
    if names is not None:
        if len(freq_amp_list) != len(names):
            raise ValueError('Both input lists must have same length')

        labels = []
        for i, _name in enumerate(names):
            _label = '{}'.format(_name)
            if len(freq_amp_list[i]) > 2:
                if (freq_amp_list[i][2] is not None):
                    _label += ' Fp={:.0f} Hz'.format(freq_amp_list[i][2])
            labels.append(_label)
    if labels is None:
        labels = [''] * len(freq_amp_list)

    saved_colors = []
    for i, _item in enumerate(freq_amp_list):
        tc = next_color()
        saved_colors.append(tc)
        if one_axes:
            ax.plot(_item[0], dbs[i], '-{}'.format(tc), lw=2, label=labels[i])
        else:
            ax[0].plot(_item[0], _item[1], '-{}'.format(tc), lw=2, label=labels[i])
            ax[0].fill_between(_item[0], 0, _item[1], lw=0, facecolor=tc, alpha=0.25)
            ax[1].plot(_item[0], dbs[i], '-{}'.format(tc), lw=2, label=labels[i])

    if one_axes:
        lower_limit = np.min(ax.get_ylim())
        ax.fill_between(_item[0], dbs[i], lower_limit, lw=0, facecolor=saved_colors[i], alpha=0.25)
        ax.set_xlabel('Frequency (Hz)')
        ax.grid()
        ax.set_ylabel('Power (dB)')
        ax.legend(fontsize='small')

    else:
        lower_limit = np.min(ax[1].get_ylim())
        for i, _item in enumerate(freq_amp_list):
            ax[1].fill_between(_item[0], dbs[i], lower_limit, lw=0, facecolor=saved_colors[i], alpha=0.25)
            for i, _item in enumerate(freq_amp_list):
                if len(freq_amp_list[i]) > 2:
                    if (freq_amp_list[i][2] is not None):
                        ax.axvline(freq_amp_list[i][2], color=saved_colors[i], ls='-')


        ax[0].set_ylabel('Power')
        ax[1].set_ylabel('Power (dB)')
        for aa in ax:
            aa.set_xlabel('Frequency (Hz)')
            # aa.set_xlim([0,np.amax(freq)/1.5])
            aa.grid()
            for i, _item in enumerate(freq_amp_list):
                 if len(freq_amp_list[i]) > 2:
                    if (freq_amp_list[i][2] is not None):
                        aa.axvline(freq_amp_list[i][2], color=saved_colors[i], ls='-')

            aa.legend(fontsize='small')



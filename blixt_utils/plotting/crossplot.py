# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: crossplot.py
#  Purpose: Plotting cross plots
#   Author: Erik Marten Blixt
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------
"""
Cross plot

Taken from the the infrapy project Blixt wrote earlier, but made independent of the infrapy library

:copyright:
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import sys
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib.font_manager import FontProperties
import logging

from blixt_utils.misc.templates import get_from_template

log = logging.getLogger(__name__)

if sys.version_info > (3, 3):
    pass
else:
    pass

cnames = [tmp['color'] for tmp, j in zip(plt.rcParams['axes.prop_cycle'], range(20))]
#cnames = list(np.roll([str(u) for u in colors.cnames.keys()], 10))

msymbols = np.array(['o','s','v','^','<','>','p','*','h','H','+','x','D','d','|','_','.','1','2','3','4','8'])


def test():
    # Set up a test plot
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

    for ax in [ax1, ax2]:
        xdata = np.random.random(100)
        ydata = np.random.random(100)
        cdata = np.random.random(100)
        #pdata = point_size(np.random.random(100), max_size=200)
        pdata = 50
        cbar = None
        odata = None
        reverse_order = False
        edge_color = True

        cbar = plot(
                    xdata,
                    ydata,
                    cdata=cdata,
                    pdata=pdata,
                    mdata=msymbols[3],
                    ax=ax,
                    cbar=cbar,
                    order_by=odata,
                    reverse_order=reverse_order,
                    edge_color=edge_color,
                    xtempl={'full_name': 'X label',
                            'unit': '-',
                            'min': 0.,
                            'max': 1.},
                    ytempl={'full_name': 'Y label',
                            'unit': '-',
                            'min': 0.,
                            'max': 1.},
                    ctempl={'full_name': 'C label',
                            'unit': '-',
                            'min': 0.,
                            'max': 1.}
                )

    # Handle the legends
    legends = []
    this_legend = ax1.legend(
        legends,
        prop=FontProperties(size='smaller'),
        scatterpoints = 1,
        markerscale=0.5,
        loc=1
    )
    plt.show()

def plot(
        xdata,
        ydata,
        cdata=None, # color the data points
        pdata=None, # determines the point size
        mdata=None,
        title=None,
        xtempl=None,
        ytempl=None,
        ctempl=None,
        ptempl=None,
        xerror=None,
        yerror=None,
        mask=None,
        fig=None,
        ax=None,
        cbar=None,
        order_by=None,
        reverse_order=False,
        show_masked=False,
        edge_color=True,
        pointsize=30,
        **kwargs
):
    """

    :param xdata:
    :param ydata:
    :param cdata:
    :param pdata:
    :param mdata:
        str or list
        determines the markers, can be a single marker string, or a list/array of matplotlib.markers.MarkerStyle
        or string of marker style
    :param title:
    :param xtempl:
        dict
        dictionary with the following keys:
            All are optional
            'bounds': list of limits for a discrete colorscale, Optional
            'center': float, midpoint value of a centered colorscale, Optional
            'colormap': str, name of the colormap (Default 'jet') OR
                list of strings with color names if scale == 'discrete'. e.g.:
                ['b', 'saddlebrown', 'greenyellow', 'mediumpurple', 'red', 'lightgreen', 'yellow']
            'description': str, data description. Not used
            'full_name': str, name of variable. Used in the label
            'id': str, short name of the variable
            'max': max value to be used in the plot
            'min': min value to be used in the plot
            'scale': str, 'lin', 'log', or 'discrete'
            'type':  str, datatype 'str', 'float', 'int'
            'unit': str,
    :param ytempl:
        same as xtempl, but for the y axis
    :param ctempl:
        same as xtempl, but for the color axis
    :param ptempl:
        same as xtempl, but for the size
    :param xerror:
        numpy array of same length as xdata
    :param yerror:
        numpy array of same length as ydata
    :param mask:
        boolean numpy array of same length as xdata
    :param fig:
    :param order_by
        numpy array
        same length as xdata
    :return:
    """

    l_fonts = kwargs.pop('l_fonts', 16)
    t_fonts = kwargs.pop('t_fonts', 13)
    grid = kwargs.pop('grid', True)

    # handle templates
    # xlabel, xlim, xcmap, xcnt, xbnds, xscale = handle_template(xtempl)
    # ylabel, ylim, ycmap, ycnt, ybnds, yscale = handle_template(ytempl)
    # clabel, clim, ccmap, ccnt, cbnds, cscale = handle_template(ctempl)
    # plabel, plim, pcmap, pcnt, pbnds, pscale = handle_template(ptempl)
    xlabel, xlim, xline_style, xplot_style = get_from_template(xtempl)
    ylabel, ylim, yline_style, yplot_style = get_from_template(ytempl)
    clabel, clim, cline_style, cplot_style = get_from_template(ctempl)
    plabel, plim, pline_style, pplot_style = get_from_template(ptempl)

    # Handle mask
    if mask is None:
        mask = np.ones((len(xdata)), dtype=bool)
    else:
        if len(mask) != len(xdata):
            raise OSError('Length of mask must match input data')

    import matplotlib.markers as mmarkers

    # set up plotting environment
    if fig is not None:
        print("WARNING: The fig parameter is deprecated, the figure instance is automatically picked from the ax object")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # handle markers
    msymbol = None
    if mdata is None:
        msymbol = 'o'
    elif (mdata is not None) and isinstance(mdata, str):
        msymbol = mdata
    # The case:
    # if (mdata is not None) and (not isinstance(mdata,str)):
    # is being handled after plotting

    # ordering of points is by default largest on top, but 'reverse_order' == True
    # reverse the ordering
    if (order_by is not None) and (len(order_by) != len(xdata)):
        raise OSError('Length of data used for ordering has wrong size')

    # if ordering data by a specific dataset
    if order_by is not None:
        if reverse_order:
            odi = np.argsort(order_by[mask])[::-1]
            nodi = np.argsort(order_by[~mask])[::-1]
        else:
            odi = np.argsort(order_by[mask])
            nodi = np.argsort(order_by[~mask])
    else:
        odi = np.arange(len(xdata[mask]))
        nodi = np.arange(len(xdata[~mask]))

    if (cdata is not None) and (not isinstance(cdata, str)):
        cdata = cdata[mask][odi]
    if isinstance(pdata, np.ndarray):
        if plim[1] is not None:
            try:
                npdata = pdata[~mask][nodi] * pointsize / plim[1]
            except ValueError:
                npdata = None
            pdata = pdata[mask][odi] * pointsize / plim[1]
        else:
            try:
                npdata = pdata[~mask][nodi] * pointsize / np.nanmedian(pdata[mask])
            except ValueError:
                npdata = None
            # pdata = pdata[mask][odi] * pointsize / np.nanmedian(pdata[mask])
            pdata = point_size(pdata[mask][odi])
    else:
        pdata = pointsize
        npdata = pointsize

    if edge_color:
        edge_color = 'b'
    else:
        edge_color = 'none'

    #
    # start cross plotting
    #

    # plot errorbars
    if xerror is not None:
        # if xscale == 'log':
        if xplot_style['scale'] == 'log':
            raise NotImplementedError('Error bars on logarithmic scale needs testing before relase')
        ax.errorbar(
            xdata[mask][odi],
            ydata[mask][odi],
            xerr=xerror[mask][odi],
            capthick=1,
            elinewidth=0.5,
            linestyle='none',
            zorder=-100.
        )

    if yerror is not None:
        if yplot_style['scale'] == 'log':
            raise NotImplementedError('Error bars on logarithmic scale needs testing before relase')
        ax.errorbar(
            xdata[mask][odi],
            ydata[mask][odi],
            yerr=yerror[mask][odi],
            capthick=1,
            elinewidth=0.5,
            linestyle='none',
            zorder=-100.
        )

    #  plot masked data in gray
    if show_masked:
        im = ax.scatter(
            xdata[~mask][nodi],
            ydata[~mask][nodi],
            c='0.4',
            s=npdata,
            marker=msymbol,
            edgecolor='none',
            alpha=0.3,
            label='_nolegend_'
        )
    # Then plot the remaining points
    im = ax.scatter(
        xdata[mask][odi],
        ydata[mask][odi],
        s=pdata,
        c=cdata,
        # cmap=ccmap,
        cmap=cplot_style['cmap'],
        vmin=clim[0],
        vmax=clim[1],
        marker=msymbol,
        picker=5,
        edgecolor=edge_color,
        **kwargs
    )

    ax.grid(grid)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if title is not None:
        ax.set_title(title)

    ax.set_xlabel(xlabel, fontsize=l_fonts)
    ax.set_ylabel(ylabel, fontsize=l_fonts)

    # when mdata is a list / array of markers, handle this
    if (mdata is not None) and (not isinstance(mdata,str)):
        if len(mdata) == len(xdata):
            paths = []
            for marker in mdata:
                if isinstance(marker, mmarkers.MarkerStyle):
                    msymbol = marker
                else:
                    msymbol = mmarkers.MarkerStyle(marker)
                path = msymbol.get_path().transformed(
                    msymbol.get_transform()
                )
                paths.append(path)
            im.set_paths(paths)

    if (cbar is None) and not isinstance(cdata, str):  # don't draw a colorbar for single colored data
        print('Creating colorbar')
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=t_fonts)
        cbar.ax.set_ylabel(clabel, fontsize=l_fonts)
    else:
        pass
        #cbar.draw_all()
        #cbar.set_clim(vmin=clim[0], vmax=clim[1]) # causes sometimes a problem with the clim and "ylim" of the colorbar being different
    ax.tick_params(axis='both', labelsize=t_fonts)

    if xplot_style['scale'] == 'log':
        ax.set_xscale('log')
    if yplot_style['scale'] == 'log':
        ax.set_yscale('log')

    return cbar


def normalize(data):
    this_min = np.nanmin(data)
    this_max = np.nanmax(data)
    return (data - this_min) / (this_max - this_min)


def point_size(data, min_size=40., max_size=500., scale='lin'):
    increase = max_size - min_size
    if scale == 'lin':
        return min_size + normalize(data) * increase

    elif scale == 'square':
        return min_size + (normalize(data) * np.sqrt(increase)) ** 2

    elif scale == 'log':
        return min_size + np.exp(
            normalize(data) * np.log(increase))
    else:
        return None


if __name__ == '__main__':
    test()

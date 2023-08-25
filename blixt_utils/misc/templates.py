# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: templates.py
#  Purpose: Handling  templates used in various plotting scripts
#   Author: Erik Marten Blixt
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------

necessary_keys = ['full_name', 'unit', 'min', 'max', 'colormap', 'center', 'bounds', 'scale',
                  'line color', 'line style', 'line width', 'marker']

default_templates = {
    'Resistivity': {'bounds': None,
                    'center': None,
                    'colormap': 'jet',
                    'description': None,
                    'max': 200.0,
                    'min': 0.2,
                    'scale': 'log',
                    'type': 'float',
                    'unit': '$\\Omega m$',
                    'line color': '#1f77b4',
                    'line style': '-',
                    'line width': 1,
                    'marker': None,
                    'full_name': 'Resistivity'},

    'Sonic': {'bounds': None,
              'center': None,
              'colormap': 'jet',
              'description': None,
              'max': 120.0,
              'min': 60.0,
              'scale': 'lin',
              'type': 'float',
              'unit': '$\\mu s/F$',
              'line color': '#2ca02c',
              'line style': '-',
              'line width': 1,
              'marker': None,
              'full_name': 'Sonic'},

    'TOC': {'bounds': None,
            'center': None,
            'colormap': 'jet',
            'description': None,
            'max': 5.0,
            'min': 1.0,
            'scale': 'lin',
            'type': 'float',
            'unit': '%',
            'line color': '#9467bd',
            'line style': '-',
            'line width': 1,
            'marker': '*',
            'full_name': 'TOC'},

    '34_5_1A': {'color': '#9467bd',
                'symbol': '>',
                'content': 'Dry',
                'kb': None,
                'uwi': None,
                'utm': None,
                'x': None,
                'y': None,
                'water depth': None,
                'note': 'Blåbær sidetrack, Old CPI and covers only Cook, No LFP. Maybe need a new well tie'}
}


def return_empty_template():
    return {key: None for key in necessary_keys}


def log_header_to_template(log_header):
    """
    Returns a template dictionary from the information in the log header.
    :param log_header:
        core.log_curve.Header
    :return:
        dict
        Dictionary contain template information used by crossplot.py
    """
    tdict = {
        'id': log_header.name if log_header.name is not None else '',
        'description': log_header.desc if log_header.desc is not None else '',
        'full_name':
            '{}, {}'.format(
                log_header.log_type, log_header.name) if log_header.log_type is not None else log_header.name,
        'type': 'float',
        'unit': log_header.unit if log_header.unit is not None else ''
    }
    return tdict


def get_from_template(template_dict: dict):
    """

    :param template_dict:
        A template dictionary for a specific logtype, e.g. default_templates['Resistivity']
    :return:
        label, limits, line_style, plot_style
    """
    label = ''
    limits = [None, None]
    line_style = {'lw': None, 'color': None, 'ls': None}
    plot_style = {'cmap': None, 'cnt': None, 'bnds': None, 'scale': None}

    if template_dict is None:
        return label, limits, line_style, plot_style

    key_list = list(template_dict.keys())

    if 'full_name' in key_list:
        label += template_dict['full_name']
    if 'unit' in key_list:
        label += ' [{}]'.format(template_dict['unit'])
    if 'min' in key_list:
        try:
            limits[0] = float(template_dict['min'])
        except TypeError:
            limits[0] = None
    if 'max' in key_list:
        try:
            limits[1] = float(template_dict['max'])
        except TypeError:
            limits[1] = None

    if 'line width' in key_list:
        line_style['lw'] = template_dict['line width']
    if 'line color' in key_list:
        line_style['color'] = template_dict['line color']
    if 'line style' in key_list:
        line_style['ls'] = template_dict['line style']

    if 'colormap' in key_list:
        plot_style['cmap'] = template_dict['colormap']
    if 'center' in key_list:
        plot_style['cnt'] = template_dict['center']
    if 'bounds' in key_list:
        plot_style['bnds'] = template_dict['bounds']
    if 'scale' in key_list:
        plot_style['scale'] = template_dict['scale']

    return label, limits, line_style, plot_style


def handle_template(template_dict):
    """
    Goes through the given template dictionary and returns values directly useful for the scatter plot
    :param template_dict:
        see x[y,c]templ in calling functions
    :return:
    label, lim, cmap, cnt, bnds, scale
    """
    label = ''
    lim = [None, None]
    cmap = None
    cnt = None
    bnds = None
    scale = 'lin'
    if isinstance(template_dict, dict):
        key_list = list(template_dict.keys())

        if 'full_name' in key_list:
            label += template_dict['full_name']
        if 'unit' in key_list:
            label += ' [{}]'.format(template_dict['unit'])
        if 'min' in key_list:
            try:
                lim[0] = float(template_dict['min'])
            except:
                lim[0] = None
        if 'max' in key_list:
            try:
                lim[1] = float(template_dict['max'])
            except:
                lim[1] = None
        if 'colormap' in key_list:
            cmap = template_dict['colormap']
        if 'center' in key_list:
            cnt = template_dict['center']
        if 'bounds' in key_list:
            bnds = template_dict['bounds']
        if 'scale' in key_list:
            scale = template_dict['scale']

    elif template_dict is None:
        pass
    else:
        raise OSError('Template should be a dictionary')

    return label, lim, cmap, cnt, bnds, scale


def test_template(template_dict):
    if isinstance(template_dict, dict):
        key_list = list(template_dict.keys())
        for key in necessary_keys:
            if key not in key_list:
                print('{} is missing in template dictionary'.format(key))
                return False
        print('Template dictionary is ok')
        return True
    else:
        print('Input template is not a dictionary')
        return False


def test():
    simple_templates = default_templates
    simple_templates['XXX'] = {'A': 'green'}

    test_template(simple_templates['Sonic'])
    test_template(simple_templates['XXX'])
    test_template('No input')
    test_template(return_empty_template())


if __name__ == '__main__':
    test()

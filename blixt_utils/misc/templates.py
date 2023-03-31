# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# Filename: templates.py
#  Purpose: Handling  templates used in various plotting scripts
#   Author: Erik Marten Blixt
#   Email: marten.blixt@gmail.com
#
# --------------------------------------------------------------------

necessary_keys = ['full_name', 'unit', 'min', 'max', 'colormap', 'center', 'bounds', 'scale']


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
    simple_templates = {
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
        'XXX': {
            'A': 'green'
        },

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
    test_template(simple_templates['Sonic'])
    test_template(simple_templates['XXX'])
    test_template('No input')
    test_template(return_empty_template())


if __name__ == '__main__':
    test()

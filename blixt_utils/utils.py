# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 08:45:27 2020

@author: mblixt
"""

import logging
import sys
import os
import subprocess
import getpass
import socket
from datetime import datetime
import numpy as np
import re

import blixt_utils.misc.masks as msks


def arrange_logging(log_to_stdout, log_to_this_file, level=logging.INFO):
    """
    :param log_to_stdout:
        bool
        If True, the logging is output to stdout
    :param log_to_this_file:
    :param level:
    :return:
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    frmt = '%(asctime)s | %(filename)s - %(funcName)s: %(levelname)s:%(message)s'
    if log_to_stdout:
        logging.basicConfig(stream=sys.stdout,
                            format=frmt,
                            level=level,
                            datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logfile = log_to_this_file
        logging.basicConfig(filename=logfile,
                            format=frmt,
                            level=level,
                            datefmt='%Y-%m-%d %H:%M:%S')


def gitversion():
    thisgit =  os.path.realpath(__file__).replace('rp_utils/returnVersion.py','.git')
    tagstring = ''
    hashstring = ''

    def cleanstring(text):
        text = text.decode("utf-8")
        text = text.rstrip('\\n')
        text = text.rstrip('\n')
        return text


    # extract the git tag
    try:
        tagstring = subprocess.check_output(['git', '--git-dir=%s' % thisgit,  'describe',
                                             '--abbrev=0'])
        #                                    '--exact-match', '--abbrev=0'])
        tagstring = cleanstring(tagstring)
    except:
        tagstring = 'Unknown tag'

    # extract the git hash
    try:
        hashstring = subprocess.check_output(['git', '--git-dir=%s' % thisgit,  'rev-parse', '--short', 'HEAD'])
        hashstring = cleanstring(hashstring)
    except:
        hashstring = 'Unknown hash'

    return 'Git: %s, %s' % (tagstring, hashstring)


def svnversion():
    try:
        svn_info = subprocess.check_output(['svn', 'info']).decode("utf-8")
    except:
        return '??'
    t = svn_info.split('\n')
    for row in t:
        if 'Revision:' in row:
            return row


def nan_corrcoef(x,y):
    maskx = ~np.ma.masked_invalid(x).mask
    masky = ~np.ma.masked_invalid(y).mask
    mask = msks.combine_masks([maskx, masky], verbose=False)
    return np.corrcoef(x[mask], y[mask])


def isnan(val):
    """
    Can test both numbers and strings if they are nan's.

    :param val:
    :return:
        bool
    """
    if isinstance(val, str):
        return False
    try:
        return np.isnan(val)
    except:
        raise NotImplementedError('Cant handle this input properly')


def conv_tops_to_wis(tops, intervals):
    """
    Convert a set of tops and intervals to "working intervals"

    :param tops:
        dict
        {'Well_name':
            {'top_A': top depth,
             'base_A': top depth,
             'top_B': top depth,
             'base_A': top depth},
         ...
        }
    :param intervals:
        list
        [{'name': 'w_interval_A',
             'tops': ['top_A', 'base_A']},
         {'name': 'w_interval_B',
             'tops': ['top_B', 'base_B']}]

    :return:
    working_intervals
    dict
    {'Well name':
        {'w_interval_A': [top depth, base depth],
         'w_interval_B': [top depth, base depth],
         ...
        },
     ...
    }
    """
    working_intervals = {}
    for wname in list(tops.keys()):
        working_intervals[wname] = {}
        for i, iname in enumerate([x['name'] for x in intervals]):
            top_name = intervals[i]['tops'][0]
            base_name = intervals[i]['tops'][1]
            if top_name.upper() not in list(tops[wname].keys()):
                continue
            if base_name.upper() not in list(tops[wname].keys()):
                continue
            working_intervals[wname][iname] = [tops[wname][top_name.upper()], tops[wname][base_name.upper()]]
    return working_intervals


def norm(arr, method='median'):
    if method == 'mean':
        x0 = np.nanmean(arr)
    elif method == 'median':
        x0 = np.nanmedian(arr)
    else:
        x0 = np.nanmin(arr)

    return (arr - x0) / np.abs(np.nanmax(arr) - np.nanmin(arr))


def handle_sonic(well):
    raise NotImplementedError('Please use native well method, sonic_to_vel ')


def log_table_in_smallcaps(log_table):
    for key in list(log_table.keys()):
        if isinstance(log_table[key], list):
            print('TRUE')
            log_table[key] = [_item.lower() for _item in log_table[key]]
        else:
            log_table[key] = log_table[key].lower()
    return log_table


def mask_string(cutoffs, wi_name):
    msk_str = ''
    for key in list(cutoffs.keys()):
        msk_str += '{}: {} [{}]'.format(
            key, cutoffs[key][0], ', '.join([str(m) for m in cutoffs[key][1]])) if \
            isinstance(cutoffs[key][1], list) else \
            '{}: {} {}, '.format(
                key, cutoffs[key][0], cutoffs[key][1])
    if (len(msk_str) > 2) and (msk_str[-2:] == ', '):
        msk_str = msk_str.rstrip(', ')
    if wi_name is not None:
        msk_str += ' Working interval: {}'.format(wi_name)
    return msk_str


def print_info(
        text: str,
        level: str,
        # logger: logging.Logger | None,
        logger: logging.Logger,
        verbose=True,
        to_logger=True,
        raiser: str | None = None
):
    """
    Prints a string to standard output when verbose is True, and to logger, and can raise errors

    :param text:
        str
        The text that should be printed
    :param level:
        str
        'info', 'warning', 'debug', 'error', 'exception'
    :param logger:
        Logger
        Logger that handel the information. See arrange_logging
    :param verbose:
        bool
        Default True
    :param to_logger:
        bool
        Default True
    :param raiser:
        str
        If provided, the type of error that should be raised
    :return:
    """
    if verbose:
        print('{}: {}'.format(level.upper(), text))
    if to_logger:
        if level.lower() in ['info', 'information']:
            logger.info(text)
        elif level.lower() == 'warning':
            logger.warning(text)
        elif level.lower() == 'debug':
            logger.debug(text)
        elif level.lower() == 'error':
            logger.error(text)
        elif level.lower() == 'exception':
            logger.exception(text)
        else:
            raise IOError('Logger level: {}, not recognized'.format(level.upper()))
    if raiser is not None:
        if raiser.lower() == 'ioerror':
            raise(IOError(text))
        elif raiser.lower() == 'notimplementederror':
            raise(NotImplementedError(text))
        elif raiser.lower() == 'oserror':
            raise (OSError(text))
        elif raiser.lower() == 'typeerror':
            raise(TypeError(text))


def add_one(in_string):
    trailing_nr = re.findall(r"\d+", in_string)
    if len(trailing_nr) > 0:
        new_trail = str(int(trailing_nr[-1]) + 1)
        in_string = in_string.replace(trailing_nr[-1], new_trail)
    else:
        in_string = in_string + ' 1'
    return in_string


def fix_well_name(well_name):
    if well_name is None:
        return
    if isinstance(well_name, str):
        return well_name.replace('/', '_').replace('-', '_').replace(' ', '').upper()
    else:
        return


def cycle_colors():
    interval_colors = ['#E3F917', '#17becf']
    # lazy iterator over two colors
    # > next()
    i = 0
    while True:
        if np.mod(i, 2) == 0:
            yield interval_colors[0]
        else:
            yield interval_colors[1]
        i += 1


def find_value(x: np.ndarray, x_index: int, snap_to: str = 'exact') -> (float, int):
    """
    Returns the value of the array x at index position x_index, when 'snap_to' is 'exact'
    But can return the nearest min or max, ... or TODO
    :param x:
    :param x_index:
    :param snap_to:
        'exact': returns (x[x_index], x_index)
        'nearest_min': returns the local minimum closest to x_index and its index
        'nearest_max': returns the local maximum closest to x_index and its index
        'nearest_extreme': returns the local extreme closest to x_index and its index
    :return:
        (value, index_position)
    """
    from scipy.signal import argrelmax, argrelmin
    if snap_to == 'exact':
        return x[x_index], x_index
    elif snap_to == 'nearest_min':
        local_minima = argrelmin(x)[0]
        closest_minima = np.argmin(np.sqrt((local_minima - x_index)**2))
        x_index = local_minima[closest_minima]
        return x[x_index], x_index
    elif snap_to == 'nearest_max':
        local_max = argrelmax(x)[0]
        closest_max = np.argmin(np.sqrt((local_max - x_index)**2))
        x_index = local_max[closest_max]
        return x[x_index], x_index
    elif snap_to == 'nearest_extreme':
        local_minima = argrelmin(x)[0]
        local_max = argrelmax(x)[0]
        closest_minima = np.argmin(np.sqrt((local_minima - x_index)**2))
        closest_max = np.argmin(np.sqrt((local_max - x_index) ** 2))
        print(x_index, local_minima[closest_minima], local_max[closest_max])
        if abs(x_index - local_minima[closest_minima]) < abs(x_index - local_max[closest_max]):
            x_index = local_minima[closest_minima]
        else:
            x_index = local_max[closest_max]
        return x[x_index], x_index
    else:
        raise IOError('Unknown snap_to: {}'.format(snap_to))


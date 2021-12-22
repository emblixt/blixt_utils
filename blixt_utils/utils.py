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

    frmt = '%(filename)s - %(funcName)s: %(levelname)s:%(message)s'
    if log_to_stdout:
        logging.basicConfig(stream=sys.stdout,
                            format=frmt,
                            level=level)
    else:
        logfile = log_to_this_file
        logging.basicConfig(filename=logfile,
                            format=frmt,
                            level=level)


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
    mask = msks.combine_masks([maskx, masky])
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

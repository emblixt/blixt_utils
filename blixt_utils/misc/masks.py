# -*- coding: utf-8 -*-
"""
Created on 2019-01-23 Mårten Blixt
Set of tools for handling masks (originally for infrasound data)
"""

import numpy as np
import logging

import pint

logger = logging.getLogger(__name__)

def create_mask(data, operator, limits):
    """
    Given input data, mask operator with limits, this routine returns the boolean mask
    Example
    t = np.arange(10)
    mask = create_mask(t,'>=',8)
    print(t[mask])
        array([8, 9])
    :param data:
        1D numpy array of data
    :param operator:
        string
        representing the masking operation
        '<':  masked_less
        '<=': masked_less_equal
        '>':  masked_greater
        '>=': masked_greater_equal
        '><': masked_inside
        '==': masked_equal
        '!=': masked_not_equal
    :param limits:
        float, or list of floats
    :return:
        numpy boolean mask array
        True values indicate that this data point fulfills the operator with the given limits
    """
    from blixt_rp.core.log_curve_new import Depth
    from blixt_utils.utils import print_info
    if  not (isinstance(data, np.ndarray) or isinstance(data, pint.Quantity) or isinstance(data, Depth)):
        print_info('Only numpy ndarray, or pint quantities are allowed as data input, not {}'.format(type(data)),'error', logger, raiser='IOError')

    # convert limits to a list of limits
    if not isinstance(limits, list) or isinstance(limits, pint.Quantity):
        limits = [limits]

    # If data is a pint.Quantity, the limits must also be pint.Quantities
    # We try to convert everything to the units of the data, and continue doing the calculation on
    # converted data without units
    if isinstance(data, pint.Quantity) or isinstance(data, Depth):
        data_unit = data.units
        data = data.magnitude
        limits = [_x.to(data_unit).magnitude for _x in limits]


    logging.debug(
        'Masking data of length {}, using the operator: {} with limits: {}'.format(
        format(str(len(data))),
        format(operator),
        ', '.join([str(x) for x in limits])
    ))

    if operator == '<':
        mask = np.ma.masked_less(data, limits[0]).mask
    elif operator == '<=':
        mask = np.ma.masked_less_equal(data, limits[0]).mask
    elif operator == '>':
        mask = np.ma.masked_greater(data, limits[0]).mask
    elif operator == '>=':
        mask = np.ma.masked_greater_equal(data, limits[0]).mask
    elif operator == '><':
        mask = np.ma.masked_inside(data, *limits).mask
    elif operator == '==':
        mask = np.ma.masked_equal(data, limits[0]).mask
    elif operator == '!=':
        mask = np.ma.masked_not_equal(data, limits[0]).mask
    else:
        raise OSError('Could not match ' + operator + ' with any valid operator')

    if type(mask) == np.bool_: # Failed to find any data in the mask
        mask = np.array(np.zeros(len(data)), dtype=bool)

    return mask


def combine_masks(masks, combine_operator='AND', verbose=False):
    """
    Combine masks of equal length, using and or or
    :param masks:
        list
        list of boolean masks
    :param combine_operator:
        string
        'AND' or 'OR'
    :return:
        boolean mask with input masks combined
    """

    # convert input to list if they're not
    if not isinstance(masks, list):
        masks = [masks]

    # start testing input
    length = 0
    for i, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            raise OSError('Mask {} is not a numpy ndarray'.format(i))
        if i == 0:
            length = len(mask)
            if verbose:
                print('- combine_masks: {} len({})'.format(i, length))
        else:
            if verbose:
                print('- combine_masks: {} len({})'.format(i, len(mask)))
            if len(mask) != length:
                raise OSError('Mask {} has different length'.format(i))
        if mask.dtype != np.dtype('bool'):
            raise OSError('Mask {} is not a boolean mask'.format(i))


    # start combining masks
    last_mask = False
    for i, mask in enumerate(masks):
        if i == 0:
            last_mask = mask
        else:
            if combine_operator == 'AND':
                this_mask = np.array([all(m) for m in zip(mask, last_mask)], dtype=bool)
            elif combine_operator == 'OR':
                this_mask = np.array([any(m) for m in zip(mask, last_mask)], dtype=bool)
            else:
                raise OSError('Only AND or OR combination of masks are implemented')
            last_mask = this_mask

    return last_mask
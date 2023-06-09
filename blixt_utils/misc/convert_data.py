import numpy as np
import logging

logger = logging.getLogger(__name__)

bar_to_gpa = 0.0001
bar_to_mpa = 0.1
bar_per_m_to_gr_cm3 = 10.2
m_to_ft = 3.28084


def convert(in_data, from_unit=None, to_unit=None):
    """
    Converts the in_data from unit 'from_unit' to 'to_unit'.

    :param in_data:
        np.array
    :param from_unit:
        str
    :param to_unit:
        str
    :return:
        Bool, np.array
    """
    # Clean up from_ and to_ units
    def clean_up(these_units):
        these_units = these_units.replace('$', '')
        these_units = these_units.replace('^', '')
        these_units = these_units.replace('.', ' ')
        return these_units.lower()

    from_unit = clean_up(from_unit)
    to_unit = clean_up(to_unit)

    # Test if units are specified and if they differ
    success = True
    wrn_txt = ''
    if from_unit is None:
        wrn_txt = "No 'from_unit' specified. No conversion done"
        success = False
    elif to_unit is None:
        wrn_txt = "No 'to_unit' specified. No conversion done"
        success = False
    elif from_unit == to_unit:
        wrn_txt = "'from_unit': {}, equals 'to_unit': {}. No conversion done".format(from_unit, to_unit)
        success = False
    if not success:
        logger.warning(wrn_txt)
        return success, in_data

    # When units are specified and different, try use them for converting
    success = True
    if (from_unit == 'ft' or from_unit == 'feet') and to_unit == 'm':
        return success, in_data / m_to_ft
    if (from_unit == 'ft' or from_unit == 'feet') and to_unit == 'km':
        return success, in_data / (1000. * m_to_ft)

    elif from_unit == 'm' and to_unit == 'ft':
        return success, in_data * m_to_ft

    elif from_unit == 'us/ft' and to_unit == 'm/s':
        return success, 1. / (m_to_ft * 1.E-6 * in_data)
    elif from_unit == 'us/ft' and to_unit == 'km/s':
        return success, 1. / (m_to_ft * 1.E-3 * in_data)

    elif from_unit == 'g/cm3' and to_unit == 'kg/m3':
        return success, 1000. * in_data
    elif from_unit == 'g/cm3' and to_unit == 'mpa/m':
        return success, 0.0098 * in_data

    elif from_unit == 'bar' and to_unit == 'mpa':
        return success, bar_to_mpa * in_data
    elif from_unit == 'bar' and to_unit == 'gpa':
        return success, bar_to_gpa * in_data

    elif from_unit == 'mpa' and to_unit == 'bar':
        return success, in_data / bar_to_mpa

    elif from_unit == 'gpa' and to_unit == 'bar':
        return success, in_data / bar_to_gpa

    elif from_unit == 'ms' and to_unit == 's':
        return success, in_data / 1000.

    elif from_unit == 's' and to_unit == 'ms':
        return success, in_data * 1000.

    else:
        success = False
        wrn_txt = "No valid combination of units specified (from {} to {}). No conversion done".format(
            from_unit, to_unit)
        logger.warning(wrn_txt)
        return success, in_data

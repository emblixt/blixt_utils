import numpy as np
import logging

logger = logging.getLogger(__name__)

bar_to_gpa = 0.0001
bar_to_mpa = 0.1
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
        np.array
    """
    # Test if units are specified
    if from_unit is None:
        logger.warning("No 'from_unit' specified, conversion failed")
        return
    elif to_unit is None:
        logger.warning("No 'to_unit' specified, conversion failed")
        return

    success = True
    # Start converting
    if from_unit == 'ft' or from_unit == 'feet':
        if to_unit == 'm':
            return in_data / m_to_ft
        elif to_unit == 'km':
            return in_data / (1000. * m_to_ft)
        else:
            success = False

    if from_unit == 'm':
        if to_unit == 'ft':
            return in_data * m_to_ft
        else:
            success = False

    if from_unit == 'us/ft':
        # clean up data
        #in_data[in_data < 20.] = np.nan
        #in_data[in_data > 300.] = np.nan
        if to_unit == 'm/s':
            return 1. / (m_to_ft * 1.E-6 * in_data)
        elif to_unit == 'km/s':
            return 1. / (m_to_ft * 1.E-3 * in_data)
        else:
            success = False

    if from_unit == 'g/cm3':
        if to_unit == 'kg/m3':
            return 1000. * in_data
        elif to_unit == 'mpa/m':
            return 0.0098 * in_data
        else:
            success = False

    if from_unit == 'bar':
        if to_unit == 'mpa':
            # return 0.1 * in_data
            return bar_to_mpa * in_data
        else:
            success = False

    if from_unit == 'mpa':
        if to_unit == 'bar':
            return in_data / bar_to_mpa
        else:
            success = False

    if from_unit == 'bar':
        if to_unit == 'gpa':
            return bar_to_gpa * in_data
        else:
            success = False

    if from_unit == 'gpa':
        if to_unit == 'bar':
            return in_data / bar_to_gpa
        else:
            success = False

    if not success:
        wrn_txt = "No valid combination of units specified (from {} to {}), conversion failed".format(
            from_unit, to_unit)
        logger.warning(wrn_txt)
        raise Warning(wrn_txt)


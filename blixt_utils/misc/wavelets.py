# some small wrappers so that we return a wavelet in a similar fashion as how
# read_petrel_wavelet() does when return_dict is true

import matplotlib.pyplot as plt
from blixt_utils.plotting.helpers import wavelet_plot


def ricker(duration, time_step, central_frequency):
    """

    :param duration:
        time duration of wavelet, in seconds
    :param time_step:
        time step of wavelet, in seconds
    :param central_frequency:
        Central frequency, Hz
    :return:
        dict
        {'wavelet':
            wavelet amplitudes,
         'time':
            time in seconds,
         'header':
            dict with info about wavelet
        }
    """
    from bruges.filters import ricker as bruges_ricker
    _wavelet = bruges_ricker(duration, time_step, central_frequency)
    return {
        'wavelet': _wavelet.amplitude,
        'time': _wavelet.time,
        'header': {
            'Name': 'Ricker',
            'Sample rate': time_step,
            'Original filename': 'None, from bruges library',
            'Normalized': False,
            'Scale factor': 1.,
            'Converted to zero phase': False,
            'Time shift': 0.,
            'Resampled': False
        }
    }


if __name__ == '__main__':
    my_wavelet = ricker(0.128, 0.001, 20.)
    wavelet_plot(None, my_wavelet['time'], my_wavelet['wavelet'], my_wavelet['header'])

    plt.show()
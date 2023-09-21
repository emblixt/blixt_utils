# some small wrappers so that we return a wavelet in a similar fashion as how
# read_petrel_wavelet() does when return_dict is true

import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

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


def convolve_with_refl(wavelet_amp, reflectivity, verbose=False):
    """
    Wrapper function that convolve a wavelet with a reflectivity function (returns an array of reflectivity values for
    given incidence angle) and adjusts the length of the output to match the length of the input (n
    :param wavelet_amp:
        np.array
        Amplitude values of the wavelet. E.G. wavelet['wavelet'] for the wavelet defined here
    :param reflectivity:
        np.array
        An np.array of reflectivity values
    :param verbose:
        bool
    :return:
    """
    n = len(reflectivity)
    wiggle = np.convolve(wavelet_amp, np.nan_to_num(reflectivity), mode='same')
    while len(wiggle) < n:
        wiggle = np.append(wiggle, np.ones(1) * wiggle[-1])  # extend with one item
        if verbose:
            print('Added one step to wiggle')
    if n < len(wiggle):
        if verbose:
            print('Removed {} steps to wiggle'.format(len(wiggle) - n))
        # try to remove one step at a time from each end of the wiggle
        _wiggle =  deepcopy(wiggle)
        for i in range(len(wiggle) - n):
            if np.mod(i, 2) == 0:
                _wiggle = _wiggle[1:]
            else:
                _wiggle = _wiggle[:-1]
        wiggle = _wiggle

    if verbose:
        fig, ax = plt.subplots()
        ax.plot(reflectivity)
        ax.plot(wiggle)
    return wiggle


def test_convolve():
    time_step = 0.001
    depth_in_time = 0.1
    n_samples = int(depth_in_time/time_step)
    refl = np.zeros(n_samples)
    refl[int(0.5 * n_samples)] = 1.
    my_wavelet = ricker(0.128, time_step, 20.)
    wiggle = convolve_with_refl(my_wavelet['wavelet'], refl, verbose=True)


if __name__ == '__main__':
    # my_wavelet = ricker(0.128, 0.001, 20.)
    # wavelet_plot(None, my_wavelet['time'], my_wavelet['wavelet'], my_wavelet['header'])

    test_convolve()
    plt.show()

﻿import numpy
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def rolling_window(a, window_len):
    """
    After
    https://github.com/seg/tutorials-2014/blob/master/1406_Make_a_synthetic/how_to_make_synthetic.ipynb
    :param a:
        numpy.ndarray
    :param window_len:
        int
    :return:
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_len + 1, window_len)
    strides = a.strides + (a.strides[-1],)
    rolled = numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return rolled


def test_rolling():
    fig, ax = plt.subplots()
    t = numpy.linspace(-2, 2, 50)
    ax.plot(t)
    ax.plot(rolling_window(t, 3), c='y')

    plt.show()


def smooth(x, window_len=11, window='hanning', method='convolution', interp_vec=None, **kwargs):
    """smooth the data using a window with requested size.

    This method is based on the code given here:
    https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Modifed to use not only convolution of a scaled window with the signal, but also include a
    running median filter and cubic spline interpolation.

    Under convolution, the signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.


    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

        method:
          'convolution' or 'median' or 'cubic_spline'

        interp_vec:
            numpy.ndarray
            when method is set to 'cubic_spline', the input signal is interpolated along this vector

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    padding_mode = kwargs.pop('padding_mode', 'edge')
    padding_reflect_type = kwargs.pop('padding_reflect_type', None)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if method == 'median':
        if np.mod(window_len, 2) == 0:  # avoid even window lengths
            window_len += 1

        _tmp = numpy.median(rolling_window(x, window_len), -1)

        # y = numpy.pad(_tmp, int(window_len/2), mode='edge')  # original
        # y = numpy.pad(_tmp, int(window_len/2), mode='linear_ramp')
        # y = numpy.pad(_tmp, int(window_len/2), mode='reflect')  # poor on sine curves
        # y = numpy.pad(_tmp, int(window_len/2), mode='wrap')  # better than 'reflect' on sine curves
        # y = numpy.pad(_tmp, int(window_len/2), mode='symmetric')  # similar to reflect on sin curves
        # y = numpy.pad(_tmp, int(window_len/2), mode='symmetric', reflect_type='odd')
        # y = numpy.pad(_tmp, int(window_len/2), mode='reflect', reflect_type='odd')  # looks very good on sine curves
        if padding_mode not in ['symmetric', 'reflect']:
            y = numpy.pad(_tmp, int(window_len/2), mode=padding_mode)
        else:
            y = numpy.pad(_tmp, int(window_len/2), mode=padding_mode, reflect_type=padding_reflect_type)

    elif method == 'convolution':
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = numpy.r_[x[window_len-1:0:-1], x, x[-2:-window_len-1:-1]]

        if window == 'flat': #moving average
            w = numpy.ones(window_len, 'd')
        else:
            w = eval('numpy.'+window+'(window_len)')

        y = numpy.convolve(w/w.sum(), s, mode='valid')

    elif method == 'cubic_spline':
        if interp_vec is None:
            raise ValueError("The interpolate vector 'interp_vec' must be given")
        y = interp1d(interp_vec, x, kind='cubic')

    else:
        raise ValueError("Method is one of 'convolution', 'median', 'cubic_spline'")

    # NOTE:
    # length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    if len(y) != len(x):
        return y[round(window_len / 2 - 1):-round(window_len / 2)]
    else:
        return y


def test_smooth():
    fig, ax = plt.subplots()

    t = numpy.linspace(-2, 2, 50)
    x = numpy.sin(2 * t)+numpy.random.random(len(t))*0.1

    ax.plot(t, x, label='Original')

    for c, m in zip(['r', 'y'], ['median', 'convolution']):
        ax.plot(t, smooth(x, method=m, window_len=21), c=c, label=m)

    ax.legend()
    plt.show()


def ampspec(signal, sr, smoothing=None, window_len=None):
    '''
    ampspec (C) aadm 2016
    Calculates amplitude spectrum of a signal with FFT,  optionally smoothed.
    
    Origin:
    https://nbviewer.jupyter.org/github/aadm/geophysical_notes/blob/master/playing_with_seismic.ipynb

    Slightly modified and upgraded by Erik Mårten Blixt 2020-08-19

    :param signal: 
        1D numpy array

    :param sr: 
        float
        sample rate in s

    :param smoothing: 
        str
        'cubic_spline:  smooths the spectrum using cubic interpolation
        'median'

    OUTPUT
    freq: frequency
    amp: amplitude
    '''

    SIGNAL = numpy.fft.fft(signal)
    freq = numpy.fft.fftfreq(signal.size, d=sr)
    keep = freq>=0
    SIGNAL = numpy.abs(SIGNAL[keep])
    freq = freq[keep]
    if smoothing == 'cubic_spline':
        freq0=numpy.linspace(freq.min(),freq.max()/2,freq.size*10)
        #f = interp1d(freq, SIGNAL, kind='cubic')
        f = smooth(SIGNAL, method='cubic_spline', interp_vec=freq)
        return freq0, f(freq0)

    elif smoothing == 'median':
        if window_len is None: 
            window_len = 11
        return freq, smooth(SIGNAL, window_len=window_len, method='median')
        
    else:
        return freq, SIGNAL


def fullspec(data, sr):
    '''
    fullspec (C) aadm 2016-2018
    Calculates amplitude spectrum of 2D numpy array.

    INPUT
    data: 2D numpy array, shape=(traces, samples)
    sr: sample rate in s

    OUTPUT
    freq: frequency
    amp: amplitude
    f_peak: average peak frequency
    '''
    amps, peaks = [], []
    for i in range(data.shape[0]):
        trace = data[i,:]
        freq, amp = ampspec(trace, sr)
        peak = freq[numpy.argmax(amp)]
        amps.append(amp)
        peaks.append(peak)
    amp0 = numpy.mean(numpy.dstack(amps), axis=-1)
    amp0 = numpy.squeeze(amp0)
    f_peak = numpy.mean(peaks)
    print('freq peak: {:.2f} Hz'.format(f_peak))
    return freq, amp0, f_peak


if __name__ == '__main__':
    test_smooth()
    # test_rolling()

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from numpy.random import default_rng


def residuals(x, t, y, target_function=None, weight=None, kwargs=None):
    """
    Returns the residual between the target function and the observed data y
    for a given weight.

    :param x:
        parameters passed to the target function
    :param t:
        np array of length N
        time, or X-axis (independent variable)
          OR
        a (2,N) array
        where N is the total number of data points in y
    :param y:
        np array of length N
        observed data
    :param target_function:
        function to calculate the residual against
            residual = target_function(t, *x) - y
        target_function takes x as arguments, and t as independent variable,
        E.G. for a linear target function:
        def target_function(t, a, b):
            return a*t + b
    :param weight:
        np array of length N
        weights between 0 and 1
        or None
    :param kwargs:
        dictionary
        keywords passed to the target function
    :return:
        np array of length N of residuals
    """
    if kwargs is None:
        kwargs = {}
    if target_function is None:
        raise TypeError('function must be provided to return a residual')

    if len(t.shape) == 2 and t.shape[0] == 2:  # 2D case
        t1, t2 = t
    else:  # assume standard 1D case
        t1 = t; t2 = None

    if weight is None:
        weight = np.ones(t1.shape)
    elif np.nanmin(weight) < 0.:
        raise ValueError('Weights should be larger than 0.')
    elif np.nanmax(weight) > 1.:
        raise ValueError('Weights should not be larger than 1.')
    elif weight.shape != y.shape:
        raise ValueError('Weights must have same dimension as input data')

    if len(t.shape) == 2 and t.shape[0] == 2:  # 2D case
        return weight*(target_function(t1, t2, *x, **kwargs) - y)
    else:  # assume standard 1D case
        return weight * (target_function(t1, *x, **kwargs) - y)


def gen_test_data(x, t, target_function=None, noise=0., n_outliers=0, seed=None):
    """

    :param x:
    :param t:
    :param target_function:
    :param noise:
    :param n_outliers:
    :param seed:
    :return:
    """
    rng = default_rng(seed)
    y = target_function(t, *x)
    y_mean = np.nanmean(y)
    error = noise * rng.standard_normal(t.size)
    outliers = rng.integers(0, t.size, n_outliers)
    error[outliers] *= 10

    return y + error


def linear_function(_t, _a, _b):
    # The linear fitting function
    return _a*_t + _b


def exp_function(_t, _a, _b, _c):
    return _a + _b * np.exp(_c * _t)


def depth_trend(_t, _vp_top, _vp_matrix, _b):
    # Vp depth trend often used by Ikon Science
    # _t is TVD
    return _vp_matrix - (_vp_matrix - _vp_top) * np.exp(-1. * _b * _t)


def play_with_data_fit():
    fig, ax = plt.subplots()

    # Data
    x = [-0.2, 5]  # True parameters of the starting linear function
    # Perturb the linear function with noise
    t = np.linspace(0, 10, 100) + np.random.random(100)  # the independent "time" variable
    y = linear_function(t, *x) + np.random.random(100) - 0.5  # The "observed" data
    # Add data to the dataset through append (out of curiosity)
    t = np.append(t, np.linspace(0, 10, 100) + np.random.random(100))
    y = np.append(y, linear_function(np.linspace(0, 10, 100), *x) + np.random.random(100) - 0.5)

    new_t = np.linspace(0, 10)  # the new independent "time" variable we use to plot the fitted function

    ax.scatter(t, y)

    # initial guess of parameters to the fitting function
    x0 = [1., 1.]

    # least_squares fit with the residual function using the linear function to calculate the error
    res = least_squares(residuals, x0, args=(t, y), kwargs={'target_function': linear_function}, verbose=2)
    res2 = least_squares(residuals, x0, args=(t, y), kwargs={'target_function': linear_function},
                         loss='cauchy', verbose=2)

    # Try with weights
    weights = np.ones(len(t))
    weights[y > 4.5] = 0  # ignore data points above
    weights[y < 3.0] = 0  # ignore data points below
    res3 = least_squares(residuals, x0, args=(t, y), kwargs={'target_function': linear_function, 'weight': weights}, verbose=2)

    print(x); print(res.x); print(res2.x); print(res3.x)

    ax.plot(new_t, linear_function(new_t, *x),
            new_t, linear_function(new_t, *res.x),
            new_t, linear_function(new_t, *res2.x),
            new_t, linear_function(new_t, *res3.x))

    plt.show()


def compare_loss_functions():
    fig, ax = plt.subplots()

    # Generate training data
    # Two blocks of data, shifted in value and "depth"
    a = 3000.
    b = 1500.
    c = -0.03
    t_min_1 = 1000.  # TVD in m
    t_max_1 = 2000.
    n_points = 300
    t_train_1 = np.linspace(t_min_1, t_max_1, n_points)
    y_train_1 = gen_test_data([a, b, c], t_train_1,  target_function=exp_function,
                            noise=4.3, n_outliers=30)

    a = 3500.
    t_min_2 = 3000.  # TVD in m
    t_max_2 = 4000.
    t_train_2 = np.linspace(t_min_2, t_max_2, n_points)
    y_train_2 = gen_test_data([a, b, c], t_train_2,  target_function=exp_function,
                              noise=4.3, n_outliers=30)

    t_train = np.concatenate([t_train_1, t_train_2])
    y_train = np.concatenate([y_train_1, y_train_2])

    ax.plot(t_train, y_train, 'o')

    # Start fitting a curve to the training data
    x0 = [1000., 1000., -0.1]
    t_test = np.linspace(t_min_1, t_max_2, n_points * 10)
    ax.plot(t_test, exp_function(t_test, a, b, c), 'k', linewidth=2, label='true')

    res_lsq = least_squares(residuals, x0,
                            args=(t_train, y_train),
                            kwargs= {'target_function': exp_function}, verbose=0)
    print('Default linear least squares: Success? {}'.format(res_lsq['success']))
    print(' {}'.format(res_lsq['message']))
    print(' x0: ', x0)
    print(' x: ', res_lsq['x'])
    ax.plot(t_test, exp_function(t_test, *res_lsq.x), label='linear loss')

    res_soft_l1 = least_squares(residuals, x0, loss='soft_l1', f_scale=0.1,
                                args=(t_train, y_train),
                                kwargs={'target_function': exp_function}, verbose=0)
    print('Soft L1 norm: Success? {}'.format(res_soft_l1['success']))
    print(' {}'.format(res_soft_l1['message']))
    print(' x0: ', x0)
    print(' x: ', res_soft_l1['x'])
    ax.plot(t_test, exp_function(t_test, *res_soft_l1.x), label='soft l1 loss')

    res_log = least_squares(residuals, x0, loss='cauchy', f_scale=0.1,
                            args=(t_train, y_train),
                            kwargs={'target_function': exp_function}, verbose=0)
    print('Cauchy norm: Success? {}'.format(res_log['success']))
    print(' {}'.format(res_log['message']))
    print(' x0: ', x0)
    print(' x: ', res_log['x'])
    ax.plot(t_test, exp_function(t_test, *res_log.x), 'r--', label='cauchy loss')

    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # play_with_data_fit()
    compare_loss_functions()
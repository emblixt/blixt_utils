import numpy as np
import os
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.interpolate import RegularGridInterpolator
import bruges
import datetime
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import colorConverter
from matplotlib.patches import Rectangle

from blixt_utils.utils import arrange_logging

logger = logging.getLogger(__name__)
text_style = {'fontsize': 'x-small', 'bbox': {'facecolor': 'w', 'alpha': 0.5}}
box_style = {'fill': False, 'lw': 0.5, 'ls': '--'}

"""
Code for generating synthetic data taken from 
https://github.com/Jun-Tam/3D-Seismic-Image-Fault-Segmentation

Heavily modified
 2021-03-26: Uses the more modern random generator: default_rng 
 2021-03-26: Number of horizons should vary according to num_horizons_rng
 ...
 ...
 2024-04-17: Added relative geological time calculation
"""


class DefineParams:
    """ Parameters for Creating Synthetic Traces """

    def __init__(self, patch_size, data_path, seed=None):
        """

        :param patch_size:
            int
            Number of cells in x, y and z
        :param data_path:
        :param seed:
            A seed to initialize the BitGenerator.
            https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
        """
        self.seed = seed

        # Turn on / off alterations:
        self.use_add_noise = True
        self.use_linear_deform = True
        self.use_gaussian_deform = True
        self.use_fault_deform = True

        # Calculate relative geological time labels
        self.calc_rgt = True

        # Synthetic 1D reflection model
        # The size of the model (size_tr) should be larger than the patch_size to cover up for
        # tilting layers and large fault offsets.
        # The cropping later is used to keep only the core part defined by patch_size
        self.buffer = 36
        size_tr = int(patch_size + 2 * self.buffer)
        self.num_horizons_rng = (5, 60)  # number of horizons in 1D model
        nx, ny, nz = ([patch_size] * 3)
        nxy = nx * ny
        nxyz = nxy * nz
        nx_tr, ny_tr, nz_tr = ([size_tr] * 3)
        nxy_tr = nx_tr * ny_tr
        nxyz_tr = nxy_tr * nz_tr
        x = np.linspace(0, nx_tr - 1, nx_tr)
        y = np.linspace(0, nx_tr - 1, ny_tr)
        z = np.linspace(0, nz_tr - 1, nz_tr)
        xy = np.reshape(np.array([np.meshgrid(x, y, indexing='ij')]), [2, nxy_tr]).T
        xyz = np.reshape(np.array([np.meshgrid(x, y, z, indexing='ij')]), [3, nxyz_tr]).T

        ' Feature Size '
        self.nx = nx  # Height of input feature
        self.ny = ny  # Width of input feature
        self.nz = nz  # Number of classes
        self.nxy = nxy  #
        self.nxyz = nxyz  #
        self.data_path = data_path

        ' Synthetic traces '
        self.dt = 0.004  # Synthetic Traces: Sampling interval (s)
        self.x = x  #
        self.y = y  #
        self.z = z  #
        self.xy = xy  #
        self.xyz = xyz  #
        self.x0 = int(nx / 2)  # Synthetic Traces: x center
        self.y0 = int(ny / 2)  # Synthetic Traces: y center
        self.z0 = int(nz / 2)  # Synthetic Traces: z center
        self.nx_tr = nx_tr  # Synthetic Traces: sampling in x
        self.ny_tr = ny_tr  # Synthetic Traces: sampling in y
        self.nz_tr = nz_tr  # Synthetic Traces: sampling in z
        self.nxy_tr = nxy_tr  #
        self.nxyz_tr = nxyz_tr  #
        self.x0_tr = int(nx_tr / 2)  # Synthetic Traces: x center
        self.y0_tr = int(ny_tr / 2)  # Synthetic Traces: y center
        self.z0_tr = int(nz_tr / 2)  # Synthetic Traces: z center

        # Noise
        self.snr_rng = (2, 5)  # Signal Noise Ratio
        # lowering both the lcut and hcut makes noise courser, useful for poor data at great depth
        self.lcut = 5  # Bandpass filter: Lower cutoff
        self.hcut = 80  # Bandpass filter: Upper cutoff

        # Wavelet
        self.t_lng = 0.082  # Ricker wavelet: Length ORIGINAL
        self.f0_rng = (20, 35)  # Wavelet freq range

        # Gaussian deformation
        self.num_gauss_rng = (1, 4)
        self.a_rng = (0, 1)  # Sinusoidal deformation: Amplitude
        self.b_rng = (0, 5)  # Sinusoidal deformation: Frequency
        self.c_rng = (0, nx_tr - 1)  # Sinusoidal deformation: Initial phase
        self.d_rng = (0, ny_tr - 1)  # Linear deformation: Slope
        self.sigma_rng = (10, 30)  #

        # Planar deformation
        self.e_rng = (0, 1)  # Linear deformation: Intercept
        self.f_rng = (0.05, 0.4)  # Gradient X direction  ORIGINALLY (0, 0.1)
        self.g_rng = (0.05, 0.4)  # Gradient X direction  ORIGINALLY (0, 0.1)

        # Faults
        #  No fault center are supposed to be closer than 32 pixels from cropped boundary
        self.x0_rng = (32, nx - 1 - 32)
        self.y0_rng = (32, ny - 1 - 32)
        self.z0_rng = (32, nz - 1 - 32)
        self.min_dist = 100  # Minimum distance between faults, Original was 20!
        self.num_flts_rng = (3, 6)  # number of faults
        # self.num_flts_rng = (5, 8)  # number of faults ORIGINAL
        # self.num_flts_rng = (1, 1)  # number of faults
        # self.throw_rng = (5, 30)  # Fault: Displacement ORIGINAL
        self.throw_rng = (5, 40)  # Fault: Displacement TESTING
        self.dip_rng = (62, 82)
        self.strike_rng = (0, 180)


class GenerateParams:
    def __init__(self, prm):

        self.rng = np.random.default_rng(prm.seed)

        # 1D reflection model
        self.num_horizons = int(self.rng.uniform(prm.num_horizons_rng[0], prm.num_horizons_rng[1]))

        # Deformation Parameters
        # 2D Gaussian Deformation
        prm.num_gauss = int(self.rng.uniform(prm.num_gauss_rng[0], prm.num_gauss_rng[1]))  #
        self.a = self.randomize_prm(prm.a_rng, pos_neg=True)  # All gaussian deformations have the same amplitude
        self.b = self.randomize_prm(prm.b_rng, prm.num_gauss, pos_neg=True)
        self.c = self.randomize_prm(prm.c_rng, prm.num_gauss)
        self.d = self.randomize_prm(prm.d_rng, prm.num_gauss)
        self.sigma = self.randomize_prm(prm.sigma_rng, prm.num_gauss)
        # Planar Deformation
        self.e = self.randomize_prm(prm.e_rng, pos_neg=True)
        self.f = self.randomize_prm(prm.f_rng, pos_neg=True)
        self.g = self.randomize_prm(prm.g_rng, pos_neg=True)

        # Fault Throw
        # self.num_faults = np.random.randint(prm.num_flts_rng[0], prm.num_flts_rng[1] + 1, 1)[0]
        self.num_faults = int(self.rng.uniform(prm.num_flts_rng[0], prm.num_flts_rng[1]))

        # Signal Noise Ratio
        self.snr = self.randomize_prm(prm.snr_rng)

        # wavelet center frequency
        self.f0 = self.randomize_prm(prm.f0_rng)

    def randomize_prm(self, lmts, n_rand=1, pos_neg=False):
        if pos_neg:
            # coeff = np.random.choice([-1, 1])
            coeff = self.rng.choice([-1, 1], n_rand)
        else:
            coeff = 1
        # prm = coeff * np.random.uniform(lmts[0], lmts[1], n_rand)
        prm = coeff * self.rng.uniform(lmts[0], lmts[1], n_rand)
        return prm

    def param_fault(self, prm):
        x0_f, y0_f, z0_f = self.pick_fault_center(prm, self.num_faults, prm.min_dist)
        throw = self.randomize_prm(prm.throw_rng, self.num_faults, pos_neg=True)
        dip = self.randomize_prm(prm.dip_rng, self.num_faults)
        strike = self.randomize_prm(prm.strike_rng, self.num_faults)
        # type_flt determines how the throw is distributed:
        # 0: constant throw, 1: linear throw, 2: Gaussian throw
        # Faults with constant throw typically cuts through the whole volume, and is given a lower probability
        types = self.rng.integers(11, size=self.num_faults)  # random integers between 0 and 10
        types[(0 < types) & (types < 6)] = 1 # 5 of eleven values are set to 1
        types[types > 5] = 2 # 5 of eleven values are set to 2
        type_flt = types
        # type_flt = self.rng.integers(2, size=self.num_faults)
        return x0_f, y0_f, z0_f, throw, dip, strike, type_flt

    def pick_fault_center(self, prm, num_faults, min_dist):
        def dist_multi_pts(coords, n_rand, min_dist):
            dist_cal = lambda x1, y1, z1, x2, y2, z2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            comb = combinations(np.arange(n_rand), 2)
            dist = []
            for i in list(comb):
                coords1 = coords[i[0]]
                coords2 = coords[i[1]]
                dist.append(dist_cal(coords1[0], coords1[1], coords1[2],
                                     coords2[0], coords2[1], coords2[2]))
            dist = np.array(dist)
            flag_dist = sum(np.array(dist) > min_dist) > 0
            return flag_dist, dist

        flag_dist = False
        x0_f = 0
        y0_f = 0
        z0_f = 0
        while not flag_dist:
            x0_f = self.randomize_prm(prm.x0_rng, num_faults)
            y0_f = self.randomize_prm(prm.y0_rng, num_faults)
            z0_f = self.randomize_prm(prm.z0_rng, num_faults)
            if num_faults == 1:
                flag_dist = True
            else:
                coords = [tuple([x0_f[i], y0_f[i], z0_f[i]]) for i in range(num_faults)]
                flag_dist, _ = dist_multi_pts(coords, num_faults, min_dist)
        return x0_f, y0_f, z0_f


class CreateSynthRefl(GenerateParams):
    def __init__(self, prm,
                 ax_one_d=None,
                 ax_deform_x=None, ax_deform_y=None, ax_deform_z=None,
                 ax_fault_x=None, ax_fault_y=None, ax_fault_z=None,
                 ax_fault_3d=None,
                 ax_rgt_1d=None,
                 ax_deform_rgt_x=None, ax_deform_rgt_y=None, ax_deform_rgt_z=None,
                 ax_fault_rgt_x=None, ax_fault_rgt_y=None, ax_fault_rgt_z=None
                 ):
        super().__init__(prm)

        self.one_d_model = np.zeros([prm.nz_tr])
        self.labels = np.zeros(prm.nxyz_tr)
        refl = self.create_1d_model(prm)
        self.rgt_1d = None
        self.rgt = None
        if prm.calc_rgt:
            self.rgt_1d = create_1d_rgt(prm)
            self.rgt = np.tile(self.rgt_1d, [prm.nxy_tr, 1])
            qc_plot_1d_rgt(prm, ax_rgt_1d)

        self.one_d_model = refl
        self.refl = np.tile(refl, [prm.nxy_tr, 1])
        self.qc_plot_1d_model(prm, ax_one_d)

        self.refl = self.deformation(prm)
        self.qc_plot_deform(prm, self.refl, ax_deform_x, ax_deform_y, ax_deform_z)
        if prm.calc_rgt:
            self.qc_plot_deform(prm, self.rgt, ax_deform_rgt_x, ax_deform_rgt_y, ax_deform_rgt_z)

        if prm.use_fault_deform:
            flag_zero_counts = False
        else:
            flag_zero_counts = True
        while not flag_zero_counts:
            self.x0_f, self.y0_f, self.z0_f, self.throw, self.dip, self.strike, self.type_flt = self.param_fault(prm)
            self.throw_shift(prm, ax_fault_3d=ax_fault_3d)
            flag_zero_counts = self.zero_counts(prm)
        self.qc_plot_fault(prm, self.refl, ax_fault_x, ax_fault_y, ax_fault_z, plot_data=False, plot_faults=True)
        if prm.calc_rgt:
            self.qc_plot_fault(prm, self.rgt, ax_fault_rgt_x, ax_fault_rgt_y, ax_fault_rgt_z,
                               plot_data=True, plot_faults=False)

    def zero_counts(self, prm):
        xyz = np.reshape(self.refl, [prm.nx_tr, prm.ny_tr, prm.nz_tr])
        xyz_crop = xyz[prm.x0_tr - prm.x0:prm.x0_tr + prm.x0,
                   prm.y0_tr - prm.y0:prm.y0_tr + prm.y0,
                   prm.z0_tr - prm.z0:prm.z0_tr + prm.z0].flatten()
        if sum(xyz_crop == 0) == 0:
            flag_zero_counts = True
        else:
            flag_zero_counts = True
        return flag_zero_counts

    def create_1d_model(self, prm):
        ''' Create 1D synthetic reflectivity model '''
        idx_refl = self.rng.integers(0, prm.nz_tr, self.num_horizons)
        refl = np.zeros(prm.nz_tr)
        refl[idx_refl] = 2 * self.rng.random(self.num_horizons) - 1
        return refl

    def qc_plot_1d_model(self, prm, ax_one_d):
        info_txt = None
        if ax_one_d is not None:
            ax_one_d.plot(self.one_d_model, np.arange(len(self.one_d_model)) * prm.dt * 1000.)
            ax_one_d.set_ylim(ax_one_d.get_ylim()[::-1])
            ax_one_d.set_xlim(-1, 1)
            ax_one_d.set_ylabel('TWT [ms]')
            ax_one_d.set_xlabel('Refl. coef')
            info_txt = 'No. refl: {}\nSeed: {}'.format(self.num_horizons, prm.seed)
            ax_one_d.text(-0.9, 0.9, info_txt, ha='left', va='top', **text_style)

    def deformation(self, prm):
        """
            Apply 2D Gaussian and Planar deformation.
            Computation is parallelized on GPU using cupy.
        """
        import cupy as cp
        xy_cp = cp.asarray(prm.xy)
        a_cp = cp.asarray(self.a)
        b_cp = cp.asarray(self.b)
        c_cp = cp.asarray(self.c)
        d_cp = cp.asarray(self.d)
        sigma_cp = cp.asarray(self.sigma)
        e_cp = cp.asarray(self.e)
        f_cp = cp.asarray(self.f)
        g_cp = cp.asarray(self.g)
        z_cp = cp.asarray(prm.z)

        func_planar = cp.ElementwiseKernel(
            in_params='T x, T y, T e, T f, T g',
            out_params='T z',
            operation= \
                '''
                z = e + f*x + g*y;
                ''',
            name='func_planar'
        )

        func_gauss2d = cp.ElementwiseKernel(
            in_params='T x, T y, T b, T c, T d, T sigma',
            out_params='T z',
            operation= \
                '''
                z = b*expf(-(powf(x-c,2) + powf(y-d,2))/(2*powf(sigma,2)));
                ''',
            name='func_gauss2d'
        )

        gauss_2d_cp = cp.zeros_like(xy_cp[:, 0])
        if prm.use_gaussian_deform:
            for i in range(len(self.b)):
                gauss_2d_cp += func_gauss2d(xy_cp[:, 0], xy_cp[:, 1], b_cp[i], c_cp[i], d_cp[i], sigma_cp[i])
        s1_cp = a_cp + (1.5 / z_cp) * cp.outer(cp.transpose(gauss_2d_cp), z_cp)
        if prm.use_linear_deform:
            s2_cp = func_planar(xy_cp[:, 0], xy_cp[:, 1], e_cp, f_cp, g_cp)
        else:
            s2_cp = cp.zeros_like(xy_cp[:, 0])

        refl_cp = cp.asarray(self.refl)
        rgt_cp = None
        if prm.calc_rgt:
            rgt_cp = cp.asarray(self.rgt)

        if prm.use_gaussian_deform or prm.use_linear_deform:
            for i in range(prm.nxy_tr):
                s = s1_cp[i, :] + s2_cp[i] + z_cp
                mat = cp.tile(z_cp, (len(s), 1)) - cp.tile(cp.expand_dims(s, 1), (1, len(z_cp)))
                refl_cp[i, :] = cp.dot(refl_cp[i, :], cp.sinc(mat))
                if prm.calc_rgt:
                    rgt_cp[i, :] = cp.dot(rgt_cp[i, :], cp.sinc(mat))

        if prm.calc_rgt:
            self.rgt = np.reshape(cp.asnumpy(rgt_cp), [prm.nxy_tr, prm.nz_tr])

        return np.reshape(cp.asnumpy(refl_cp), [prm.nxy_tr, prm.nz_tr])

    def qc_plot_deform(self, prm, data, ax_deform_x, ax_deform_y, ax_deform_z):
        """

        :param prm:
        :param data:
            self.refl or self.rgt
        :param ax_deform_x:
        :param ax_deform_y:
        :param ax_deform_z:
        :return:
        """
        tmp = None
        info_txt = None
        if ax_deform_x is not None:
            half_way = int(0.5 * prm.ny_tr)
            # tmp = np.reshape(self.refl, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_deform_x.imshow(np.transpose(tmp[:, half_way, :]))
            ax_deform_x.set_xlabel('X direction at Y={}'.format(half_way))
            # yticks = ax_deform_x.get_yticks() * prm.dt * 1000
            # ax_deform_x.set_yticklabels(yticks.astype(int))
            if prm.use_gaussian_deform:
                info_txt = 'Gaussian deform\n no. gauss: {:.0f}\n ampl: {:.2f}\n'.format(
                    prm.num_gauss, self.a[0])
                info_txt += (" freq: [" + ', '.join(['%.2f'] * len(self.b)) + "]") % tuple(self.b)
            else:
                info_txt = 'No Gaussian deform'
            if prm.use_linear_deform:
                info_txt += '\nPlanar deform\n Intercept: {:.2f}\n X grad: {:.2f}\n Y grad: {:.2f}'.format(
                    self.e[0], self.f[0], self.g[0])
            else:
                info_txt += '\nNo planar deform'
            # ax_deform_x.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_deform_x.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_deform_y is not None:
            half_way = int(0.5 * prm.nx_tr)
            if tmp is None:
                # tmp = np.reshape(self.refl, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
                tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_deform_y.imshow(np.transpose(tmp[half_way, :, :]))
            # yticks = ax_deform_y.get_yticks() * prm.dt * 1000
            # ax_deform_y.set_yticklabels(yticks.astype(int))
            ax_deform_y.set_xlabel('Y direction at X={}'.format(half_way))
            # if info_txt is not None:
            #    ax_deform_y.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_deform_y.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_deform_z is not None:
            half_way = int(0.5 * prm.nz_tr)
            if tmp is None:
                # tmp = np.reshape(self.refl, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
                tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_deform_z.imshow(tmp[:, :, half_way])
            ax_deform_z.set_xlabel('X direction at TWT={:.0f}ms'.format(half_way * prm.dt * 1000.))
            if info_txt is not None:
                ax_deform_z.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_deform_z.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

    def throw_shift(self, prm, ax_fault_3d=None):
        """ Add fault throw with linear and gaussian offset """

        def fault_plane(_x, _y, _x1, _y1, _z1, _theta, _phi):
            return _z1 + (np.cos(_phi) * (_x - _x1) + np.sin(_phi) * (_y - _y1)) * np.tan(_theta)

        def z_proj(_x, _y, _z, x0_f, y0_f, z0_f, _theta, _phi):
            _x1 = x0_f + (prm.nx_tr - prm.nx) / 2
            _y1 = y0_f + (prm.ny_tr - prm.ny) / 2
            _z1 = z0_f + (prm.nz_tr - prm.nz) / 2

            _z_flt_plane = fault_plane(_x, _y, _x1, _y1, _z1, _theta, _phi)
            if ax_fault_3d is not None:
                # Create 2D meshgrids for x and y
                x2d = np.reshape(_x, (prm.nx_tr, prm.ny_tr, prm.nz_tr))[:, :, 0]
                y2d = np.reshape(_y, (prm.nx_tr, prm.ny_tr, prm.nz_tr))[:, :, 0]
                _ = ax_fault_3d.plot_surface(x2d, y2d, fault_plane(x2d, y2d, _x1, _y1, _z1, _theta, _phi))
            return _z_flt_plane

        def fault_throw(theta, phi, throw, z0_f, type_flt, prm):
            """ Define z shifts"""
            z1 = (prm.nz_tr - prm.nz) / 2 + z0_f
            z2 = (prm.nz_tr - prm.nz) / 2 + prm.nz  # removing the last prm.nz made a huge difference
            z3 = (prm.nz_tr - prm.nz) / 2
            if type_flt == 0:  # constant offset
                    z_shift = throw * np.ones(prm.nz_tr)
            elif type_flt == 1:  # Linear offset
                if throw > 0:  # Normal fault
                    z_shift = throw * np.cos(theta) * (prm.z - z1) / (z2 - z1)
                    z_shift[z_shift < 0] = 0
                else:  # Reverse fault
                    z_shift = throw * np.cos(theta) * (prm.z - z1) / (z3 - z1)
                    z_shift[z_shift > 0] = 0
            else:  # Gaussian offset
                gaussian1d = lambda z, sigma: throw * np.sin(theta) * np.exp(-(z - z1) ** 2 / (2 * sigma ** 2))
                z_shift = gaussian1d(prm.z, sigma=20)

            """ flag offset """
            flag_offset = np.zeros([prm.nxy_tr, prm.nz_tr], dtype=bool)
            for i in range(prm.nxy_tr):
                flag_offset[i, :] = np.abs(z_shift) > 0.5  # Original limit was 1
            flag_offset = np.reshape(flag_offset, prm.nxyz_tr)
            return z_shift, flag_offset

        def replace(xyz0, idx_repl, x1, y1, z1, prm):
            """ Replace """
            xyz1 = np.reshape(xyz0.copy(), [prm.nx_tr, prm.ny_tr, prm.nz_tr])
            func_3d_interp = RegularGridInterpolator((prm.x, prm.y, prm.z), xyz1, method='linear',
                                                     bounds_error=False, fill_value=0)
            idx_interp = np.reshape(idx_repl, prm.nxyz_tr)
            xyz1 = np.reshape(xyz1, prm.nxyz_tr)
            xyz1[idx_interp] = func_3d_interp((x1[idx_interp], y1[idx_interp], z1[idx_interp]))
            return xyz1

        flag_zero_counts = False
        while not flag_zero_counts:
            # self.x0_f, self.y0_f, self.z0_f, self.throw, self.dip, self.strike, self.type_flt = self.param_fault(prm)
            for i in range(len(self.throw)):
                # z values on a fault plane
                theta = self.dip[i] / 180 * np.pi
                phi = self.strike[i] / 180 * np.pi
                x, y, z = prm.xyz[:, 0], prm.xyz[:, 1], prm.xyz[:, 2]
                z_flt_plane = z_proj(x, y, z, self.x0_f[i], self.y0_f[i], self.z0_f[i], theta, phi)
                idx_repl = prm.xyz[:, 2] <= z_flt_plane
                z_shift, flag_offset = \
                    fault_throw(theta, phi, self.throw[i], self.z0_f[i], self.type_flt[i], prm)
                x1 = prm.xyz[:, 0] - np.tile(z_shift, prm.nxy_tr) * np.cos(theta) * np.cos(phi)
                y1 = prm.xyz[:, 1] - np.tile(z_shift, prm.nxy_tr) * np.cos(theta) * np.sin(phi)
                z1 = prm.xyz[:, 2] - np.tile(z_shift, prm.nxy_tr) * np.sin(theta)

                # Fault throw
                refl = self.refl.copy()
                refl = replace(refl, idx_repl, x1, y1, z1, prm)
                self.refl = np.reshape(refl, [prm.nxy_tr, prm.nz_tr])
                if prm.calc_rgt:
                    rgt = self.rgt.copy()
                    rgt = replace(rgt, idx_repl, x1, y1, z1, prm)
                    self.rgt = np.reshape(rgt, [prm.nxy_tr, prm.nz_tr])

                # Fault Label
                labels = self.labels.copy()
                flt_flag = (0.5 * np.tan(self.dip[i] / 180 * np.pi) > abs(z - z_flt_plane)) & flag_offset
                labels[flt_flag] = 1
                self.labels = labels
            flag_zero_counts = self.zero_counts(prm)

    def qc_plot_fault(self, prm, data, ax_fault_x, ax_fault_y, ax_fault_z, plot_data=True, plot_faults=True):
        tmp = None
        lbls = None
        info_txt = None
        # create transparent colormap for labels
        color1 = colorConverter.to_rgba('black')
        color2 = colorConverter.to_rgba('white')
        cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2', [color2, color1], 256)
        cmap2._init()
        # create your alpha array and fill the colormap with them.
        # here it is progressive, but you can create whathever you want
        alphas = np.linspace(0, 1.0, cmap2.N + 3)
        cmap2._lut[:, -1] = alphas

        if ax_fault_x is not None:
            half_way = int(0.5 * prm.ny_tr)
            tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            lbls = np.reshape(self.labels, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            if plot_data:
                ax_fault_x.imshow(np.transpose(tmp[:, half_way, :]), origin='lower')
            if plot_faults:
                ax_fault_x.imshow(np.transpose(lbls[:, half_way, :]), cmap=cmap2, origin='lower')
            ax_fault_x.set_xlabel('X direction at Y={}'.format(half_way))
            ax_fault_x.set_ylim(ax_fault_x.get_ylim()[::-1])
            if prm.use_fault_deform:
                info_txt = 'Faults\n no. faults: {}\n min. dist: {}\n'.format(self.num_faults, prm.min_dist)
                info_txt += (" throw: [" + ', '.join(['%.0f'] * len(self.throw)) + "]\n") % tuple(self.throw)
                info_txt += (" dip: [" + ', '.join(['%.0f'] * len(self.dip)) + "]\n") % tuple(self.dip)
                info_txt += (" strike: [" + ', '.join(['%.0f'] * len(self.strike)) + "]\n") % tuple(self.strike)
                info_txt += (" type: [" + ', '.join(['%i'] * len(self.type_flt)) + "]") % tuple(self.type_flt)
            else:
                info_txt = 'No Faults'
            # ax_fault_x.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_fault_x.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_fault_y is not None:
            half_way = int(0.5 * prm.nx_tr)
            if tmp is None:
                tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            if lbls is None:
                lbls = np.reshape(self.labels, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            if plot_data:
                ax_fault_y.imshow(np.transpose(tmp[half_way, :, :]), origin='lower')
            if plot_faults:
                ax_fault_y.imshow(np.transpose(lbls[half_way, :, :]), cmap=cmap2, origin='lower')
            ax_fault_y.set_xlabel('Y direction at X={}'.format(half_way))
            ax_fault_y.set_ylim(ax_fault_y.get_ylim()[::-1])
            # if info_txt is not None:
            #    ax_fault_y.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_fault_y.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_fault_z is not None:
            half_way = int(0.5 * prm.nz_tr)
            if tmp is None:
                tmp = np.reshape(data, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            if lbls is None:
                lbls = np.reshape(self.labels, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            if plot_data:
                ax_fault_z.imshow(tmp[:, :, half_way])
            if plot_faults:
                ax_fault_z.imshow(lbls[:, :, half_way], cmap=cmap2)
            ax_fault_z.set_xlabel('X direction at TWT={:.0f}ms'.format(half_way * prm.dt * 1000.))
            if info_txt is not None:
                ax_fault_z.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_fault_z.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))


class CreateSyntheticTrace(CreateSynthRefl):
    def __init__(self, prm,
                 ax_one_d=None,
                 ax_deform_x=None, ax_deform_y=None, ax_deform_z=None,
                 ax_fault_x=None, ax_fault_y=None, ax_fault_z=None,
                 ax_one_d_conv=None, ax_one_d_noise=None,
                 ax_seismic_x=None, ax_seismic_y=None, ax_seismic_z=None,
                 ax_fault_3d=None,
                 ax_rgt_1d=None,
                 ax_deform_rgt_x=None, ax_deform_rgt_y=None, ax_deform_rgt_z=None,
                 ax_fault_rgt_x = None, ax_fault_rgt_y = None, ax_fault_rgt_z = None

    ):

        super().__init__(prm,
                         ax_one_d=ax_one_d,
                         ax_deform_x=ax_deform_x, ax_deform_y=ax_deform_y, ax_deform_z=ax_deform_z,
                         ax_fault_x=ax_fault_x, ax_fault_y=ax_fault_y, ax_fault_z=ax_fault_z,
                         ax_fault_3d=ax_fault_3d,
                         ax_rgt_1d=ax_rgt_1d,
                         ax_deform_rgt_x=ax_deform_rgt_x, ax_deform_rgt_y=ax_deform_rgt_y,
                         ax_deform_rgt_z=ax_deform_rgt_z,
                         ax_fault_rgt_x=ax_fault_rgt_x, ax_fault_rgt_y=ax_fault_rgt_y,
                         ax_fault_rgt_z=ax_fault_rgt_z
                         )
        self.wavelet = None
        self.traces = np.zeros([prm.nxy_tr, prm.nz_tr])
        self.convolve_wavelet(prm)
        self.qc_plot_1d_conv(prm, ax_one_d_conv)
        self.add_noise(prm)
        self.qc_plot_1d_noise(prm, ax_one_d_noise)
        self.standardizer()
        self.qc_plot_seismic(prm, ax_seismic_x, ax_seismic_y, ax_seismic_z)
        self.crop_center_patch(prm)
        self.traces = np.reshape(self.traces, [prm.nx, prm.ny, prm.nz])
        self.labels = np.reshape(self.labels, [prm.nx, prm.ny, prm.nz])
        self.rgt = np.reshape(self.rgt, [prm.nx, prm.ny, prm.nz])

    def convolve_wavelet(self, prm):
        ''' Convolve reflectivity model with a Ricker wavelet '''
        this_wavelet = bruges.filters.wavelets.ricker(prm.t_lng, prm.dt, self.f0)
        self.wavelet = this_wavelet.amplitude
        for i in range(prm.nxy_tr):
            self.traces[i, :] = np.convolve(self.refl[i, :], this_wavelet.amplitude, mode='same')

    def qc_plot_1d_conv(self, prm, ax_one_d_conv):
        info_txt = None
        if ax_one_d_conv is not None:
            one_d_conv = np.convolve(self.one_d_model, self.wavelet, mode='same')
            ax_one_d_conv.plot(one_d_conv, np.arange(len(self.one_d_model)) * prm.dt * 1000.)
            ax_one_d_conv.set_ylim(ax_one_d_conv.get_ylim()[::-1])
            ax_one_d_conv.set_ylabel('TWT [ms]')
            ax_one_d_conv.set_xlabel('Seismic amplitude')
            info_txt = 'Wavelet: Ricker, {:.1f}Hz\n'.format(self.f0[0])
            info_txt += ' duration: {:.1f}ms, dt: {:.1f}ms'.format(prm.t_lng * 1000., prm.dt * 1000.)
            ax_one_d_conv.text(one_d_conv.min(), 1, info_txt, ha='left', va='top', **text_style)

    def add_noise(self, prm):
        ''' Add some noise to traces to imitate real seismic data '''
        if prm.use_add_noise:
            b, a = this_butter_filter(prm)
            for i in range(prm.nxy_tr):
                noise = bruges.noise.noise_db(self.traces[i, :], self.snr)
                self.traces[i, :] = filtfilt(b, a, self.traces[i, :] + noise)

    def qc_plot_1d_noise(self, prm, ax_one_d_noise):
        info_txt = None
        if ax_one_d_noise is not None:
            b, a = this_butter_filter(prm)
            one_d_noisy = None
            one_d_conv = np.convolve(self.one_d_model, self.wavelet, mode='same')
            if prm.use_add_noise:
                noise = bruges.noise.noise_db(one_d_conv, self.snr)
                one_d_noisy = filtfilt(b, a, one_d_conv + noise)
                ax_one_d_noise.plot(one_d_noisy, np.arange(len(self.one_d_model)) * prm.dt * 1000.)
            else:
                ax_one_d_noise.plot(one_d_conv, np.arange(len(self.one_d_model)) * prm.dt * 1000.)
            ax_one_d_noise.set_ylim(ax_one_d_noise.get_ylim()[::-1])
            ax_one_d_noise.set_ylabel('TWT [ms]')
            ax_one_d_noise.set_xlabel('Seismic amplitude')
            if prm.use_add_noise:
                info_txt = 'Noise: SNR {:.1f}\n'.format(self.snr[0])
                info_txt += 'low: {:.1f}Hz, high: {:.1f}Hz'.format(prm.lcut, prm.hcut)
                ax_one_d_noise.text(one_d_noisy.min(), 1, info_txt, ha='left', va='top', **text_style)
            else:
                info_txt = 'Noise: None'
                ax_one_d_noise.text(one_d_conv.min(), 1, info_txt, ha='left', va='top', **text_style)

    def qc_plot_seismic(self, prm, ax_seismic_x, ax_seismic_y, ax_seismic_z):
        tmp = None
        info_txt = None
        if ax_seismic_x is not None:
            half_way = int(0.5 * prm.ny_tr)
            tmp = np.reshape(self.traces, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_seismic_x.imshow(np.transpose(tmp[:, half_way, :]), cmap='seismic')
            ax_seismic_x.set_xlabel('X direction at Y={}'.format(half_way))
            # yticks = ax_seismic_x.get_yticks() * prm.dt * 1000
            # ax_seismic_x.set_yticklabels(yticks.astype(int))
            info_txt = 'Simulated seismic'
            # ax_seismic_x.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_seismic_x.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_seismic_y is not None:
            half_way = int(0.5 * prm.nx_tr)
            if tmp is None:
                tmp = np.reshape(self.traces, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_seismic_y.imshow(np.transpose(tmp[half_way, :, :]), cmap='seismic')
            # yticks = ax_seismic_y.get_yticks() * prm.dt * 1000
            # ax_seismic_y.set_yticklabels(yticks.astype(int))
            ax_seismic_y.set_xlabel('Y direction at X={}'.format(half_way))
            # if info_txt is not None:
            #    ax_seismic_y.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_seismic_y.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

        if ax_seismic_z is not None:
            half_way = int(0.5 * prm.nz_tr)
            if tmp is None:
                tmp = np.reshape(self.traces, (prm.nx_tr, prm.ny_tr, prm.nz_tr))
            ax_seismic_z.imshow(tmp[:, :, half_way], cmap='seismic')
            ax_seismic_z.set_xlabel('X direction at TWT={:.0f}ms'.format(half_way * prm.dt * 1000.))
            if info_txt is not None:
                ax_seismic_z.text(10, 10, info_txt, ha='left', va='top', **text_style)
            ax_seismic_z.add_patch(
                Rectangle((prm.buffer, prm.buffer), prm.nx, prm.ny, **box_style))

    def crop_center_patch(self, prm):
        ''' Extract the central part in the input size '''

        def func_crop(xyz):
            xyz = np.reshape(xyz, [prm.nx_tr, prm.ny_tr, prm.nz_tr])
            xyz_crop = xyz[prm.x0_tr - prm.x0:prm.x0_tr + prm.x0,
                       prm.y0_tr - prm.y0:prm.y0_tr + prm.y0,
                       prm.z0_tr - prm.z0:prm.z0_tr + prm.z0]
            return np.reshape(xyz_crop, [prm.nxy, prm.nz])

        self.traces = func_crop(self.traces)
        self.labels = np.reshape(self.labels, [prm.nxy_tr, prm.nz_tr])
        self.labels = func_crop(self.labels)
        if prm.calc_rgt:
            self.rgt = np.reshape(self.rgt, [prm.nxy_tr, prm.nz_tr])
            self.rgt = func_crop(self.rgt)

    def standardizer(self):
        ''' Standardize amplitudes within the image '''
        std_func = lambda x: (x - np.mean(x)) / np.std(x)
        tr_std = std_func(self.traces)
        # Thew following two lines caused the output to look pixelized
        # tr_std[tr_std > 1] = 1
        # tr_std[tr_std < -1] = -1
        self.traces = tr_std


def this_butter_filter(prm):
    order = 5
    nyq = 1 / prm.dt / 2
    low = prm.lcut / nyq
    high = prm.hcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def create_1d_rgt(prm):
    """
    Calculate the relative geological time, which should vary from 1 to 150 in the z direction within the
    patch size

    :param prm:
    :return:
    """
    _z = np.arange(prm.nz_tr)
    _a = 149 / (prm.nz_tr - 2 * prm.buffer)
    _y0 = 0.5 * (151 - _a * prm.nz_tr)
    return _y0 + _a * _z


def qc_plot_1d_rgt(prm, ax):
    info_txt = None
    if ax is not None:
        rgt = create_1d_rgt(prm)
        ax.plot(rgt, np.arange(prm.nz_tr))
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel('Relative geological time')
        ax.axhline(prm.buffer, ls='--', lw=0.5)
        ax.axhline(prm.buffer + prm.nz, ls='--', lw=0.5)
        ax.axvline(1.0, ls='--', lw=0.5)
        ax.axvline(150, ls='--', lw=0.5)


def datagen(root_dir, _seed, mode, patch_size=128):
    """
    Generates an object with an array of synthetic traces and labels that indicates the positions of faults
        return_object.traces
        return_object.labels
    :param root_dir:
        str
        folder name to where qc plots and log are stored
    :param _seed:
        int
        A seed to initialize the BitGenerator.
        https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
    :param mode:
        str
        if True, data are generated in validation mode, else in training mode
        This does not affect the data, but is only meant to used in the logging
    :param patch_size:
        int
        Result arrays will be patch_size x patch_size x patch_size large
    :return:
        object with synthetic traces and labels as np.ndarrays
        return_object.traces
        return_object.labels

    """
    # Testing plotting fault planes in 3D
    ax3d = None

    fig, axes = plt.subplots(4, 3, figsize=(12, 18))
    fig.suptitle('{}, seed: {}'.format(mode, _seed))

    prm = DefineParams(patch_size, root_dir, _seed)

    synt_traces = CreateSyntheticTrace(prm,
                                       ax_one_d=axes[0][0], ax_one_d_conv=axes[0][1], ax_one_d_noise=axes[0][2],
                                       ax_deform_x=axes[1][1], ax_deform_y=axes[1][2], ax_deform_z=axes[1][0],
                                       ax_fault_x=axes[2][1], ax_fault_y=axes[2][2], ax_fault_z=axes[2][0],
                                       ax_seismic_x=axes[3][1], ax_seismic_y=axes[3][2], ax_seismic_z=axes[3][0],
                                       ax_fault_3d=ax3d
                                       )

    # Save qc plot
    this_time = datetime.datetime.now().strftime('%H.%M')
    fig.savefig(os.path.join(root_dir, '{} {} {}.png'.format(mode, this_time, _seed)))

    logger.info('  Seed: {}'.format(_seed))
    logger.info('  Mode: {}'.format(mode))
    logger.info('  Patch_size: {}'.format(patch_size))
    logger.info('  Number_horizons: {}'.format(synt_traces.num_horizons))
    logger.info('  Wavelet_frequency: {:.2f}'.format(synt_traces.f0[0]))  # Hz
    logger.info('  Wavelet_duration: {:.2f}'.format(prm.t_lng * 1000.))  # ms
    logger.info('  Wavelet_dt: {:.2f}'.format(prm.dt * 1000.))
    if prm.use_add_noise:
        logger.info('  SNR: {:.2f}'.format(synt_traces.snr[0]))
        logger.info('  Noise_band_low: {:.2f}'.format(prm.lcut))  # Hz
        logger.info('  Noise_band_high: {:.2f}'.format(prm.hcut))  # Hz
    else:
        logger.info('  SNR: No noise')
    if prm.use_gaussian_deform:
        logger.info('  Number_Gaussians: {:.0f}'.format(prm.num_gauss))
        _str = '  Gaussian_amplitudes: [' + ', '.join(['{:2f}'] * len(synt_traces.a)) + ']'
        logger.info(_str.format(*synt_traces.a))
        _str = '  Gaussian_frequencies: [' + ', '.join(['{:2f}'] * len(synt_traces.b)) + ']'
        logger.info(_str.format(*synt_traces.b))
    else:
        logger.info('  Number_Gaussians: 0')

    if prm.use_linear_deform:
        logger.info('  Deform_intercept: {:.2f}'.format(synt_traces.e[0]))
        logger.info('  Deform_X_grad: {:.2f}'.format(synt_traces.f[0]))
        logger.info('  Deform_Y_grad: {:.2f}'.format(synt_traces.g[0]))
    else:
        logger.info('  No linear deformation')

    if prm.use_fault_deform:
        logger.info('  Number_faults: {}'.format(synt_traces.num_faults))
        logger.info('  Fault_min_distance: {}'.format(prm.min_dist))
        _str = '  Fault_throws: [' + ', '.join(['{:.2f}'] * len(synt_traces.throw)) + ']'
        logger.info(_str.format(*synt_traces.throw))
        _str = '  Fault_dips: [' + ', '.join(['{:.2f}'] * len(synt_traces.dip)) + ']'
        logger.info(_str.format(*synt_traces.dip))
        _str = '  Fault_strikes: [' + ', '.join(['{:.2f}'] * len(synt_traces.strike)) + ']'
        logger.info(_str.format(*synt_traces.strike))
        _str = '  Fault_types: [' + ', '.join(['{:}'] * len(synt_traces.type_flt)) + ']'
        logger.info(_str.format(*synt_traces.type_flt))
    else:
        logger.info('  Number_faults: 0')
    logger.info(' ')


def test(path, seed=None, patch_size=128):
    import os
    import json
    """
    test creating a synthetic dataset with labels
    The params are stored in the log, and QC images are stored under the path folder
    :param path:
        str
        folder name where QC images are stored
    :param seed:
        A seed to initialize the BitGenerator.
        https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng
    :param patch_size:
        int
    :return:
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import projections
    # fig3d, ax3d = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax3d = None

    fig, axes = plt.subplots(4, 3, figsize=(12, 18))
    this_date = datetime.datetime.now().strftime('%Y-%m-%dT%H.%M.%S.%f')
    fig.suptitle('Run {}'.format(this_date))

    fig_rgt, axes_rgt = plt.subplots(4, 3, figsize=(12, 18))
    fig_rgt.suptitle('Run {}'.format(this_date))

    prm = DefineParams(patch_size, path, seed)

    synt_traces = CreateSyntheticTrace(prm,
                                       ax_one_d=axes[0][0], ax_one_d_conv=axes[0][1], ax_one_d_noise=axes[0][2],
                                       ax_deform_x=axes[1][1], ax_deform_y=axes[1][2], ax_deform_z=axes[1][0],
                                       ax_fault_x=axes[2][1], ax_fault_y=axes[2][2], ax_fault_z=axes[2][0],
                                       ax_seismic_x=axes[3][1], ax_seismic_y=axes[3][2], ax_seismic_z=axes[3][0],
                                       ax_fault_3d=ax3d,
                                       ax_rgt_1d=axes_rgt[0][0],
                                       ax_deform_rgt_x=axes_rgt[1][1], ax_deform_rgt_y=axes_rgt[1][2],
                                       ax_deform_rgt_z=axes_rgt[1][0],
                                       ax_fault_rgt_x=axes_rgt[2][1], ax_fault_rgt_y=axes_rgt[2][2],
                                       ax_fault_rgt_z=axes_rgt[2][0],
                                       )

    # Save qc plot
    # fig.savefig(os.path.join(path, '{}.png'.format(this_date)))

    ## Save data to json
    # with open(os.path.join(path, '{}_traces.json'.format(this_date)), 'w') as f:
    #    json.dump(synt_traces.traces.tolist(), f)
    # with open(os.path.join(path, '{}_labels.json'.format(this_date)), 'w') as f:
    #    json.dump(synt_traces.labels.tolist(), f)

    logger.info('Generating training data {}, random seed: {}'.format(this_date, prm.seed))
    logger.info(' 1D reflection model:')
    logger.info('  Number of horizons : ')
    logger.info('  {} :'.format(
        synt_traces.num_horizons
    ))
    logger.info(' Sampling interval : Patch size : Bandpass low filter cutoff : Bandpass high filter cutoff'
                ' : Wavelet : Wavelet length : ')
    logger.info(' {} : {} : {} : {} : {} : {} '.format(
        prm.dt, prm.nx, prm.lcut, prm.hcut, 'Ricker', prm.t_lng
    ))

    return synt_traces


if __name__ == '__main__':
    base_folder = "C:\\tmp"
    arrange_logging(True, os.path.join(base_folder, "test.log"))
    seed = None
    synt_traces = test(base_folder, seed)
    # np.save(os.path.join(base_folder, 'seismic.dat'), synt_traces.traces)
    # np.save(os.path.join(base_folder, 'labels.dat'), synt_traces.labels)
    # np.save(os.path.join(base_folder, 'rgt.dat'), synt_traces.rgt)
    synt_traces.traces.astype(np.float32).tofile(os.path.join(base_folder, 'seismic.dat'))
    synt_traces.labels.astype(np.float32).tofile(os.path.join(base_folder, 'labels.dat'))
    synt_traces.rgt.astype(np.float32).tofile(os.path.join(base_folder, 'rgt.dat'))
    plt.show()

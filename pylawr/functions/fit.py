#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from copy import deepcopy

# External modules
import xarray as xr
import numpy as np
import scipy.spatial

# Internal modules
from pylawr.transform.temporal.extrapolation import Extrapolator
from pylawr.utilities.decorators import log_decorator
from pylawr.grid.cartesian import CartesianGrid
from pylawr.functions.grid import get_masked_grid, prepare_grid
from pylawr.field import get_verified_grid
from pylawr.remap import NearestNeighbor, OrdinaryKriging
from pylawr.transform.inference import SIRParticleFilter, laplace_pdf, \
    random_walk, KernelVariogram

logger = logging.getLogger(__name__)


@log_decorator(logger)
def fit_extrapolator(refl_array, pre_refl_path,
                     grid_extrapolation=CartesianGrid(start=-30000,
                                                      nr_points=600),
                     remapper=NearestNeighbor(1), *args, **kwargs):
    """
    Fits an extrapolator with given reflectivity and path to old reflectivity.

    Parameters
    ----------
    refl_array : :py:class:`~xarray.DataArray`
        This reflectivity array is handled as last available reflectivity.
    pre_refl_path : str, file or xarray.backends.*DataStore
        This path is passed to :py:func:`xarray.open_dataarray`. Stored array
        should be a processed array by ``pylawr``.
    grid_extrapolation : :py:class:`~pylawr.grid.cartesian.CartesianGrid`
        This grid is passed to
        :py:meth:`~pylawr.transform.temporal.extrapolation.Extrapolator.fit`.
        If no grid is given, the grid from ``refl_array`` is used. Default is
        None.
    args :
        list of additional arguments, which are passed to
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator` during
        initialization.
    kwargs
        dictionary of additional keyword arguments, which are passed to
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator` during
        initialization.

    Returns
    -------
    extrapolator :
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator`
        This extrapolator was fitted to given array and previous array with
        given grid.
    """
    pre_refl_array = xr.open_dataarray(pre_refl_path)

    grid_data = get_verified_grid(refl_array)

    if not isinstance(grid_data, CartesianGrid):
        remapper.fit(grid_data, grid_extrapolation)
        refl_array = remapper.remap(refl_array)
        pre_refl_array = remapper.remap(pre_refl_array)
    else:
        grid_extrapolation = grid_data

    extrapolator = Extrapolator(*args, **kwargs)
    extrapolator.fit(array=refl_array, array_pre=pre_refl_array,
                     grid=grid_extrapolation)

    return extrapolator


@log_decorator(logger)
def extrapolate_offline(refl_array, pre_refl_array, time_extra,
                        grid_extrapolation=CartesianGrid(),
                        remapper=NearestNeighbor(1), *args, **kwargs):
    """
    Fits an extrapolator for temporal interpolation between two given arrays.
    The extrapolator is applied to both fields and then the weighted average is
    returned as extrapolated field. The weights are based on a linear dynamics
    assumptions and anti-proportional from array time to ``time_extra``.

    Parameters
    ----------
    refl_array : :py:class:`~xarray.DataArray`
        This reflectivity array is handled as last available reflectivity.
    pre_refl_path : str, file or xarray.backends.*DataStore
        This path is passed to :py:func:`xarray.open_dataarray`. Stored array
        should be a processed array by ``pylawr``.
    grid_extrapolation : :py:class:`~pylawr.grid.cartesian.CartesianGrid`
        This grid is passed to
        :py:meth:`~pylawr.transform.temporal.extrapolation.Extrapolator.fit`.
        If no grid is given, the grid from ``refl_array`` is used. Default is
        None.
    args :
        list of additional arguments, which are passed to
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator` during
        initialization.
    kwargs
        dictionary of additional keyword arguments, which are passed to
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator` during
        initialization.

    Returns
    -------
    extrapolated_field :
        :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator`
        This extrapolator was fitted to given array and previous array with
        given grid.
    """
    grid_data = get_verified_grid(refl_array)

    if not isinstance(grid_data, CartesianGrid):
        remapper.fit(grid_data, grid_extrapolation)
        refl_array = remapper.remap(refl_array)
        pre_refl_array = remapper.remap(pre_refl_array)
    else:
        grid_extrapolation = grid_data

    extrapolator = Extrapolator(*args, **kwargs)
    extrapolator.fit(array=refl_array, array_pre=pre_refl_array,
                     grid=grid_extrapolation)

    extrapolated_fwd = extrapolator.transform(pre_refl_array, time=time_extra)
    extrapolated_bwd = extrapolator.transform(refl_array, time=time_extra)

    td_fwd = np.abs(time_extra - pre_refl_array.indexes['time']).values
    td_bwd = np.abs(refl_array.indexes['time'] - time_extra).values
    weight_fwd = float(td_bwd / (td_bwd+td_fwd))
    weight_bwd = float(td_fwd / (td_bwd+td_fwd))
    if weight_bwd > 0.95:
        print('Use backward')
        weighted_average = refl_array.copy(deep=True)
    elif weight_fwd > 0.95:
        print('Use forward')
        weighted_average = pre_refl_array.copy(deep=True)
    else:
        print('Use weighted average, weights: {0:.2f}, {1:.2f}'.format(
            weight_fwd, weight_bwd
        ))
        weighted_average = weight_fwd * extrapolated_fwd + \
                           weight_bwd * extrapolated_bwd
    weighted_average['time'] = [time_extra, ]
    refl_extrapolated = weighted_average.lawr.set_grid_coordinates(
        grid_extrapolation
    )
    return refl_extrapolated


@log_decorator(logger)
def sample_sq_diff(refl_array, samples=10000, rnd=None):
    """
    Function to sample squared differences between different grid point and
    their reflectivities. This function can be used to construct an empirical
    variogram. Only reflectivites larger than 5 dBZ are sampled. These squared
    differences are based on a stochastic assumption for improved performance.

    Parameters
    ----------
    refl_array : :py:class:`xarray.DataArray`
        This reflectivity array is used to sample the grid points and their
        values. This reflectivity array needs a grid.
    samples : int, optional
        This number of samples is drawn from given array. Default is 10000.
    rnd : :py:class:`numpy.random.RandomState`, int or None, optional
        This random state is used to generate the samples. If no random state
        is given a new one is created with either given seed (int) or the cpu
        time (None). Default is None.

    Returns
    -------
    grid_dist : :py:class:`numpy.ndarray`
        This is the grid distance in meters between the compared points. The
        length of this array equals the specified number of samples.
    squared_diff : :py:class:`numpy.ndarray`
        This is the squared difference between two random sampled points. The
        length of this array equals the specified number of samples.
    """
    if not isinstance(rnd, np.random.RandomState):
        rnd = np.random.RandomState(rnd)
    rain_mask = (refl_array > 5).values
    rain_grid = get_masked_grid(
        refl_array.lawr.grid, rain_mask.squeeze()
    )
    rain_grid_array = prepare_grid(rain_grid)
    idx_samples = rnd.choice(rain_grid_array.shape[0], size=(samples, 2))
    sampled_grid = rain_grid_array.values[idx_samples, :]
    sampled_values = refl_array.values[rain_mask][idx_samples]
    grid_dist = scipy.spatial.minkowski_distance(
        sampled_grid[:, 0], sampled_grid[:, 1]
    )
    squared_diff = np.power(sampled_values[:, 0]-sampled_values[:, 1], 2)
    return grid_dist, squared_diff


@log_decorator(logger)
def sample_variogram(refl_array, samples=10000, bins=50, rnd=None):
    """
    This function samples a variogram from given reflectivity field. A variogram
    specifies the spatial correlation within a reflectivity. This variogram can
    be used to optimize the parameters for interpolation or remapping. This
    variogram sampling is based on a stochstic approximation, where a given
    number of grid points are sampled and used to create the variogram. This
    increases the performance of the variogram creation and could improve the
    optimization like stochastic gradient descent. This function uses
    :py:func:`~pylawr.functions.fit.sample_sq_diff` to sample the squared
    differences between grid points. Afterwards they are binned with
    :py:func:`~scipy.stats.binned_statistics` and a median value.

    Parameters
    ----------
    refl_array : :py:class:`xarray.DataArray`
        This reflectivity array is used to sample the grid points and their
        values. This reflectivity array needs a grid.
    samples : int, optional
        This number of samples is drawn from given array. Default is 10000.
    bins : int or sequence of scalars, optional
        This specifies the number of bins within the variogram. For further
        information please see: :py:func:`~scipy.stats.binned_statistics`.
        Default is 50.
    rnd : :py:class:`numpy.random.RandomState`, int or None, optional
        This random state is used to generate the samples. If no random state
        is given a new one is created with either given seed (int) or the cpu
        time (None). Default is None.

    Returns
    -------
    bin_center : :py:class:`numpy.ndarray`
        These are the center points of the bins. The length of this array equals
        the number of specified bins.
    bin_vario : :py:class:`numpy.ndarray`
        These are the binned and sampled variogram values. The length of this
        array equals the number of specified bins.
    """
    grid_dist, squared_diff = sample_sq_diff(refl_array, samples=samples,
                                             rnd=rnd)
    bin_vario, bin_edges, _ = scipy.stats.binned_statistic(
        grid_dist, squared_diff, statistic='median', bins=bins
    )
    bin_center = np.convolve(bin_edges, np.ones((2,))/2, mode='valid')
    bin_vario /= 2
    return bin_center, bin_vario


@log_decorator(logger)
def fit_kriging(refl_array, samples=10000, iterations=1, particle_filter=None,
                kriging=None, bins=50, rnd=None, ens_size=100,
                ens_threshold=10):
    """
    This function fits kriging instances with particle filters and a stochastic
    variogram matching.

    Parameters
    ----------
    refl_array : :py:class:`xarray.DataArray`
        The kriging instance is fitted to this reflectivity array. This
        reflectivity array needs a set grid to determine the distances between
        different grid points.
    samples : int, optional
        This number of samples are drawn from given reflectivity data during
        each iteration for the stochastic variogram matching (default = 10000).
    iterations : int, optional
        This number of iterations is used to fit the kriging instance to given
        reflectivity data. The default number of one iteration is recommended in
        an online setting, while more than 100 iterations should be used if
        kriging is fitted from scratch.
    particle_filter : :py:class:`~pylawr.transform.inference.SIRParticleFilter`
    or None, optional
        This initialized particle filter is used to fit kriging to given
        reflectivity. If no particle filter (default) is given, a new sequential
        importance resampling particle filter is constructed and used.
    kriging : child of :py:class:`~pylawr.remap.SimpleKriging` or None, optional
        This kriging instance is fitted and returned by this function. If no
        kriging instance is given (default), then a new ordinary kriging
        instance with a RBF and white noise kernel is fitted.
    bins : int, optional
        This number of bins (default=50) is used to construct the stochastic
        variogram. A large number of bins, increases the resolution of the
        variogram, but also increases the observational uncertainty.
    rnd : :py:class:`~numpy.random.RandomState`, int or None, optional
        This random state is used to sample the stochastic variogram. If no
        random state is given, a new one is initialized with given random seed
        (int) or with processor clock time (None, default).
    ens_size : int, optional
        This number of ensemble members (default=100) is used for the particle
        filter to fit the kriging instance.
    ens_threshold : int, optional
        If the number of effective ensemble members in particle filter is below
        this threshold (default=10), then the ensemble is resampled.

    Returns
    -------
    kriging : child of :py:class:`~pylawr.remap.SimpleKriging`
        The fitted kriging instance with the weighted and averaged parameters
        estimated with returned particle filter. This kriging instance is ready
        to remap data.
    particle_filter : :py:class:`~pylawr.transform.inference.SIRParticleFilter`
        This particle filter instance was used to infer the parameters for
        returned kriging instance.

    Notes
    -----
    In its default settings, this function iterates only once over given radar
    data. To fit kriging instances from scratch, it is recommended to set the
    number of iterations at least to 100.
    """
    if kriging is None:
        kriging = OrdinaryKriging()
    else:
        kriging = deepcopy(kriging)

    if particle_filter is None:
        obs_operator = KernelVariogram(kriging.kernel)
        parameters = np.array([p.value for p in kriging.kernel.params])
        particle_filter = SIRParticleFilter(
            obs_op_func=obs_operator,
            predict_func=random_walk,
            prob_func=laplace_pdf,
            params_fg=np.repeat(parameters[None, ...], repeats=ens_size,
                                axis=0),
            ens_size=ens_size,
            ens_threshold=ens_threshold

        )

    rain_mask = (refl_array > 5).values
    rain_sum = np.sum(rain_mask)
    no_rain = rain_sum < 5000
    if no_rain:
        logger.warning('I cannot fit the kriging, because there is not enough '
                       'rain to fit variogram ({0:d})'.format(rain_sum))
        return kriging, particle_filter

    for _ in range(iterations):
        obs_dist, obs_vario = sample_variogram(
            refl_array, samples=samples, bins=bins, rnd=rnd
        )
        mean_params = particle_filter.fit(
            obs=obs_vario, time=refl_array.time.values[0], obs_dist=obs_dist,
            noise=0.05
        )
    for i, param in enumerate(kriging.kernel.params):
        param.value = mean_params[i]
    return kriging, particle_filter

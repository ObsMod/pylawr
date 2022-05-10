#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from copy import deepcopy

# External modules
from wradlib.clutter import filter_gabella
from wradlib.atten import correct_attenuation_constrained, \
    constraint_dbz, constraint_pia
import numpy as np
import xarray as xr

# Internal modules
from pylawr.field import array_has_tag, tag_array, get_verified_grid
from pylawr.transform.filter.tdbz import TDBZFilter
from pylawr.transform.filter.spin import SPINFilter
from pylawr.transform.filter.spike import SPKFilter
from pylawr.transform.filter.ring import RINGFilter
from pylawr.transform.filter.speckle import SpeckleFilter
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.transform.temporal.noiseremover import NoiseRemover
from pylawr.transform.spatial.beamexpansion import BeamExpansion, \
    TAG_BEAM_EXPANSION_UNCORR
from pylawr.transform.attenuation.atten_corr_dual import \
    AttenuationCorrectionDual
from pylawr.grid.polar import PolarGrid
from pylawr.transform.spatial.interpolation import Interpolator
from pylawr.remap import OrdinaryKriging, NearestNeighbor
from pylawr.utilities.decorators import log_decorator


logger = logging.getLogger(__name__)


@log_decorator(logger)
def remove_noise(reflectivity, noise_remover=None):
    """
    Remove background noise from given reflectivity field.

    Parameters
    ----------
    reflectivity : :py:class:`xarray.DataArray`
        The noise level is determined from this reflectivity array and noise
        is removed from this array. If this array has no beam expansion tag, it
        is assumed that the beam expansion effect is corrected.
    noise_remover : :py:class:`pylawr.filter.noiseremover.NoiseRemover` or None
        This noise remover is fitted to given reflectivity and further used to
        remove noise from given reflectivity. If no noise_remover is given a new
        noise remover is created and fitted. Default is None.

    Returns
    -------
    refl_filtered : :py:class:`xarray.DataArray`
        Reflectivity with filtered noise. The noise level was determined for
        given reflectivity.
    noise_remover : :py:class:`pylawr.filter.noiseremover.NoiseRemover`
        The fitted noise remover. If no noise_remover was given as argument,
        this is a new noise remover.
    """
    if not isinstance(noise_remover, NoiseRemover):
        noise_remover = NoiseRemover()
    beam_expansion_filter = BeamExpansion()
    if not array_has_tag(reflectivity, TAG_BEAM_EXPANSION_UNCORR):
        refl_filtered = beam_expansion_filter.transform(
            reflectivity.lawr.to_z(), inverse=True
        )
    else:
        refl_filtered = reflectivity.lawr.to_z()
    noise_remover.fit(refl_filtered)
    refl_filtered = noise_remover.transform(refl_filtered)
    refl_filtered = beam_expansion_filter.transform(refl_filtered.lawr.to_z())
    return refl_filtered, noise_remover


@log_decorator(logger)
def remove_clutter_lawr(reflectivity):
    """
    Remove clutter from given reflectivity field of lawr. The clutter is removed
    with pre-initialized :py:class:`~pylawr.filter.tdbz.TDBZFilter`,
    :py:class:`~pylawr.filter.spin.SPINFilter`. The clutter detection is tuned
    for the X band radars.

    Parameters
    ----------
    reflectivity : :py:class:`xarray.DataArray`
        The cluttermaps are fitted to this reflectivity array and clutter is
        removed from this array.

    Returns
    -------
    refl_filtered : :py:class:`xarray.DataArray`
        Reflectivity with filtered clutter. All clutter values are set to
        :py:class:`np.nan`.
    cluttermap : OrderedDict(str,
            :py:class:`pylawr.filter.cluttermap.ClutterMap`)
        These filters were used in this order to filter the clutter out of the
        reflectivity.
    """
    cluttermap = ClutterMap('ClutterMapLawr', fuzzy_threshold=0.)

    tdbz_filter = TDBZFilter(window_size=17, threshold=30.)
    spin_filter = SPINFilter(threshold=3., window_size=15, window_criterion=.2)
    spk_filter_one = SPKFilter(threshold=3., window_size=11,
                               window_criterion=.5, spike_width=1)
    spk_filter_two = SPKFilter(threshold=3., window_size=11,
                               window_criterion=.5, spike_width=2)
    ring_filter_one = RINGFilter(threshold=3., window_size=11,
                                 window_criterion=.5, ring_width=1)
    ring_filter_two = RINGFilter(threshold=3., window_size=11,
                                 window_criterion=.5, ring_width=2)
    speckle_filter_33 = SpeckleFilter(threshold=2, window_size=(3, 3))
    speckle_filter_35 = SpeckleFilter(threshold=4, window_size=(3, 5))
    speckle_filter_55 = SpeckleFilter(threshold=10, window_size=(5, 5))
    speckle_filter_57 = SpeckleFilter(threshold=16, window_size=(5, 7))
    speckle_filter_77 = SpeckleFilter(threshold=26, window_size=(7, 7))

    # fit the filters on reflectivity
    tdbz_clt = tdbz_filter.create_cluttermap(reflectivity)
    spin_clt = spin_filter.create_cluttermap(reflectivity)
    spk_clt_one = spk_filter_one.create_cluttermap(reflectivity, addname='One')
    spk_clt_two = spk_filter_two.create_cluttermap(reflectivity, addname='Two')
    ring_clt_one = ring_filter_one.create_cluttermap(reflectivity,
                                                     addname='One')
    ring_clt_two = ring_filter_two.create_cluttermap(reflectivity,
                                                     addname='Two')

    cluttermap.append(tdbz_clt)
    cluttermap.append(spin_clt)
    cluttermap.append(spk_clt_one)
    cluttermap.append(spk_clt_two)
    cluttermap.append(ring_clt_one)
    cluttermap.append(ring_clt_two)

    # filter reflectivity for speckle filter
    refl_filtered = cluttermap.transform(reflectivity)

    # fit the speckle filter on filtered reflectivty
    cmap_speckle_33 = speckle_filter_33.create_cluttermap(refl_filtered,
                                                          addname='W33')
    cmap_speckle_35 = speckle_filter_35.create_cluttermap(refl_filtered,
                                                          addname='W35')
    cmap_speckle_55 = speckle_filter_55.create_cluttermap(refl_filtered,
                                                          addname='W55')
    cmap_speckle_57 = speckle_filter_57.create_cluttermap(refl_filtered,
                                                          addname='W57')
    cmap_speckle_77 = speckle_filter_77.create_cluttermap(refl_filtered,
                                                          addname='W77')

    cluttermap.append(cmap_speckle_33)
    cluttermap.append(cmap_speckle_35)
    cluttermap.append(cmap_speckle_55)
    cluttermap.append(cmap_speckle_57)
    cluttermap.append(cmap_speckle_77)

    # apply all fitted filters on initial reflectivity
    refl_filtered = cluttermap.transform(reflectivity)

    return refl_filtered, cluttermap


@log_decorator(logger)
def remove_clutter_dwd(reflectivity):
    """
    Remove clutter from given reflectivity field of DWD. The clutter is removed
    with pre-initialized :py:class:`~wradlib.clutter.filter_gabella`. The
    clutter detection is tuned for the C band radars.

    Parameters
    ----------
    reflectivity : :py:class:`xarray.DataArray`
        The cluttermaps are fitted to this reflectivity array and clutter is
        removed from this array.

    Returns
    -------
    refl_filtered : :py:class:`xarray.DataArray`
        Reflectivity with filtered clutter. All clutter values are set to
        :py:class:`np.nan`.
    cluttermap : :py:class:`pylawr.filter.cluttermap.ClutterMap`
        This fitted clutter map was used to filter given reflectivity array.
    """
    cluttermap = ClutterMap('ClutterMapDwd', fuzzy_threshold=0.)
    gabella = filter_gabella(reflectivity.values[0], wsize=5, thrsnorain=0.,
                             tr1=6., n_p=8, tr2=1.3, rm_nans=False,
                             radial=False, cartesian=False)[None, ...]
    gabella_clt = ClutterMap('GabellaFilter', gabella.astype(int))
    cluttermap.append(gabella_clt)
    refl_filtered = cluttermap.transform(reflectivity)

    return refl_filtered, gabella_clt


def _set_max_trunc_radius(remapper, trunc_radius):
    try:
        decorr_param = remapper.kernel.get_named_param('decorrelation')[0]
        remapper.max_dist = decorr_param.value * trunc_radius
        logger.info(
            'Set truncation radius to {0:.0f} m'.format(remapper.max_dist)
        )
    except (IndexError, TypeError, AttributeError):
        pass
    return remapper


@log_decorator(logger)
def interpolate_center_no_rain(refl_array, center_len=1500):
    interp_array = refl_array.copy(deep=True)
    center_pixels = interp_array.loc[
        dict(range=slice(None, center_len))
    ]
    center_sum_rain = np.sum((center_pixels > 5))
    center_rain = center_sum_rain > 0.5 * center_pixels.size
    if not center_rain:
        center_non_nan = ~np.isnan(center_pixels)
        interp_array.loc[
            dict(range=slice(None, center_len))
        ] = center_pixels.where(center_non_nan, -32.5)
    interp_array = interp_array.lawr.set_metadata(refl_array)
    return interp_array


@log_decorator(logger)
def interpolate_missing(reflectivity, remapper=None, trunc_radius=2,
                        zero_thres=0.34, zero_field=(11, 11)):
    """
    Interpolate missing values of given reflectivity array with given remapper.
    The truncation radius defines the maximum distance in decorrelation lengths
    of source points for interpolation. The maximum distance for remapper is set
    to this value.

    Parameters
    ----------
    reflectivity : :py:class:`xarray.DataArray`
        The remapper is fitted to this reflectivity array and NaN values are
        interpolated in this array.
    remapper : child of :py:class:`~pylawr.remap.base.BaseRemap` or None
        This remapper is used to fill missing values within given reflectivity.
        If no remapper is given, a new
        :py:class:`~pylawr.remap.nearest.NearestNeighbor` remapper is created.
        Default is None.
    trunc_radius : float or None, optional
        If no points within this truncation radius are found, no interpolation
        is possible. If this truncation radius is None, no truncation is done.
        If this radius is a float (default = 2), then the truncation will be
        performed for trunc_radius * decorrelation length of the remapper.
    zero_field : tuple(int)
        The size of the local receptive field to search for valid values, which
        have rain. If the number of rain values is below a given threshold, the
        missing values are set to zero. Default is (31, 31).
    zero_thres : float, optional
        If the number of rain values within a local receptive field is below
        this threshold, the missing values are set to zero. Default is 0.34.

    Returns
    -------
    refl_filtered : :py:class:`xarray.DataArray`
        Reflectivity with interpolated values. All values are either filled with
        no rain if a given criterion is reached or interpolated by given
        remapper.
    remapper : child of :py:class:`~pylawr.remap.base.BaseRemap`
        This remapper was used to fill holes within given reflectivity.

    Warnings
    --------
    Given remapper is fitted to given reflectivity array.

    To use a truncation length, it is necessary that a parameter named
    ``decorrelation`` is registered within given kriging instance. At the
    moment, truncation is only possible for kriging. If you want to use
    truncation for other algorithms, you have to manually set ``max_dist`` for
    this algorithm.
    """
    if remapper is None:
        remapper = OrdinaryKriging()
    else:
        remapper = deepcopy(remapper)
    remapper = _set_max_trunc_radius(remapper, trunc_radius)
    int_filter = Interpolator(threshold=0.95, algorithm=remapper,
                              zero_thres=zero_thres, zero_field=zero_field)
    reflectivity = reflectivity.lawr.to_dbz()
    refl_interpolated = int_filter.transform(reflectivity)
    return refl_interpolated, remapper


@log_decorator(logger)
def correct_attenuation_dwd(refl, constraint_pia_arg=20.):
    """
    Gate-by-Gate attenuation correction for a single weather radar based on
    modification of :cite:`kramer2008` according to the iterative estimation
    of k-Z relationship. The parameters are tuned for C-band by ourselves
    using MRR WMH measurements of the years 2013 to 2020 covering 10252 fit
    values. This method uses wradlib method
    :py:func:`wradlib.atten.correct_attenuation_constrained` to calculate
    additive correction.

    Parameters
    ----------
    refl : :py:class:`xarray.DataArray`
        Radar field of reflectivity in decibel (dBZ).
    constraint_pia_arg : float, optional
        Constraint of the PIA in dB. A higher constraint is more likely to
        result in an unstable attenuation correction.

    Returns
    -------
    refl_corrected : :py:class:`xarray.DataArray`
        Attenuation corrected radar field.
    pia : :py:class:`xarray.DataArray`
        Path integrated attenuation field.
    """
    pia = correct_attenuation_constrained(refl.values,
                                          a_max=5.9e-6,
                                          a_min=2.71e-6,
                                          n_a=200,
                                          b_max=0.97,
                                          b_min=0.89,
                                          n_b=10,
                                          gate_length=0.25,
                                          constraints=[constraint_dbz,
                                                       constraint_pia],
                                          constraint_args=[
                                              [59.0], [constraint_pia_arg]]
                                          )

    pia = xr.DataArray(
        data=pia,
        coords=refl.coords,
        dims=refl.dims,
    )
    pia = pia.lawr.set_variable('pia')

    if np.sometrue(np.isinf(pia)):
        logger.error('The PIA is highly unstable and the attenuation '
                     'correction is not applied!')
        return refl, pia
    else:
        if np.max(pia) > constraint_pia_arg:
            logger.warning('The PIA estimate is unstable and exceeds the '
                           'constraint_pia_arg.')

        refl_corrected = refl + pia
        refl_corrected = refl_corrected.lawr.set_metadata(refl)
        tag_array(refl_corrected, "attenuation-corr-single")

        return refl_corrected, pia


@log_decorator(logger)
def correct_attenuation_lawr(refl, constraint_pia_arg=10.):
    """
    Gate-by-Gate attenuation correction for a single weather radar based on
    modification of :cite:`kramer2008` according to the iterative estimation
    of k-Z relationship. The parameters are tuned for X-band by ourselves
    using MRR WMH measurements of the years 2013 to 2020 covering 108056 fit
    values. This method uses wradlib method
    :py:func:`wradlib.atten.correct_attenuation_constrained` to calculate
    additive correction.

    Parameters
    ----------
    refl : :py:class:`xarray.DataArray`
        Radar field of reflectivity in decibel (dBZ).
    constraint_pia_arg : float, optional
        Constraint of the PIA in dB. A higher constraint is more likely to
        result in an unstable attenuation correction.

    Returns
    -------
    refl_corrected : :py:class:`xarray.DataArray`
        Attenuation corrected radar field. If the PIA estimate is valid/finite.
    pia : :py:class:`xarray.DataArray`
        Path integrated attenuation field.
    """
    pia = correct_attenuation_constrained(refl.values,
                                          a_max=9.52e-5,
                                          a_min=4.02e-5,
                                          n_a=200,
                                          b_max=0.9,
                                          b_min=0.79,
                                          n_b=10,
                                          gate_length=0.06,
                                          constraints=[constraint_dbz,
                                                       constraint_pia],
                                          constraint_args=[
                                              [59.0], [constraint_pia_arg]]
                                          )

    pia = xr.DataArray(
        data=pia,
        coords=refl.coords,
        dims=refl.dims,
    )
    pia = pia.lawr.set_variable('pia')

    if np.sometrue(np.isinf(pia)):
        logger.error('The PIA is highly unstable and the attenuation '
                     'correction is not applied!')
        return refl, pia
    else:
        if np.max(pia) > constraint_pia_arg:
            logger.warning('The PIA estimate is unstable and exceeds the '
                           'constraint_pia_arg.')

        refl_corrected = refl + pia
        refl_corrected = refl_corrected.lawr.set_metadata(refl)
        tag_array(refl_corrected, "attenuation-corr-single")

        return refl_corrected, pia


@log_decorator(logger)
def correct_attenuation_dual(refl_attenuated, refl_robust):
    """
    This method corrects reflectivity measurements of weather radars that
    operate in attenuation-influenced frequency bands (X-band) using
    observations from less attenuated radar systems (C-band) based on
    :cite:`lengfeld2016`.

    Parameters
    ----------
    refl_attenuated : :py:class:`xarray.DataArray`
        Radar field of attenuation-influenced weather radar
    refl_robust : :py:class:`xarray.DataArray`
        Radar field of less attenuated weather radar

    Returns
    -------
    refl_corrected : :py:class:`xarray.DataArray`
        Attenuation corrected radar field.
    pia : :py:class:`xarray.DataArray`
        Path integrated attenuation field.
    """
    grid_attenuated = get_verified_grid(refl_attenuated)
    grid_robust = get_verified_grid(refl_robust)

    # define coarse grid for the calculation
    # calculation on pattern area with dwd range resolution
    nr_ranges = int(round(grid_attenuated.nr_ranges
                          * grid_attenuated.range_res
                          / grid_robust.range_res))

    grid = PolarGrid(grid_attenuated.center,
                     nr_ranges=nr_ranges,
                     range_res=grid_robust.range_res)

    # remap data on coarse grid
    remap = NearestNeighbor(1)
    remap.fit(grid_robust, grid)
    remapped_robust = remap.remap(refl_robust)
    remap.fit(grid_attenuated, grid)
    remapped_attenuated = remap.remap(refl_attenuated)

    # estimate the attenuation
    atten_corr = AttenuationCorrectionDual()
    atten_corr.fit(remapped_attenuated, remapped_robust)

    # remap attenuation data on fine grid
    remap.fit(grid, grid_attenuated)
    pia = remap.remap(atten_corr.attenuation)

    # correct the attenuation
    refl_corrected = refl_attenuated + pia
    refl_corrected = refl_corrected.lawr.set_metadata(refl_attenuated)
    tag_array(refl_corrected, "attenuation-corr-dual-isotonic")
    refl_corrected = refl_corrected.lawr.set_grid_coordinates(grid_attenuated)

    return refl_corrected, pia


@log_decorator(logger)
def correct_attenuation(refl_pattern, refl_dwd=None):
    """
    This method corrects reflectivity measurements of weather radars that
    operate in attenuation-influenced frequency bands (X-band) using
    observations from less attenuated radar systems (C-band). If the data of the
    less attenuated radar system is not available the gate-by-gate correction
    procedures based on :cite:`kramer2008` with
    :py:class:`~wradlib.atten.correct_attenuation_constrained` is applied.

    Parameters
    ----------
    refl_pattern : :py:class:`xarray.DataArray`
        Radar field of PATTERN measurement
    
    refl_dwd : :py:class:`xarray.DataArray` or None
        Radar field of DWD measurement

    Returns
    -------
    refl_corrected : :py:class:`xarray.DataArray`
        Attenuation corrected radar field
    pia : :py:class:`xarray.DataArray`
        Path integrated attenuation field.
    """
    if refl_dwd is None:
        refl_corrected, pia = correct_attenuation_lawr(refl_pattern)
    else:
        refl_corrected, pia = correct_attenuation_dual(refl_pattern, refl_dwd)
    return refl_corrected, pia


@log_decorator(logger)
def extrapolation_offline(now, prev_refl, next_refl, extrapolator):
    """
    Extrapolates two reflectivity fields to required time step for offline
    processing. We extrapolate from the previous reflectivity field and
    analogous from the following, next reflectivity field to required time step,
    thus the mean of the two extrapolated fields is the reflectivity field of
    the needed time step.

    Parameters
    ----------
    now : :py:class:`numpy.datetime64`
        Current time step within time interval of `prev_refl` and `next_refl`.
    prev_refl : :py:class:`~xarray.DataArray`
        Reflectivity array from previous time step.
    next_refl : :py:class:`~xarray.DataArray`
        Reflectivity array from next time step.
    extrapolator : :py:class:`~pylawr.transform.temporal.extrapolation.Extrapolator`
        Fitted extrapolator to given arrays.

    Returns
    -------
    refl : :py:class:`~xarray.DataArray`
        Reflectivity array from current time step.
    """
    prev_refl_now = extrapolator.transform(prev_refl,
                                           time=now)
    next_refl_now = extrapolator.transform(next_refl,
                                           time=now)

    refl = deepcopy(prev_refl_now)
    refl.values = np.nanmean(
        np.array([prev_refl_now.values, next_refl_now.values]),
        axis=0)
    refl = refl.lawr.set_metadata(prev_refl_now)

    return refl

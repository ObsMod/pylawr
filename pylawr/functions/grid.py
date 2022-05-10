#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import math

# External modules
import numpy as np

# Internal modules
import xarray as xr
from pylawr.grid import LatLonGrid
from pylawr.grid.unstructured import UnstructuredGrid


logger = logging.getLogger(__name__)


def get_latlon_grid(orig_grid):
    """
    This function can be used to get a rectangular
    :py:class:`~pylawr.grid.latlon.LatLonGrid`. The edges of the rectangular
    grid are determined by the boundary values of the original grid. The median
    step-wide of the original grid is used as resolution.

    Parameters
    ----------
    orig_grid : child of :py:class:`pylawr.grid.BaseGrid`
        This original grid is used to determine edges and step wide of the
        latitude and longitude grid.

    Returns
    -------
    latlon_grid : :py:class:`pylawr.grid.latlon.LatLonGrid`
        This grid is created based on given original grid with edges and step
        wide determined by the original grid.
    """
    lat, lon = orig_grid.lat_lon
    start_lat = np.min(lat)
    start_lon = np.min(lon)
    res_lat = np.median(np.abs(np.diff(lat, axis=-1)))
    res_lon = np.median(np.abs(np.diff(lon, axis=0)))
    points_lat = math.ceil((np.max(lat) - start_lat)/res_lat)
    points_lon = math.ceil((np.max(lon) - start_lon)/res_lon)
    latlon_grid = LatLonGrid(
        start=(start_lat, start_lon),
        resolution=(res_lat, res_lon),
        nr_points=(points_lat, points_lon),
        center=orig_grid.center
    )
    return latlon_grid


def remap_data(data, orig_grid, new_grid, remapper=None):
    """
    This function remaps data based on given grids with given remap instance.
    If the remap instance was not fitted yet, it will be fitted.

    Parameters
    ----------
    data : :py:class:`xarray.DataArray`
        This data is remapped by given remapper. The last coordinate(s) should
        be the grid coordinates.
    orig_grid : child of :py:class:`pylawr.grid.BaseGrid`
        This grid is used as basis grid for the remapping. The last data
        coordinates should have the same shape as this grid.
    new_grid : child of :py:class:`pylawr.grid.BaseGrid`
        The data is remapped to this grid.
    remapper : child of :py:class:`pylawr.remap.BaseRemap` or None, optional
        This remapper is used to remap given data to new grid. If the remapper
        is already fitted, it is assumed that the remapper was fitted to given
        grids. Default value (None) is an unfitted
        :py:class:`~pylawr.remap.NearestNeighbor` instance with a single nearest
        neighbor.

    Returns
    -------
    remapped_data : :py:class:`xarray.DataArray`
        This data was remapped to ``new_grid`` with given remapper.
    remapper : child of :py:class:`pylawr.remap.BaseRemap`
        This remapper was used to remap given data. This remapper is fitted to
        given grids.

    Warnings
    --------
    For a fitted ``remapper``, we assume that the ``remapper`` was fitted to
    given grids!

    Notes
    -----
    Most of processing time is spend on fitting given ``remapper``. It is
    recommended to give a fitted remapper to increase the speed of this
    function.
    """
    if remapper is None:
        from pylawr.remap import NearestNeighbor
        remapper = NearestNeighbor(1)
    if not remapper.fitted:
        remapper.fit(orig_grid, new_grid)
    remapped_data = remapper.remap(data)
    return remapped_data, remapper


def get_masked_grid(origin_grid, mask_array):
    """
    Mask value within a given origin grid.

    Parameters
    ----------
    mask_array : :py:class:`numpy.ndarray` (bool)
        This mask is applied on given ``origin_grid``. Should have the same
        shape as ``origin_grid``. All values set to True are used, while
        values set to False are dismissed.
    origin_grid : child of :py:class:`pylawr.grid.base.BaseGrid`
        This grid is used to infer the latitude, longitude and altitude
        values for the unstructured grid.

    Returns
    -------
    masked_grid : :py:class:`pylawr.grid.unstructured.UnstructuredGrid`
        The unstructured grid with all non-masked values. This unstructured
        grid gets its latitude, longitude and altitude values from given
        ``origin_grid``. All values set to True within ``mask_array`` are
        used.
    """
    lat_lon = np.array(origin_grid.lat_lon)
    masked_latlon = lat_lon[..., mask_array].T
    masked_altitude = origin_grid.altitude[mask_array][..., None]
    masked_vals = np.concatenate([masked_latlon, masked_altitude], axis=-1)
    masked_grid = UnstructuredGrid(masked_vals, origin_grid.center)
    return masked_grid


def get_cartesian(grid, use_altitude=False):
    """
    This method calculates cartesian coordinates based on the given grid.
    The cartesian coordinates are calculated based on simplifications and
    uses trigonometric functions. The latitude (lat), longitude (lon) and
    height values (h) are extracted from the given grid, while the
    earth_radius (R) is set for the remapping.

    ..math::

        x = R * cos(lat) * cos(lon)

        y = R * cos(lat) * sin(lon)

        z = h + R * sin(lat)

    Parameters
    ----------
    grid : child of :py:class:`pylawr.grid.BaseGrid`
        The cartesian coordinates are calculated based on this grid. The
        grid needs the methods: :py:meth:`get_altitude` and
        :py:meth:`get_lat_lon` and should be a child of
        :py:class:`~pylawr.grid.BaseGrid`.
    use_altitude : bool
        Use the altitude for cartesian coordinates. Default is `False`.
        Note, using the correct altitude for the cartesian coordinates for
        remapping may not preferred.

    Returns
    -------
    cartesian : :py:class:`xarray.DataArray`
        The calculated coordinates based on the given grid. The last
        dimension is `x`, `y` and `z`, while the other dimension sizes are
        defined by the shape of the input grid.
    """
    try:
        lat_lon = grid.get_lat_lon()
        altitude = grid.get_altitude()
        earth_radius = grid.earth_radius
    except AttributeError:
        raise TypeError(
            'The given grid is no valid grid and cannot be used!'
        )
    lat_rad = np.deg2rad(lat_lon['lat'])
    lon_rad = np.deg2rad(lat_lon['lon'])

    transposed_dims = [d for d in lat_rad.dims] + ['coord_names', ]

    x = earth_radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = earth_radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = earth_radius * np.sin(lat_rad)

    if use_altitude:
        z += altitude

    cartesian = xr.concat([x, y, z], dim='coord_names')
    cartesian['coord_names'] = ['x', 'y', 'z']
    cartesian = cartesian.transpose(*transposed_dims)
    return cartesian


def prepare_grid(grid):
    """
    This method is used to prepare the grid for the remapping, the grid
    is transformed into cartesian coordinates with
    :py:meth:``_get_cartesian`` and then stacked and transposed.

    Parameters
    ----------
    grid : child of :py:class:`pylawr.grid.BaseGrid`
        The cartesian coordinates are calculated based on this grid. The
        grid needs the methods: :py:meth:`get_altitude` and
        :py:meth:`get_lat_lon` and should be a child of
        :py:class:`~pylawr.grid.BaseGrid`.

    Returns
    -------
    cartesian : :py:class:`xarray.DataArray`
        The calculated coordinates based on the given grid. The last
        dimension is `x`, `y` and `z`, while the first dimension is a
        stacked dimension based on the original grid dimensions.
    """
    cartesian = get_cartesian(grid)
    if len(grid.coord_names) > 1:
        cartesian = cartesian.stack(stacked=grid.coord_names)
        cartesian = cartesian.transpose()
    return cartesian

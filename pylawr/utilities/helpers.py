#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from datetime import datetime

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.field import tag_array


logger = logging.getLogger(__name__)


def create_array(grid,
                 const_val=0,
                 timestamp=datetime.utcnow(),
                 tag='testing',
                 var='dbz'):
    """
    Create a testing array based on given grid. The array is initialized with
    given constant value.

    Parameters
    ----------
    grid : child of :py:class:`pylawr.grid.BaseGrid`
        The array will be created based on these grid coordinates.
    const_val : float, optional
        The grid will have this constant value.
    timestamp : datetime.datetime, optional
        Value of time coordinate.
    tag : str, optional
        This tag is added to the array.
    var : str, optional
        Variable of array following naming conventions.


    Returns
    -------
    array : :py:class:`xarray.DataArray`
        This array has the shape of the grid coordinates and an additional time
        axis as first dimension. The time is initialized as 01/01/1970. The grid
        is appended to this array with ``set_grid_coordinates``.
    """
    coords = grid.get_coordinates()
    coords['time'] = [timestamp]
    dims = ['time', *grid.coord_names]
    shape = [1, *grid.grid_shape]
    data = np.ones(shape, dtype=float) * const_val
    array = xr.DataArray(
        data,
        coords=coords,
        dims=dims
    )
    array = array.lawr.set_variable(var)
    array = array.lawr.set_grid_coordinates(grid)
    if tag:
        tag_array(array, tag)
    return array


def polar_padding(reflectivity, pad_size=(5, 5)):
    """
    Utility function for polar padding. In polar padding, the lower boundary is
    wrapped, while the upper boundary is reflected. The left and right
    boundaries are connected.

    .. code-block:: python

        array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])

    with a field size of (3,3) will be padded to:

    .. code-block:: python

        array([[2, 3, 4, 1, 2, 3],
               [4, 1, 2, 3, 4, 1],
               [8, 5, 6, 7, 8, 5],
               [12, 9, 10, 11, 12, 9],
               [12, 9, 10, 11, 12, 9]])

    Parameters
    ----------
    reflectivity : numpy.ndarray
        This reflectivity field will be padded. The last two dimension should be
        (azimuth, range).
    pad_size : tuple(int, int), optional
        The padding of (azimuth, range).

    Returns
    -------
    padded_refl : numpy.ndarray
        The padded reflectivity field.
    """
    lower_pad = np.roll(reflectivity, int(reflectivity.shape[-1]/2), axis=-1)
    lower_pad = lower_pad[..., pad_size[0]-1::-1, :]
    upper_pad = reflectivity[..., :-pad_size[0]-1:-1, :]
    padded_refl = np.concatenate([lower_pad, reflectivity, upper_pad], axis=-2)
    left_pad = padded_refl[..., -pad_size[1]:]
    right_pad = padded_refl[..., :pad_size[1]]
    padded_refl = np.concatenate([left_pad, padded_refl, right_pad], axis=-1)
    return padded_refl

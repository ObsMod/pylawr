#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.09.17
#
# Created for pattern
#
#
#    Copyright (C) {2017}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging

# External modules
import xarray as xr

import cartopy.crs as ccrs

import numpy as np


# Internal modules
from .base import BaseLayer
from pylawr.grid.base import BaseGrid


# Workaround for plotter, see: https://github.com/SciTools/cartopy/issues/1120
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

logger = logging.getLogger(__name__)


class RadarFieldLayer(BaseLayer):
    """
    Gridded data, like radar data, can be plotted with this layer. The given
    radar field is plotted on a subplot, which can be specified during
    plotting. A given grid is used to specify the boundaries for the radar
    field.

    Parameters
    ----------
    radar_field : :py:class:`xarray.DataArray` or :py:class:`numpy.ndarray`
        This array is plotted with this layer. It has to be castable to a
        one- or two-dimensional :py:class:`numpy.ndarray`. The array needs
        the same shape as the grid to be plottable. Missing values can be
        marked as :py:class:`numpy.nan` and are masked during  plotting. If
        no grid is given, the radar field need to have a grid specified in
        :py:attr:`radar_field.lawr.grid <pylawr.field.RadarField.grid>`.
    grid : child of :py:class:`pylawr.grid.base.BaseGrid` or None
        This grid is used to infer the boundaries for the radar field. For
        this the grid need to have
        :py:attr:`pylawr.grid.base.BaseGrid.lat_lon_bounds`. If no grid is
        given, the grid will be inferred from radar field. The grid need to
        have the same shape as the squeezed radar field. Default is None.
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot. Default is 0.
    **settings
        Variable keyword arguments dict, which is passed during plotting to
        :py:meth:`~matplotlib.axes.Axes.pcolormesh` and which sets the
        plotting behaviour of this layer. All keyword arguments except
        `transform` can be used.

    Attributes
    ----------
    zorder : int
        The z-order defines the drawing order of the layers. If no z-order
        is set, the z-order will be set during plotting based on layer order
        in subplot.
    settings : dict
        The settings for this layer. It is passed to
        :py:meth:`~matplotlib.axes.Axes.pcolormesh`. These settings can be
        accessed directly and via a dict like interface of this layer. All
        possible kwargs of :py:meth:`~matplotlib.axes.Axes.pcolormesh` can
        set to this dictionary.
    field : :py:class:`xarray.DataArray` or :py:class:`numpy.ndarray`
        This array is plotted with this layer. It is castable to a one- or
        two-dimensional :py:class:`numpy.ndarray` and has during plotting
        the same shape as the grid. Missing values are marked as
        :py:class:`numpy.nan` and are masked during plotting.
    grid : child of :py:class:`pylawr.grid.base.BaseGrid` or None
        This grid is used to infer the boundaries for the radar field with
        :py:attr:`pylawr.grid.base.BaseGrid.lat_lon_bounds`. The grid has
        the same shape as the radar_field. If the grid is None, then the
        grid will be inferred during plotting from
        :py:attr:`radar_field.lawr.grid <pylawr.field.RadarField.grid>`.
    plot_store : :py:class:`matplotlib.collections.QuadMesh` or None
        The plotted element of this layer. This is the storage for the
        returned element of :py:meth:`~matplotlib.axes.Axes.pcolormesh`.
        If this storage is None, this layer was not plotted yet.
    """
    def __init__(self, radar_field, grid=None, zorder=0, **settings):
        super().__init__(zorder, **settings)
        self._field = None
        self._grid = None
        self._transform = ccrs.PlateCarree()
        self.grid = grid
        self.field = radar_field
        self.plot_store = None

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, field):
        if not isinstance(field, xr.DataArray) and \
                not isinstance(field, np.ndarray):
            raise TypeError('The given radar field is not a valid '
                            '``xarray.DataArray`` or ``np.array``')
        if field.squeeze().ndim > 2:
            raise ValueError('The given radar field is not castable into one '
                             'or two dimensions and cannot be plotted.')
        self._field = field

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        if isinstance(grid, BaseGrid) or grid is None:
            self._grid = grid
        else:
            raise TypeError(
                'The given grid has to be a valid pylawr grid or None. The '
                'given grid type was: {0:s}'.format(str(type(grid)))
            )

    def _get_checked_grid(self):
        """
        Get the grid for plotting purpose. If grid is not set, the grid will be
        inferred from radar field. The grid is checked against the radar field
        if the shapes match.

        Returns
        -------
        grid : child of :py:class:`pylawr.grid.base.BaseGrid`
            The inferred and checked grid. This grid has the same shape as the
            radar field.
        """
        grid = self._grid
        if grid is None:
            try:
                grid = self.field.lawr.grid
            except (TypeError, AttributeError):
                raise AttributeError('No grid is set and the grid is not '
                                     'accessible, please set the grid instead')
        if hasattr(self.field, 'lawr'):
            self.field.lawr.check_grid(grid=grid)
        return grid

    def plot(self, ax):
        """
        This method plots the radar field onto given axes with set zorder. For
        plotting :py:meth:`~matplotlib.axes.Axes.pcolormesh` is called based on
        given radar field and inferred grid. All nan-values of the given radar
        field are masked.

        Parameters
        ----------
        ax : :py:class:`matplotlib.axes.Axes`
            This layer is plotted with set zorder on this subplot.
        """
        grid = self._get_checked_grid()
        lat_bounds, lon_bounds = grid.lat_lon_bounds
        try:
            numpy_data = self.field.values.squeeze()
        except AttributeError:
            numpy_data = self.field.squeeze()
        mask = ~np.isfinite(numpy_data)
        masked_data = np.ma.array(data=numpy_data, mask=mask)
        self.plot_store = ax.pcolormesh(
            lon_bounds, lat_bounds, masked_data, transform=self._transform,
            zorder=self.zorder, **self.settings
        )
        self._collection.append(self.plot_store)

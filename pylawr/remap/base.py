#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import abc
import warnings

# External modules
import xarray as xr

# Internal modules
import pylawr.functions.grid as grid_funcs
from pylawr.utilities.decorators import log_decorator


logger = logging.getLogger(__name__)


class NotImprovableError(Exception):
    pass


class NotFittedError(Exception):
    pass


class BaseRemap(object):
    """
    The BaseRemap is a base class for all remap subclasses.

    Parameters
    ----------
    """
    def __init__(self):
        self._grid_in = None
        self._grid_out = None

    def _prepare_grid(self, grid):
        warnings.warn(
            'The method `_prepare_grid` will be removed in the next release',
            category=DeprecationWarning
        )
        return grid_funcs.prepare_grid(grid)

    @property
    @abc.abstractmethod
    def fitted(self):
        """
        Check if the remapping is fitted.

        Returns
        -------
        fitted : bool
            If this remap object is fitted.
        """
        pass

    @abc.abstractmethod
    def fit(self, grid_in, grid_out):
        """
        Fit the remapping for given grids. If the remapping is refitted
        and only the output grid changed the old point neighbours are recovered
        when possible. If the input grid in a refitting the old tree is reused.

        Parameters
        ----------
        grid_in : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from this grid to another grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        grid_out : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from another grid to this grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        """
        pass

    def optimize(self, data, warm_start=False):
        """
        Optimize the parameter of the remapping based on given data.

        Parameters
        ----------
        data : :py:class:`xarray.DataArray`
            The data is used to optimize the parameters of this remapping.
        warm_start : bool, optional
            If the last optimized values should used as starting values for this
            optimization loop. Default is False.

        Returns
        -------
        self : Interpolation
            This optimized interpolation instance.
        """
        raise NotImprovableError('This interpolation cannot be optimized!')

    def _stack_grid_coords(self, data):
        """
        The data is stacked along matching grid coordinates. It is further
        checked whether the array can use this grid.

        Parameters
        ----------
        data : :py:class:`xarray.DataArray`
            This data is stacked.

        Returns
        -------
        stacked_data : :py:class:`xarray.DataArray`
            The stacked data based on the coordinates names of the input grid.
            The grid coordinates are stacked to a new grid coordinate.
        """
        data = data.lawr.set_grid_coordinates(self._grid_in)
        if len(self._grid_in.coord_names) > 1:
            stacked_data = data.stack(grid=self._grid_in.coord_names)
            dims_wo_grid = [
                d for d in data.dims
                if d not in self._grid_in.coord_names
            ]
            transpose_dims = dims_wo_grid + ['grid']
            stacked_data = stacked_data.transpose(*transpose_dims)
        else:
            rename_mapping = {self._grid_in.coord_names[0]: 'grid'}
            stacked_data = data.rename(rename_mapping)
        return stacked_data

    def _array_postprocess(self, remapped_data, original_data):
        """
        Replace the grid of the stacked data with the output grid. Unstack the
        stacked data and set the output grid as grid of the returned data.

        Parameters
        ----------
        remapped_data : :py:class:`numpy.ndarray`
            The remapped data which should be converted into a xarray with
            ``grid_out``'s coordinates.
        original_data : :py:class:`xarray.DataArray`
            This original data array is used as template for the new data array.

        Returns
        -------
        gridded_data : :py:class:`xarray.DataArray`
            The unstacked data with replaced grid. The output grid is accessible
            with :py:attr:``gridded_data.lawr.grid``.
        """
        data_coords = {dim: original_data[dim] for dim in original_data.dims
                       if dim != 'grid'}
        data_coords['grid'] = self._grid_out.get_multiindex()
        data_array = xr.DataArray(
            data=remapped_data,
            coords=data_coords,
            dims=original_data.dims,
            attrs=original_data.attrs,
            name=original_data.name
        )
        if len(self._grid_out.grid_shape) > 1:
            data_array = data_array.unstack('grid')
        gridded_data = data_array.lawr.set_grid_coordinates(self._grid_out)
        return gridded_data

    @abc.abstractmethod
    def _remap_method(self, data):
        pass

    @log_decorator(logger)
    def remap(self, data):
        """
        Remap the given data with the optimized parameters and the given
        grids.

        Parameters
        ----------
        data : :py:class:`xarray.DataArray`
            This data is remapped from grid_in to grid_out. The data should
            have the same shape as the input grid shape.

        Returns
        -------
        remapped_data : :py:class:`xarray.DataArray`
            The array with the remapped data based on grid_out. The grid_out
            is accessible via ``remapped_data.lawr.Grid``.

        Raises
        ------
        NotFittedError
            A NotFittedError is raised if the fit method was not called yet.

        Warnings
        --------
        Make sure that there are no NaNs are within the grid because the mapping
        from input grid to output grid is static and therefore not "NaN-safe".
        To fill up NaN values you can use ``pylawr.filter.interpolation``.
        """
        if not self.fitted:
            raise NotFittedError('This interpolation was not fitted yet!')
        stacked_data = self._stack_grid_coords(data)
        stacked_remap_data = self._remap_method(stacked_data)
        remapped_data = self._array_postprocess(stacked_remap_data,
                                                stacked_data)
        return remapped_data

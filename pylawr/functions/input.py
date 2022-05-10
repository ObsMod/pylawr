#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import xarray as xr

# Internal modules
from pylawr.datahandler import LawrHandler, DWDHDF5Handler
from pylawr.grid import PolarGrid
from pylawr.transform.spatial.beamexpansion import TAG_BEAM_EXPANSION_CORR
from pylawr.utilities.decorators import log_decorator
from pylawr.utilities.conventions import naming_convention

logger = logging.getLogger(__name__)


@log_decorator(logger)
def read_lawr_ascii(file_handler, grid=None):
    """
    Read in ascii-data from X-band local area weather radars of the University
    Hamburg.

    Parameters
    ----------
    file_handler : file handler object
        This file handler is used to read in the data. This file handler needs
        to be opened.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None, optional
        This grid is used to set coordinates of read in reflectivity. If no
        grid is given, a :py:class:~pylawr.grid.polar.PolarGrid` with default
        arguments is initialized. Default is None.

    Returns
    -------
    read_refl : :py:class:`xarray.DataArray`
        Read-in logarithmic reflectivity. The beam expansion is corrected and
        added as tag.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None
        This grid was used to set grid coordinates of returned reflectivity.
    """
    if grid is None:
        grid = PolarGrid()
    data_handler = LawrHandler(file_handler)
    read_refl = data_handler.get_reflectivity()
    read_refl = read_refl.lawr.set_grid_coordinates(grid)
    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
    logger.info('Read netcdf with first timestamp ' +
                str(read_refl.time.values[0]))
    return read_refl, grid


@log_decorator(logger)
def read_lawr_nc_level0(file_path, grid=None, height=95,
                        azi_offset=0., beam_ele=3., keep_attrs=True):
    """
    Read in NETCDF level 0 data from X-band local area weather radars of the
    University Hamburg.

    Parameters
    ----------
    file_path : str
        This file path is opened as :py:class:`~xarray.Dataset`.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None, optional
        This grid is used to set coordinates of read in reflectivity. If no
        grid is given, a :py:class:~pylawr.grid.polar.PolarGrid` with the
        arguments specified in level 0 data is initialized. Default is None.
    height : float, optional
        The heigth of the radar. The height is not specified in level 0 data.
        Default is 95 m.
    azi_offset : float, optional
        The azimuth offset in degrees used for
        :py:class:~pylawr.grid.polar.PolarGrid`. Default is 0.
    beam_ele : float, optional
        The beam elevation in degrees used for
        :py:class:~pylawr.grid.polar.PolarGrid`. Default is 3. The usage of
        the beam elevation given by level 0 data is not recommended.
    keep_attrs : bool, optional
         If the original attributes of the level 0 file will be kept.
         Default is True.

    Returns
    -------
    read_refl : :py:class:`xarray.DataArray`
        Read-in logarithmic reflectivity. The beam expansion is corrected and
        added as tag.
    grid : child of :py:class:`~pylawr.grid.polar.PolarGrid`
        This grid is specified in level 0 data of reflectivity
    """
    dataset = xr.open_dataset(file_path, engine='netcdf4')

    coordinates = dict(
        time=(['time', ],
              dataset.Time.values,
              naming_convention['time']),
        azimuth=(['azimuth', ],
                 dataset.Azimuth.values,
                 naming_convention['azimuth']),
        range=(['range', ],
               dataset.Distance.values,
               naming_convention['range'])
    )

    attrs = naming_convention['dbz']
    if keep_attrs:
        attrs.update(dataset.attrs)

    read_refl = xr.DataArray(
        data=dataset["Polar_Reflectivity"].values,
        coords=coordinates,
        dims=['time', 'azimuth', 'range'],
        attrs=attrs)

    read_refl = read_refl.lawr.set_variable('dbz')

    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)

    if grid is None:
        grid = PolarGrid(center=(dataset.attrs["latitude"],
                                 dataset.attrs["longitude"],
                                 height),
                         beam_ele=beam_ele,
                         azi_offset=azi_offset,
                         )

    read_refl = read_refl.lawr.set_grid_coordinates(grid)

    read_refl = read_refl.compute()
    dataset.close()

    return read_refl, grid


@log_decorator(logger)
def read_lawr_nc_old(file_path, grid=None, height=95):
    """
    Read in NETCDF level 1 data from X-band local area weather radars of the
    University Hamburg. The highest processed reflectivity will be returned
    (beam expansion corrected, clutter and noise removed, X-Band-Correction and
    C-Band-Correction is applied).

    Parameters
    ----------
    file_path : str
        This file path is opened as :py:class:`~xarray.Dataset`.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None, optional
        This grid is used to set coordinates of read in reflectivity. If no
        grid is given, a :py:class:~pylawr.grid.polar.PolarGrid` with the
        default arguments is initialized. Default is None.
    height : float
        The heigth of the radar. The height is not specified in level 1 data.
        Default is 95 m.

    Returns
    -------
    read_refl : :py:class:`xarray.DataArray`
        Read-in logarithmic reflectivity. The beam expansion is corrected,
        clutter and noise is removed. The old C-Band-Correction is applied.
        Some tags are added.
    grid : child of :py:class:`~pylawr.grid.polar.PolarGrid`
        This grid is specified in level 0 data of reflectivity
    """
    dataset = xr.open_dataset(file_path, engine='netcdf4')

    coordinates = dict(
        time=dataset.Time.values,
        azimuth=dataset.Azimuth.values,
        range=dataset["Att_Corr_Cband_Reflectivity"].dist.values)

    attrs = dict(unit='dBZ')
    attrs.update(dataset.attrs)

    read_refl = xr.DataArray(
        data=dataset["Att_Corr_Cband_Reflectivity"].values,
        coords=coordinates,
        dims=['time', 'azimuth', 'range'],
        attrs=attrs)

    read_refl = read_refl.lawr.set_variable('dbz')

    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)

    read_refl.lawr.add_tag('old algorithms: ' + dataset.used_algorithms)

    if grid is None:
        grid = PolarGrid(center=(float(dataset.latitude[:-1]),
                                 float(dataset.longitude[:-1]),
                                 height),
                         beam_ele=float(dataset.elevation),
                         )

    read_refl = read_refl.lawr.set_grid_coordinates(grid)

    read_refl = read_refl.compute()
    dataset.close()

    return read_refl, grid


@log_decorator(logger)
def read_lawr_nc_new(file_path, grid=None):
    """
    Read in LAWR data in NetCDF format. This function can be used to read in
    NetCDF data, which was previously saved with this software package. This
    function opens file path as :py:class:`~xarray.Dataset`, where then
    ``reflectivitiy`` is used as variable. If a grid is given, then the grid
    is set and returned. In the future, the grid will be inferred from NetCDF
    file.

    Parameters
    ----------
    file_path : str
        This file path is opened as :py:class:`~xarray.Dataset` and should have
        ``reflectivity`` as variable.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None, optional
        This grid is used to set coordinates of read in reflectivity. If no
        grid is given, the grid will be inferred from attributes in Dataset.
        Default is None.

    Returns
    -------
    read_refl : :py:class:`xarray.DataArray`
        Read-in logarithmic reflectivity. The beam expansion is corrected and
        added as tag.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None
        This grid was used to set grid coordinates of returned reflectivity.
    """
    with xr.open_dataset(file_path, engine='netcdf4') as dataset:
        read_refl = dataset['dbz'].compute()
    if grid is not None:
        read_refl = read_refl.lawr.set_grid_coordinates(grid)
    # TODO: Add function to decode grid from NetCDF
    logger.info('Read netcdf with first timestamp ' +
                str(read_refl.time.values[0]))
    return read_refl, grid


@log_decorator(logger)
def read_dwd_hdf5(file_handler, grid=None):
    """
    Read in DWD data in HDF5 format. This function also can be used to read in
    any OPERA Data Information Model (ODIM) conform HDF5 data, but there is no
    guarantee that this works for any data.

    Parameters
    ----------
    file_handler : filelike object
        This file handler is passed to
        :py:class:`~pylawr.datahandler.hdf5.DWDHDF5Handler` and this file
        handler has to be open.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None, optional
        This grid is used to set coordinates of read in reflectivity. If no
        grid is given, a :py:class:~pylawr.grid.polar.PolarGrid` with default
        arguments is initialized. Default is None.

    Returns
    -------
    read_refl : :py:class:`xarray.DataArray`
        Read-in logarithmic reflectivity. The beam expansion is corrected and
        added as tag.
    grid : child of :py:class:`~pylawr.grid.base.BaseGrid` or None
        This grid was used to set grid coordinates of returned reflectivity.
    """
    data_handler = DWDHDF5Handler(file_handler)
    read_refl = data_handler.get_reflectivity()
    if grid is None:
        grid = data_handler.grid
    read_refl = read_refl.lawr.set_grid_coordinates(grid)
    read_refl.lawr.add_tag(TAG_BEAM_EXPANSION_CORR)
    return read_refl, grid

#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import os
import shutil

# External modules

# Internal modules
from pylawr.utilities.decorators import log_decorator


logger = logging.getLogger(__name__)


@log_decorator(logger)
def save_netcdf(xr_ds, save_path, encoding=None):
    """
    Save a :py:class:`xarray.Dataset` to NetCDF with given save path securely.
    This function saves the NetCDF to a temporary path. This file is then moved
    to given save path. This ensures that the file cannot be written and opened
    at the same time.

    Warning
    -------
    Time `encoding` like
    `encoding={'time': {'units': 'days since 1970-01-01'}}` results in
    inconsistencies (only microseconds are affected).

    Parameters
    ----------
    xr_ds : :py:class:`xarray.Dataset` or :py:class:`xarray.DataArray`
        This dataset is saved as NetCDF file to given ``save_path`` and can be
        opened with any NetCDF library.
    save_path : str
        The dataset is saved to this path. The temporary path is also based on
        specified file name in this path.
    """
    _, filename = os.path.split(save_path)
    tmp_path = '/tmp/{0:s}'.format(filename)
    xr_ds.to_netcdf(tmp_path, encoding=encoding)
    shutil.move(tmp_path, save_path)

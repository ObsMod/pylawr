#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import abc

# External modules

# Internal modules
from ..grid import GridNotAvailableError


logger = logging.getLogger(__name__)


class DataHandler(object):
    """
    The data handler is responsible for a single opened file. The data
    handler
    has methods to decode the data from this file.

    Parameters
    ----------
    fh : filelike object
        The data is read from this filelike object.
    """
    def __init__(self, fh):
        self._data = None
        self._fh = None
        self._available_dates = None
        self.fh = fh

    def close(self):
        try:
            self._data.close()
        except AttributeError:
            pass
        self._data = None

    @property
    def fh(self):
        return self._fh

    @fh.setter
    def fh(self, fh):
        try:
            fh.readable()
        except AttributeError:
            raise TypeError('The given object is not a valid filelike object!')
        self._fh = fh

    @abc.abstractmethod
    def get_reflectivity(self):
        """
        Get the radar reflectivity from this file handler.

        Returns
        -------
        radar_field : xarray.DataArray or None
            The radar field from this handler with the date, azimuth and
            range as dimension. The attributes are the decoded header
            information.
        """
        pass

    @property
    def available_dates(self):
        if self._available_dates is None:
            self._available_dates = self._get_available_dates()
        return self._available_dates

    @abc.abstractmethod
    def _get_available_dates(self):
        pass

    @property
    def grid(self):
        """
        The get_grid method could be used to decode the grid from the file.

        Returns
        -------
        grid : Gridlike type
            The readed grid from the file.

        Raises
        ------
        ValueError
            The grid couldn't decoded from the file.
        """
        raise GridNotAvailableError(
            'There is no way to decode the grid from the file!')

#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging
import abc
import datetime

# external modules
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class MemoryMixin(abc.ABC):
    """
    Abstract base class for dynamic filters (e.g. static clutter or noise is
    dependent of the recent states)
    """

    def __init__(self):
        self._trainable_vars = ()

    def reset(self):
        """
        Reset the trainable values of this filter. The trainable values will
        be set to their default value.
        """
        for var in self._trainable_vars:
            setattr(self, var, None)
        logger.warning(
            'The filter {0:s} is reset!'.format(self.__class__.__name__)
        )

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the filter to a new state
        """
        pass

    def set_xr_params(self, ds):
        """
        The parameters from an xarray dataset are read and converted into
        attributes of this filter.

        Parameters
        ----------
        ds : xarray.Dataset
            the filter parameters
        """
        for var in ds.variables:
            setattr(self, var, ds[var].values)

    @classmethod
    def from_xarray(cls, ds):
        """
        Create a filter from a given parameters dataset.

        Parameters
        ----------
        ds : :any:`xarray.Dataset`
            a filter parameters dataset

        Returns
        -------
        Filter
            the filter with the specified parameters
        """
        f = cls()
        f.set_xr_params(ds)
        return f

    @abc.abstractmethod
    def to_xarray(self):
        """
        Serialize this filter's parameters to an :any:`xarray.Dataset`

        Returns
        -------
        :any:`xarray.Dataset`
            the filter's parameters as dataset
        """
        pass

    @property
    @abc.abstractmethod
    def fitted(self):
        pass

    @staticmethod
    def _get_time(array, time_obj=None):
        """
        Get the time from array or given time_obj.

        Parameters
        ----------
        array: array-like
            The **linear** reflectivity :math:`Z`
        time_obj : datetime.datetime, numpy.datetime64 or None
            The time ``array`` was recorded. If None, the time will be
            determined from ``array.time``, if available, otherwise it will be
            set to :any:`datetime.datetime.now`. Default is None.

        Returns
        -------
        time_obj : numpy.datetime64
            The time from given time_obj or determined automatically as
            numpy.datetime64 object.
        """
        logger.debug("Determining the time of this reflectivity dataarray")
        if time_obj is None:
            try:
                time_obj = array.time.values[0]
                assert time_obj.size == 1
                logger.debug("Using the dataarray's 'time' attribute value")
            except IndexError:
                time_obj = array.time.values
            except (AttributeError, AssertionError):
                time_obj = datetime.datetime.now()
                logger.debug("Dataarray has no 'time' attribute. Using current "
                             "system time.")
        elif not isinstance(time_obj, (datetime.datetime, np.datetime64)):
            raise TypeError(
                'The given time object needs to be `datetime.datetime`, '
                '`numpy.datetime64` or None!'
            )
        time_obj = np.datetime64(time_obj)
        return time_obj

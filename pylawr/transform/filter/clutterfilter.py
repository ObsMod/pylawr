#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging
import abc

logger = logging.getLogger(__name__)


class ClutterFilter(abc.ABC):
    """
    Abstract base class for clutter filters

    Attributes
    ----------
    threshold: float
            The threshold of the reflectivities to filter (dBZ)
    """

    def __init__(self, threshold):
        self._threshold = threshold
        self.map = None

    @abc.abstractmethod
    def calc_map(self, refl):
        """
        Computes field for detecting clutter based on the filter.

        Parameters
        ----------
        refl : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        """
        pass

    @abc.abstractmethod
    def create_cluttermap(self, refl=None):
        """
        Creates clutter map based on the field and threshold.

        Returns
        -------
        cmap :  :py:class:`~pylawr.transform.filter.ClutterMap`
                The cluttermap of ``_map`` to transform the reflectivity,
                contains probabilities of clutter
                (between one and zero).
        refl : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        """
        pass

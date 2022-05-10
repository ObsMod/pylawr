#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging

# external modules
import numpy as np

# internal modules
from pylawr.transform.filter.clutterfilter import ClutterFilter
from pylawr.transform.filter.cluttermap import ClutterMap
from pylawr.utilities.decorators import log_decorator

logger = logging.getLogger(__name__)


class RINGFilter(ClutterFilter):
    """
    Calculates gradients of consecutive range gates for clutter detection.
    If the difference between consecutive range gates exceeds a threshold for
    percentage (window_criterion) of consecutive beams within a window,
    the gate is identified as clutter. This filter is similar to the
    `SPKFilter`, only the axis is changed.

    With the default parameters a ring of six consecutive beams is identified
    as clutter.

    Attributes
    ----------
    threshold: float
        The threshold of the reflectivities to filter (dBZ)
    _wz: int
        The number of consecutive beams in which the threshold has to
        be exceeded
    _wc: float
        The percentage of consecutive beam gates the criteria have to be
        fulfilled
    _rw: int
        The assumed ring width to detect.

    """
    def __init__(self, threshold=3., window_size=11, window_criterion=.5,
                 ring_width=1):
        super().__init__(threshold)
        self._threshold = threshold
        self._wz = window_size
        self._wc = window_criterion
        self._rw = ring_width
        self.map = None

    @log_decorator(logger)
    def calc_map(self, array):
        """
        Computes ring field for detecting clutter. The procedure deals also
        with arrays containing NaN values. The map is the number of
        threshold exceeds within the window for this gate.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        """
        # make sure given array is on db scale
        array = array.lawr.to_dbz()

        # gradients for adjacent beams with assumed spike width to detect
        diff1 = np.subtract(array.values,
                            np.roll(array, -self._rw, 2))
        diff2 = np.subtract(array.values,
                            np.roll(array, self._rw, 2))

        exceed = np.zeros(diff1.shape)
        exceed[np.logical_and(
            np.greater_equal(diff1, self._threshold),
            np.greater_equal(diff2, self._threshold))] = 1

        # number of exceeded gradient thresholds
        self.map = np.zeros(array.shape)

        # compute the exceedance in adjacent range gates
        for cgate in range(int(-self._wz/2),
                           int(self._wz/2)+1):
            self.map += np.roll(exceed, cgate, 1)

    def create_cluttermap(self, array=None, addname=''):
        """
        Creates clutter map based on the field and threshold.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        addname : str
            Supplement to the name of the created
            :py:class:`~pylawr.transform.filter.ClutterMap`
            (should be unique), otherwise appended cluttermaps overwrite
            the results.

        Returns
        -------
        cmap :  :py:class:`~pylawr.transform.filter.ClutterMap`
                The cluttermap of ``_map`` to transform the reflectivity,
                contains probabilities of clutter (between one and zero).
        """
        if array is not None:
            self.calc_map(array)

        clutter = np.zeros(self.map.shape)
        clutter[self.map >= (self._wc * self._wz)] = 1

        return ClutterMap((str(self.__class__.__name__) + str(addname)),
                          clutter)

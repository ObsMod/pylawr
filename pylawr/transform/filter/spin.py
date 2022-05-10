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


class SPINFilter(ClutterFilter):
    """
    Calculate SPIN change of the reflectivity for clutter detection according to
    Hubbert et al. (2009).

    Attributes
    ----------
    threshold: float
            The threshold of the reflectivities to filter (dBZ)
    _wz: int
            The number of consecutive range gates in which the threshold has to
            be exceeded
    _wc: float
        The percentage of consecutive range gates the criteria have to be
        fulfilled
    """
    def __init__(self, threshold=3, window_size=11, window_criterion=0.1):
        super().__init__(threshold)
        self._threshold = threshold
        self._wz = window_size
        self._wc = window_criterion
        self.map = None

    @log_decorator(logger)
    def calc_map(self, array):
        """
        Computes SPIN field for detecting clutter. The SPIN feature field is
        a measure of how often the reflectivity gradient changes sign along
        the distance. The procedure deals also with arrays containing NaN
        values.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        """
        # make sure given array is on db scale
        array = array.lawr.to_dbz()

        # pad the array with two to caclulate the gradients
        array = np.pad(array, ((0, 0), (0, 0), (1, 1)), 'reflect')

        # first gradient: X_i - X_i-1
        grad_left = np.subtract(array,
                                np.roll(array, 1, 2))

        # second gradient: X_i+1 - X_i
        grad_right = np.subtract(np.roll(array, -1, 2),
                                 array)

        # reshape the two arrays of gradients on normal size
        grad_left = grad_left[:, :, 1:(array.shape[2] - 1)]

        grad_right = grad_right[:, :, 1:(array.shape[2] - 1)]

        # pad the two arrays of gradients for window size
        nb_pad = int(self._wz / 2) + 1

        grad_left = np.pad(grad_left,
                           ((0, 0), (0, 0), (nb_pad, nb_pad)), 'reflect')

        grad_right = np.pad(grad_right,
                            ((0, 0), (0, 0), (nb_pad, nb_pad)), 'reflect')

        # first condition: sign{X_i - X_i-1} = -sign{X_i+1 - X_i}
        map_sign = np.multiply(np.sign(grad_left), -1.*np.sign(grad_right))

        # if positive (same sign) condition is fulfilled, otherwise the map
        # should be zero
        map_sign[map_sign == -1] = 0
        map_sign[np.isnan(map_sign)] = 0

        # second condition: {|X_i - X_i-1| + |X_i +1 - X_i|} / 2. > SPIN_thres
        map_mean = np.add(abs(grad_left), abs(grad_right)) / 2.

        map_mean[np.isnan(map_mean)] = 0
        map_mean[map_mean <= self._threshold] = 0
        map_mean[map_mean > self._threshold] = 1

        # combine conditions within a window of 11 range gates around the
        # centre range gate
        self.map = np.zeros(map_mean.shape)

        for cgate in range(int(-self._wz / 2), int(self._wz / 2) + 1):
            self.map += np.multiply(np.roll(map_sign, cgate, 2),
                                    np.roll(map_mean, cgate, 2))

        # reshape after padding
        self.map = self.map[:, :, nb_pad:(self.map.shape[2] - nb_pad)]

        # relative number of consecutive range gates detecting clutter
        self.map = self.map / self._wz

    def create_cluttermap(self, array=None, addname=''):
        """
        Creates clutter map based on the field and window criteria. If both
        criteria of the SPIN filter are fulfilled in more than `self._wc`
        percent of the consecutive range gates, the centre range gate is
        flagged as clutter.

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

        clutter[self.map >= self._wc] = 1

        return ClutterMap((str(self.__class__.__name__) + str(addname)),
                          clutter)

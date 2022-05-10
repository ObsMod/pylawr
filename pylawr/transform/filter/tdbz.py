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


class TDBZFilter(ClutterFilter):
    """
    Calculate texture of reflectivity (TDBZ) for clutter detection according to
    Hubbert et al. (2009).

    Attributes
    ----------
    threshold: float
            The threshold of the reflectivities to filter (dBZ)
    _wz: int
            The number of consecutive range gates in which the threshold has to
            be exceeded
    """
    def __init__(self, threshold=3., window_size=5):
        super().__init__(threshold)
        self._threshold = threshold
        self._wz = window_size
        self.map = None

    @log_decorator(logger)
    def calc_map(self, array):
        """
        Computes TDBZ field for detecting clutter. The TDBZ field is computed
        as the average of the squared logarithmic reflectivity difference
        between adjacent range gates. The procedure deals also with arrays
        containing NaN values.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
               The reflectivity array to operate on.
        """
        # make sure given array is on db scale
        array = array.lawr.to_dbz()

        # pad array to fulfill boundary conditions
        nb_pad = int(self._wz / 2) + 1
        array = np.pad(array,
                       ((0, 0),
                        (0, 0),
                        (nb_pad, nb_pad)),
                       'reflect')

        # initialise var to calculate the sq. log. reflectivity difference
        diffsq = np.zeros(array.shape + (self._wz,))

        # compute the sq. log. ref. diff. for each range gate used
        for cgate in range(int(-self._wz/2),
                           int(self._wz/2)+1):
            diffsq[:, :, :, cgate] = np.square(
                np.subtract(np.roll(array, cgate, 2),
                            np.roll(array, (cgate - 1), 2))
            )

        # compute average
        self.map = np.nanmean(diffsq, 3)

        # reshape after padding
        self.map = self.map[:, :, nb_pad:(self.map.shape[2]-nb_pad)]

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
        clutter[self.map >= self._threshold] = 1

        return ClutterMap((str(self.__class__.__name__) + str(addname)),
                          clutter)

#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import scipy.signal

# Internal modules
from .clutterfilter import ClutterFilter
from .cluttermap import ClutterMap
from pylawr.utilities.decorators import log_decorator
from pylawr.utilities.helpers import polar_padding


logger = logging.getLogger(__name__)


class SpeckleFilter(ClutterFilter):
    """
    The speckle filter checks the number of rain pixels in a neighborhood. If
    this number of rain pixels is lower than a given threshold, this pixel is
    declared as clutter. The boundary is zero-padded.

    Attributes
    ----------
    threshold: int, optional
        The threshold of the rain pixels in the neighborhood (default=2)
    window_size : tuple(int, int)
        This window size is checked (default=(3, 3))
    """
    def __init__(self, threshold=2, window_size=(3, 3)):
        super().__init__(threshold=threshold)
        self.window_size = window_size

    @property
    def window_product(self):
        return np.product(self.window_size)

    @log_decorator(logger)
    def calc_map(self, array):
        """
        This method estimates the number of rain pixels in the neighborhood and
        if the center pixel is also a rain pixel.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            This data array is used to determine the number of rain pixels and
            is used to estimate the raw cluttermap without threshold.
        """
        rain_map = array.lawr.get_rain_mask(threshold=-10).astype(int).squeeze()
        padding_size = [int((s-1)/2) for s in self.window_size]
        padded_rain_map = polar_padding(rain_map, padding_size)
        conv_kernel = np.ones(self.window_size)
        rain_pixels = scipy.signal.convolve2d(
            padded_rain_map, conv_kernel, mode='valid'
        )
        self.map = rain_map * self.window_product - rain_pixels

    def create_cluttermap(self, array=None, addname=''):
        """
        Creates clutter map based on field, set neighborhood size and set
        threshold. If the number of rain pixels in the neighborhood is lower
        than given threshold, the center pixel is declared as clutter.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray` or None, optional
            This data array is used to determine the number of rain pixels and
            is used to estimate the cluttermap. If this is None (default), the
            stored clutter map is used.
        addname : str
            Supplement to the name of the created
            :py:class:`~pylawr.transform.filter.ClutterMap`
            (should be unique), otherwise appended `ClutterMaps` overwrite
            the results.

        Returns
        -------
        cmap : :py:class:`pylawr.transform.filter.clutter.ClutterMap`
                The Cluttermap of ``_map`` to transform the reflectivity,
                contains probabilities of clutter (between one and zero).
        """
        if array is not None:
            self.calc_map(array)
        clutter = np.zeros(self.map.shape)
        threshold = self.window_product - self._threshold
        clutter[self.map > threshold] = 1
        return ClutterMap((str(self.__class__.__name__) + str(addname)),
                          clutter[None, ...])

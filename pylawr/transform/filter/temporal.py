#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np

# Internal modules
from .clutterfilter import ClutterFilter
from .cluttermap import ClutterMap
from ..memorymixin import MemoryMixin


logger = logging.getLogger(__name__)


class TemporalFilter(MemoryMixin, ClutterFilter):
    """
    The temporal clutter filter compares current radar image with `n`-saved
    radar images. If the number of rain pixels within the last `n`-images is
    below a given threshold, then the pixel is identified as clutter. In this
    implementation ``fit`` and ``calc_map`` are split, because they are applied
    at different times: fit is normally applied before any clutter is removed
    and calc map is only used to identify current rain pixels and can be applied
    at any time, also after removing some clutter.

    Parameters
    ----------
    store_n_images : int, optional
        This number of images (default=3) is stored within this clutter filter.
    threshold : int, optional
        This threshold (default=3) is used to decide if a rain pixel is clutter
        or not. If the number of rain pixels within the stored images is below
        this threshold, the rain pixel is identified as clutter.

    Notes
    -----
    :py:meth:`~pylawr.transform.filter.temporal.TemporalFilter.fit` should be
    normally applied before any other clutter filter was applied, while
    :py:meth:`~pylawr.transform.filter.temporal.TemporalFilter.calc_map` can be
    used anywhere.
    """
    def __init__(self, store_n_images=3, threshold=3):
        super().__init__()
        self._threshold = threshold
        self.store_n_images = store_n_images
        self._hist_maps = None
        self._map = None

    def fit(self, array):
        """
        This fit method sets the history maps, which are used to determine
        clutter. Also the maps are constrained to the last `n`-maps. This method
        is independent from :py:meth:`calc_map`.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            This array is concatenated to the historic maps.
        """
        rain_map = (array.lawr.to_dbz().values > 5).astype(int).squeeze()
        rain_map = rain_map[None, ...]
        if self._hist_maps is None:
            self._hist_maps = rain_map
        else:
            self._hist_maps = np.concatenate([self._hist_maps, rain_map],
                                             axis=0)
            self._hist_maps = self._hist_maps[-self.store_n_images:]

    @property
    def fitted(self):
        """
        If the number of stored images equals the set number of stored images.

        Returns
        -------
        fitted : bool
            If the number of stored images equals the set number of stored
            images.
        """
        fitted = False
        if self._hist_maps is not None:
            fitted = self._hist_maps.shape[0] == self.store_n_images
        return fitted

    def to_xarray(self):
        pass

    def calc_map(self, array):
        """
        This converts given array to rain pixels and sets this rain array as
        current map. This method set current map independent from
        :py:meth:`fit`.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            This array is converted into rain pixels.
        """
        array_dbz = array.lawr.to_dbz()
        self._map = (array_dbz.values > 5).astype(int).squeeze()

    def create_cluttermap(self, array=None, addname=''):
        """
        This method creates the cluttermap based on historic rain maps, current
        rain map and set threshold.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray` or None, optional
            If this array is given, then this will be converted into rain pixels
            and set as current map.
        addname : str
            Supplement to the name of the created `ClutterMap`
            (should be unique), otherwise appended `ClutterMaps` overwrite
            the results.

        Returns
        -------
        clutter_map : :py:class:`pylawr.transform.filter.cluttermap.ClutterMap`
            This is the cluttermap of this filter, dependent on stored images.

        Notes
        -----
        If the number of stored images is lower than the set attribute
        ``store_n_images``, an empty clutter map will be  returned.
        """
        if array is not None:
            self.calc_map(array)
        if self.fitted:
            sum_rain = np.sum(self._hist_maps, axis=0)
            thres_exceeded = sum_rain < self._threshold
            clutter = (thres_exceeded*self._map).astype(int)
        else:
            clutter = np.zeros_like(self._map)

        return ClutterMap((str(self.__class__.__name__) + str(addname)),
                          clutter[None, ...])

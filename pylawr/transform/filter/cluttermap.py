#!/bin/env python
# -*- coding: utf-8 -*-

# system modules
import logging
from collections import OrderedDict
from copy import deepcopy

# external modules
import xarray
import numpy as np

# internal modules
from pylawr.transform.transformer import Transformer
from pylawr.field import tag_array


logger = logging.getLogger(__name__)


class ClutterMap(Transformer):
    """
    Class to transform/filter the `RadarField` on basis of a clutter map,
    calculated with calc_cluttermap of `ClutterFilter`

    Parameters
    ----------
    name : string
        The name of used clutter filter.
    cmap : :py:class:`numpy.ndarray`
        Probability of clutter for each grid point.
    fuzzy_threshold : float
        Relative value to the number of clutter maps, which indicate clutter,
        to fulfill clutter condition. The interval of this fuzzy threshold is
        [0, 1]. A value of 1 describes that every cluttermap has to detect
        clutter so that this value is filtered out.

    Attributes
    ----------
    layers : :py:class:`collections.OrderedDict`
        Dictionary with zero to many clutter maps
    weights : :py:class:`collections.OrderedDict`
        Dictionary with the weights to corresponding clutter maps.
    fuzzy_threshold : float
        Relative value to the number of clutter maps, which indicate clutter,
        to fulfill clutter condition. The interval of this fuzzy threshold is
        [0, 1]. A value of 1 describes that every cluttermap has to detect
        clutter so that this value is filtered out.
    """
    def __init__(self, name=None, cmap=None, fuzzy_threshold=1):
        self.layers = OrderedDict()
        self.weights = OrderedDict()
        if name is not None and cmap is not None:
            self.layers[name] = cmap
            self.weights[name] = 1
        self.fuzzy_threshold = fuzzy_threshold

    def __str__(self):
        layer_weights = ['{0}*{1:s}'.format(self.weights[k], k)
                         for k in self.layers.keys()]
        weighted_str = ', '.join(layer_weights)
        return 'ClutterMap({0:s})'.format(weighted_str)

    def __getitem__(self, item):
        new_cmap = self.__class__(name=item, cmap=self.layers[item])
        return new_cmap

    def __setitem__(self, key, value):
        if key not in self.layers.keys():
            self.layers[key] = value
            self.weights[key] = 1
        else:
            raise KeyError(
                'A cluttermap with name {0:s} already exists'.format(key)
            )

    def mean(self):
        """
        Get the weighted mean of this cluttermap. The cluttermap is weighted
        with set weights.

        Returns
        -------
        mean_values : py:class:`numpy.ndarray`
            The weighted mean values with the same shape as a single cluttermap.
        """
        weights = list(self.weights.values())
        mean_values = np.average(self.array, axis=0, weights=weights)
        return mean_values

    @property
    def array(self):
        """
        Get the values of the cluttermaps as array.

        Returns
        -------
        layer_array : :py:class:`numpy.ndarray`
            The concatenated values of the different cluttermaps in preserved
            order. The cluttermaps are concatenated over a new first dimension.
        """
        layer_values_list = list(self.layers.values())
        layer_array = np.concatenate(layer_values_list)
        if layer_array.ndim < 3:
            layer_array = layer_array[None, ...]
        return layer_array

    def transform(self, array, *args, **kwargs):
        """
        Transform a given ``array`` with a polar grid according to the
        ``ClutterMap`` object. The clutter maps in the layers attribute are
        applied on the radar field. If the number of clutter maps are greater
        or equal to the fuzzy threshold at a specific grid point, clutter is
        detected.

        Parameters
        ----------
        array : :py:class:`xarray.DataArray`
            The reflectivity array to operate on.

        Returns
        -------
        transformed_array : :py:class:`xarray.DataArray`
            The array with removed clutter as NaN-values
        """
        clutter = np.zeros_like(array)
        transformed_array = deepcopy(array)
        try:
            clutter += self.mean()
        except ValueError:
            pass
        if self.fuzzy_threshold != 0:
            clutter = clutter >= self.fuzzy_threshold
        else:
            clutter = clutter > self.fuzzy_threshold
        transformed_array.values[clutter] = np.nan
        logger.info('{0:s} removed {1:d} clutter pixels'.format(
            str(self), int(np.sum(clutter)))
        )
        tag_array(transformed_array, 'filtered with {0:s}'.format(str(self)))
        try:
            transformed_array = transformed_array.lawr.set_grid_coordinates(
                array.lawr.grid
            )
        except TypeError:
            pass
        return transformed_array

    def append(self, other_clutter):
        """
        Appends an other instance of
        :py:class:`pylawr.filter.cluttermap.ClutterMap` to this instance.

        Parameters
        ----------
        other_clutter : :py:class:`pylawr.filter.cluttermap.ClutterMap`
            Some instance of this class. The layers of this class are used to
            update the layers of this cluttermap.
        """
        for k, v in other_clutter.layers.items():
            if self.layers and v.shape != list(self.layers.values())[0].shape:
                raise ValueError('The cluttermap of {0:s} has not the same '
                                 'shape as this cluttermap!'.format(k))
            self.layers[k] = v
            self.weights[k] = other_clutter.weights[k]

#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
from copy import deepcopy

# External modules
import numpy as np

# Internal modules
from .tree import TreeRemap


logger = logging.getLogger(__name__)


class NearestNeighbor(TreeRemap):
    """
    NearestNeighbur implements the nearest neighbor remapping. To
    find the nearest neighbor a kd-tree is used. The n-nearest neighbors
    are averaged to obtain the field value.

    Parameters
    ----------
    n_neighbors : int, optional
        This number of nearest neighbour values is averaged to obtain the
        remapped value. Default is 1.
    max_dist : float or None, optional
        Use for remapping only neighbors within this distance to the
        interpolated point. If this maximum distance is None (default), no
        restriction is used (this number is internally converted to
        infinity), while an float number indicates a constrain in meters.
    """
    def __init__(self, n_neighbors=1, max_dist=None):
        super().__init__(n_neighbors=n_neighbors, max_dist=max_dist)

    def _remap_method(self, data):
        """
        This remap method uses the average of the n-nearest neighbors as field
        value.
        """
        neighbor_values = data.values[..., self._locs]
        neighbor_values[..., self._out_of_bound_locs] = np.nan
        remapped_data = np.nanmedian(neighbor_values, axis=-1)
        remapped_data[self._neighbors_not_available(neighbor_values)] = np.nan
        return remapped_data

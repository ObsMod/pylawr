#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
from scipy.spatial import cKDTree
import numpy as np

# Internal modules
from .base import BaseRemap
from pylawr.utilities.decorators import log_decorator


logger = logging.getLogger(__name__)


class TreeRemap(BaseRemap):
    """
    The TreeRemap is a base class for all remapping subclasses,
    where the n nearest neighbors are used. The remapping is based on
    scipy.spatial.cKDTree to get the nearest neighbor points.

    Parameters
    ----------
    n_neighbors : int, optional
        For every output grid point this number of neighbors is searched
        within the input grid points. Default is 20.
    max_dist : float or None, optional
        Use for remapping only neighbors within this distance to the
        interpolated point. If this maximum distance is None (default), no
        restriction is used (this number is internally converted to
        infinity), while an float number indicates a constrain in meters.
    """
    def __init__(self, n_neighbors=20, max_dist=None):
        super().__init__()
        self._tree = None
        self._dists = None
        self._locs = None
        self._max_dist = None
        self._inner_max_dist = 0.5
        self._inner_neighbors = 0.5
        self.max_dist = max_dist
        self.n_neighbors = n_neighbors

    def __str__(self):
        return_str = '{0:s}(n_neighbors={1:d})'.format(
            self.__class__.__name__,
            self.n_neighbors)
        return return_str

    @property
    def _out_of_bound_locs(self):
        return self._locs == -1

    @property
    def max_dist(self):
        return self._max_dist

    @max_dist.setter
    def max_dist(self, new_dist):
        if new_dist is None:
            self._max_dist = np.inf
        elif isinstance(new_dist, (int, float)):
            self._max_dist = new_dist
        else:
            raise TypeError('Given maximum distance is not a float or None!')

    @property
    def fitted(self):
        """
        Check if the remapping is fitted.

        Returns
        -------
        fitted : bool
            If this tree based remap object is fitted.
        """
        return self._dists is not None and self._locs is not None

    def _get_tree(self, prepared_src, prepared_trg):
        tree = cKDTree(prepared_src)
        dists, locs = tree.query(prepared_trg, k=self.n_neighbors,
                                 distance_upper_bound=self._max_dist)
        return dists, locs, tree

    def _neighbors_not_available(self, neighbor_vals):
        outer_radius_mask = self._all_neighbors_nan(neighbor_vals)
        # inner_radius_mask = self._inner_radius_neighors()
        # neighbors_avail_mask = np.logical_or(outer_radius_mask,
        #                                      inner_radius_mask)
        return outer_radius_mask
    #
    # def _inner_radius_neighors(self):
    #     dists_in_range = self._dists < (self.max_dist * self._inner_max_dist)
    #     sum_d_in_range = np.sum(dists_in_range.astype(int), axis=-1)
    #     n_neighbors_thres = int(self._inner_neighbors*self.n_neighbors)
    #     nan_mask = sum_d_in_range < n_neighbors_thres
    #     logger.debug(
    #         'Number of pixels where inner radius criterion is fulfilled: '
    #         '{0:d}'.format(np.sum(nan_mask))
    #     )
    #     return nan_mask

    @staticmethod
    def _all_neighbors_nan(neighbor_vals):
        sum_of_nans = np.sum(np.isnan(neighbor_vals).astype(int), axis=-1)
        nan_mask = sum_of_nans > 0
        logger.debug(
            'Number of pixels where not all neighbors are available: '
            '{0:d}'.format(np.sum(nan_mask))
        )
        return nan_mask

    @log_decorator(logger)
    def fit(self, grid_in, grid_out):
        """
        Fit the remapping for given grids. This fitting method searches the
        nearest neighbor points via kd-tree and sets corresponding distances and
        locations.

        Parameters
        ----------
        grid_in : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from this grid to another grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        grid_out : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from another grid to this grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        """
        prepared_src = self._prepare_grid(grid_in)
        prepared_trg = self._prepare_grid(grid_out)
        logger.debug('Source points: {0}'.format(prepared_src.shape))
        logger.debug('Target points: {0}'.format(prepared_trg.shape))
        dists, locs, tree = self._get_tree(prepared_src, prepared_trg)
        self._tree = tree
        if self.n_neighbors == 1:
            self._dists = dists[..., np.newaxis]
            self._locs = locs[..., np.newaxis]
        else:
            self._dists = dists
            self._locs = locs
        out_of_bound_locs = self._locs >= prepared_src.shape[0]
        self._locs[out_of_bound_locs] = -1

        self._grid_in = grid_in
        self._grid_out = grid_out

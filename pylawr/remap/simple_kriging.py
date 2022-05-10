#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import scipy.spatial.distance
import scipy.sparse.linalg

# Internal modules
from .base import NotFittedError
from .tree import TreeRemap
from .kernel import gaussian_rbf, WhiteNoise
from .kernel.base_ops import BaseKernel


logger = logging.getLogger(__name__)


DEFAULT_ALPHA = 1E-8


class SimpleKriging(TreeRemap):
    """
    Simple kriging is the base class for all kriging instances. Simple
    kriging
    is also known as Gaussian Process in machine learning with distances as
    specified predictor. Here, we assume that the interpolated / remapped
    field
    is stationary and the expectation of this field is zero everywhere. This
    kriging is based on localized kriging, where the number of localized
    neighbors can be specified. In this simple kriging, the weights are not
    normalized to 1 such that resulting weights are not physical
    plausible. For
    physical plausible kriging, please use
    :py:class:`~pylawr.remap.ord_kriging.OrdinaryKriging`.

    Parameters
    ----------
    kernel : child of :py:class:`~pylawr.remap.kernel.base_ops.BaseKernel`
    or None, optional
        This kernel is used to transform distances into a new feature space,
        where the features can be used as non-linear predictors. In
        kriging this
        is also called covariance function. This covariance function should
        resemble a typical precipitation covariance field. If this kernel is
        None, a default radial basis function kernel with white noise is
        used. Default is None.
    n_neighbors : int, optional
        This is the number of neighbors, used to interpolate / remap a
        specific point. Default is 10.
    max_dist : float or None, optional
        Use for remapping only neighbors within this distance to the
        interpolated point. If this maximum distance is None (default), no
        restriction is used (this number is internally converted to
        infinity), while an float number indicates a constrain in meters.
    alpha : float or None, optional
        This alpha value is used as regularization parameter to stabilize
        the kriging solution. This alpha parameter is added to the diagonal
        of the covariance matrix. This regularization is a kind of Tikhonov
        regularization and adds a small white noise kernel to the data.
    """
    def __init__(self, kernel=None, n_neighbors=10, max_dist=None, alpha=None):
        super().__init__(n_neighbors=n_neighbors, max_dist=max_dist)
        self._kernel = None
        self._weights = None
        self._alpha = None
        self.kernel = kernel
        self.alpha = alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        if isinstance(new_alpha, (int, float)):
            self._alpha = new_alpha
        elif new_alpha is None:
            self._alpha = DEFAULT_ALPHA
        else:
            raise TypeError('The regularization factor has to be either None '
                            'or a number!')

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, new_kernel):
        if isinstance(new_kernel, BaseKernel):
            self._kernel = new_kernel
        elif new_kernel is None:
            data_kernel = gaussian_rbf(length_scale=5000, stddev=5)
            noise_kernel = WhiteNoise(data_kernel.placeholders[0],
                                      noise_level=0.5)
            self._kernel = data_kernel + noise_kernel
        else:
            raise TypeError('Given kernel has to be a valid kernel or None!')

    @staticmethod
    def _get_distance_matrix(points):
        dist_matrix = scipy.spatial.minkowski_distance(
            points[..., None, :], points[..., None, :, :]
        )
        return dist_matrix

    def _const_rkhs_matrix(self, src_points):
        dist_matrix = self._get_distance_matrix(src_points)
        kernel_matrix = self.kernel(dist_matrix)
        diag_ind = np.diag_indices(self.n_neighbors)
        kernel_matrix[..., diag_ind[0], diag_ind[1]] += self.alpha
        return kernel_matrix

    def _get_k_matrix(self, dists):
        k_matrix = self.kernel(dists)
        return k_matrix

    @staticmethod
    def _estimate_weights(rkhs, target):
        weights = np.linalg.solve(rkhs, target)
        return weights

    def _remap_method(self, data):
        neighbor_values = data.values[..., self._locs]
        neighbor_values[..., self._out_of_bound_locs] = np.nan
        mean = np.mean(neighbor_values, axis=-1)
        interp_values = np.moveaxis(
            np.moveaxis(neighbor_values, -1, 0) - mean, 0, -1
        )
        remapped_data = mean + np.nansum(interp_values * self._weights, axis=-1)
        remapped_data[self._neighbors_not_available(interp_values)] = np.nan
        return remapped_data

    @property
    def covariance(self):
        """
        Get the point specific interpolation variance of this kriging instance.
        Every point variance is independently estimated. This variance reflects
        the interpolation uncertainty, if the field is stationary and has no
        bias compared to the kriging assumptions.

        Returns
        -------
        post_cov : :py:class:`numpy.ndarray`
            This is the estimated posterior variance, based on specified kernel
            and estimated weights.

        Raises
        ------
        NotFittedError
            A NotFittedError is raised if fit was not called yet.
        """
        if not self.fitted:
            raise NotFittedError('This kriging was not fitted yet!')
        k_matrix = self._get_k_matrix(self._dists)
        cov_reduction = np.sum(k_matrix * self._weights, axis=-1)
        prior_cov = self.kernel.diag(np.zeros_like(cov_reduction))
        post_cov = prior_cov - cov_reduction
        return post_cov

    def fit(self, grid_in, grid_out):
        """
        Fit the remapping for given grids. This fitting method searches the
        nearest neighbor points via kd-tree and sets corresponding distances and
        locations. This fitting method also estimates the kriging weights based
        on these distances and locations.

        Parameters
        ----------
        grid_in : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from this grid to another grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        grid_out : child of :py:class:`pylawr.grid.BaseGrid`
            The data is remapped from another grid to this grid. This grid
            needs to have :py:meth:`get_altitude` and :py:meth:`get_lat_lon`.
        """
        super().fit(grid_in, grid_out)
        src_points = self._prepare_grid(grid_in)
        local_points = src_points.values[self._locs, :]
        rkhs = self._const_rkhs_matrix(local_points)
        k_mat = self._get_k_matrix(self._dists)
        self._weights = self._estimate_weights(rkhs, k_mat)

#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging

# External modules
import numpy as np
import scipy.spatial.distance

# Internal modules
from .simple_kriging import SimpleKriging


logger = logging.getLogger(__name__)


class OrdinaryKriging(SimpleKriging):
    """
    Ordinary kriging is the default kriging class for interpolation purpose.
    Here, we assume that the interpolated / remapped field is stationary and
    the expectation of this field can be estimated by the mean. This
    kriging is based on localized kriging, where the number of localized
    neighbors can be specified. In this ordinary kriging, the weights are
    normalized to unity. The resulting weights are physically plausible.
    This constrain is introduced as lagrange multiplier.

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
        super().__init__(kernel=kernel, n_neighbors=n_neighbors,
                         max_dist=max_dist, alpha=alpha)

    def _const_rkhs_matrix(self, src_points):
        rkhs_matrix = super()._const_rkhs_matrix(src_points)

        kernel_matrix = np.ones(
            (src_points.shape[0], self.n_neighbors+1, self.n_neighbors+1)
        )
        kernel_matrix[..., :-1, :-1] = rkhs_matrix
        kernel_matrix[..., -1, -1] = 0
        return kernel_matrix

    def _get_k_matrix(self, dists):
        k_kernel = super()._get_k_matrix(dists)
        eval_kernel = np.ones((*dists.shape[:-1], self.n_neighbors+1))
        eval_kernel[..., :-1] = k_kernel
        return eval_kernel

    def _remap_method(self, data):
        neighbor_values = data.values[..., self._locs]
        neighbor_values[..., self._out_of_bound_locs] = np.nan
        neighbor_weights = self._weights[..., :-1]
        remapped_data = np.nansum(neighbor_values * neighbor_weights, axis=-1)
        remapped_data[self._neighbors_not_available(neighbor_values)] = np.nan
        return remapped_data

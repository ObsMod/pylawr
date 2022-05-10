#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch, MagicMock
import time

# External modules
import xarray as xr
import numpy as np
import scipy.spatial
import scipy.linalg
import scipy.linalg.lapack

# Internal modules
from pylawr.remap.base import NotFittedError
from pylawr.remap.simple_kriging import SimpleKriging, DEFAULT_ALPHA
from pylawr.remap.kernel import gaussian_rbf
from pylawr.grid import PolarGrid, CartesianGrid
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestSimpleKriging(unittest.TestCase):
    def setUp(self):
        self.kernel = gaussian_rbf(length_scale=20000)
        self.intp = SimpleKriging(kernel=self.kernel, n_neighbors=20)
        self.source = PolarGrid()
        self.target = CartesianGrid(nr_points=40, resolution=10000, start=0)
        self.array = create_array(self.source, 30)
        self.array = self.array.lawr.set_grid_coordinates(self.source)

    def test_alpha_returns_private_alpha(self):
        self.intp._alpha = 12345
        self.assertEqual(self.intp.alpha, 12345)

    def test_alpha_set_new_alpha(self):
        self.intp._alpha = None
        self.intp.alpha = 12345
        self.assertEqual(self.intp._alpha, 12345)

    def test_alpha_sets_to_default_for_none(self):
        self.intp._alpha = None
        self.intp.alpha = None
        self.assertEqual(self.intp._alpha, DEFAULT_ALPHA)

    def test_alpha_raises_type_error_if_wrong(self):
        with self.assertRaises(TypeError):
            self.intp.alpha = 'test'

    def test_kernel_returns_private_kernel(self):
        self.intp._kernel = 12
        self.assertEqual(self.intp.kernel, 12)

    def test_kernel_sets_private_kernel(self):
        self.intp._kernel = None
        kernel = gaussian_rbf()
        self.intp.kernel = kernel
        self.assertIsNotNone(self.intp._kernel)
        self.assertEqual(id(kernel), id(self.intp._kernel))

    def test_fit_calls_fit_from_tree_based(self):
        src_grid = self.intp._prepare_grid(self.source)
        trg_grid = self.intp._prepare_grid(self.target)
        dists, locs, tree = self.intp._get_tree(src_grid, trg_grid)
        self.intp._tree = tree
        self.intp._locs = locs
        self.intp._dists = dists
        self.intp._grid_in = self.source
        self.intp._grid_out = self.target
        with patch('pylawr.remap.tree.TreeRemap.fit') as p:
            self.intp.fit(self.source, self.target)
        p.assert_called_once_with(self.source, self.target)

    def test_get_distance_matrix_returns_dist_matrix_for_points(self):
        self.intp.n_neighbors = 20
        self.intp.fit(self.source, self.target)
        src_points = self.intp._prepare_grid(self.source)
        src_points = src_points.values[self.intp._locs, :]

        dist_matrix = self.intp._get_distance_matrix(src_points)
        for ind in range(src_points.shape[0]):
            tmp_points = src_points[ind]
            right_dist = scipy.spatial.distance_matrix(tmp_points, tmp_points)
            tmp_dist = dist_matrix[ind]
            np.testing.assert_equal(tmp_dist, right_dist)

    def test_k_dist_returns_target_array(self):
        self.intp.fit(self.source, self.target)
        right_kernel = self.intp.kernel(self.intp._dists)
        np.testing.assert_equal(self.intp._get_k_matrix(self.intp._dists),
                                right_kernel)

    def test_rkhs_calls_get_dist_matrix(self):
        self.intp.fit(self.source, self.target)
        src_grid = self.intp._prepare_grid(self.source)
        src_points = src_grid.values[self.intp._locs, :]
        dist_matrix = self.intp._get_distance_matrix(src_points)
        self.intp._get_distance_matrix = MagicMock(return_value=dist_matrix)
        _ = self.intp._const_rkhs_matrix(src_points)
        self.intp._get_distance_matrix.assert_called_once_with(src_points)

    def test_rkhs_returns_hilbert_kernel(self):
        self.intp.n_neighbors = 20
        self.intp.fit(self.source, self.target)
        src_grid = self.intp._prepare_grid(self.source)
        src_points = src_grid.values[self.intp._locs, :]

        dist_matrix = self.intp._get_distance_matrix(src_points)
        kernel_matrix = self.intp.kernel(dist_matrix)
        diag_ind = np.diag_indices(self.intp.n_neighbors)
        kernel_matrix[..., diag_ind[0], diag_ind[1]] += self.intp.alpha

        rkhs = self.intp._const_rkhs_matrix(src_points)
        np.testing.assert_equal(rkhs, kernel_matrix)

    def test_estimate_weights_solves_linear_equation(self):
        self.intp.n_neighbors = 4
        self.intp.fit(self.source, self.target)
        src_grid = self.intp._prepare_grid(self.source)
        src_points = src_grid.values[self.intp._locs, :]
        rkhs = self.intp._const_rkhs_matrix(src_points)
        targets = self.intp._get_k_matrix(self.intp._dists)
        ret_weights = self.intp._estimate_weights(rkhs, targets)
        weights = np.linalg.solve(rkhs, targets)
        np.testing.assert_equal(ret_weights, weights)

    def test_fit_calls_estimate_weights(self):
        self.intp.n_neighbors = 4
        self.intp.fit(self.source, self.target)
        src_grid = self.intp._prepare_grid(self.source)
        src_points = src_grid.values[self.intp._locs, :]
        rkhs = self.intp._const_rkhs_matrix(src_points)
        self.intp._weights = self.intp._estimate_weights(
            rkhs, self.intp._get_k_matrix(self.intp._dists)
        )
        self.intp._estimate_weights = MagicMock(return_value=self.intp._weights)
        self.intp.fit(self.source, self.target)
        self.intp._estimate_weights.assert_called_once()
        np.testing.assert_equal(
            self.intp._estimate_weights.call_args[0][0], rkhs
        )
        np.testing.assert_equal(
            self.intp._estimate_weights.call_args[0][1],
            self.intp._get_k_matrix(self.intp._dists)
        )

    def test_covariance_returns_kriging_covariance(self):
        self.intp.n_neighbors = 4
        self.intp.kernel.params[0].value = 10
        self.intp.fit(self.source, self.target)
        k_t_matrix = np.swapaxes(self.intp._get_k_matrix(self.intp._dists),
                                 -2, -1)
        var_reduction = np.diagonal(np.matmul(self.intp._weights, k_t_matrix))
        covariance = self.intp.kernel(0) - var_reduction
        np.testing.assert_almost_equal(self.intp.covariance, covariance)

    def test_covariance_raises_notfittederror(self):
        self.intp.n_neighbors = 4
        self.intp.kernel.params[0].value = 10
        with self.assertRaises(NotFittedError) as e:
            _ = self.intp.covariance
        self.assertEqual(
            str(e.exception), 'This kriging was not fitted yet!'
        )

    def test_remap_multiplies_weights_with_data(self):
        self.intp.n_neighbors = 4
        self.intp.kernel.params[0].value = 10
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neigbor_vals = stacked_array.values[..., self.intp._locs]
        mean = np.mean(neigbor_vals, axis=-1)
        interp_values = np.moveaxis(
            np.moveaxis(neigbor_vals, -1, 0) - mean, 0, -1
        )
        kriging_vals = mean + np.sum(interp_values * self.intp._weights,
                                     axis=-1)

        ret_vals = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(ret_vals, kriging_vals)

    def test_remap_sets_oob_locs_to_nan(self):
        self.intp.n_neighbors = 4
        self.intp.kernel.params[0].value = 10
        self.intp.max_dist = 10000
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neigbor_vals = stacked_array.values[..., self.intp._locs]
        neigbor_vals[..., self.intp._out_of_bound_locs] = np.nan
        mean = np.mean(neigbor_vals, axis=-1)
        interp_values = np.moveaxis(
            np.moveaxis(neigbor_vals, -1, 0) - mean, 0, -1
        )
        kriging_vals = mean + np.nansum(interp_values * self.intp._weights,
                                        axis=-1)
        kriging_vals[self.intp._neighbors_not_available(interp_values)] = np.nan

        ret_vals = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(ret_vals, kriging_vals)

    def test_remap_remaps_field_with_kriging(self):
        self.intp.fit(self.source, self.target)
        prepared_data = self.intp._stack_grid_coords(self.array)
        stacked_out = self.intp._remap_method(prepared_data)
        right_interpolated = self.intp._array_postprocess(stacked_out,
                                                          prepared_data)

        returned_interpolated = self.intp.remap(self.array)
        xr.testing.assert_identical(returned_interpolated, right_interpolated)


if __name__ == '__main__':
    unittest.main()

#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.remap.ord_kriging import OrdinaryKriging
from pylawr.remap.simple_kriging import DEFAULT_ALPHA
from pylawr.remap.kernel import gaussian_rbf
from pylawr.grid import PolarGrid, CartesianGrid
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestOrdinaryKriging(unittest.TestCase):
    def setUp(self):
        self.kernel = gaussian_rbf(length_scale=20000)
        self.intp = OrdinaryKriging(kernel=self.kernel, n_neighbors=20)
        self.source = PolarGrid()
        self.target = CartesianGrid(nr_points=40, resolution=10000, start=0)
        self.array = create_array(self.source, 30.)
        self.array = self.array.lawr.set_grid_coordinates(self.source)

    def test_k_dist_returns_target_array(self):
        self.intp.fit(self.source, self.target)
        right_kernel = np.ones(
            (self.intp._dists.shape[0], self.intp.n_neighbors+1)
        )
        right_kernel[:, :-1] = self.intp.kernel(self.intp._dists)
        np.testing.assert_equal(self.intp._get_k_matrix(self.intp._dists),
                                right_kernel)

    def test_remap_multiplies_weights_with_data(self):
        self.intp.n_neighbors = 4
        self.intp.kernel.params[0].value = 10
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neigbor_vals = stacked_array.values[..., self.intp._locs]
        kriging_vals = np.sum(neigbor_vals * self.intp._weights[..., :-1],
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
        kriging_vals = np.nansum(
            neigbor_vals * self.intp._weights[..., :-1],
            axis=-1
        )
        kriging_vals[self.intp._neighbors_not_available(neigbor_vals)] = np.nan

        ret_vals = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(ret_vals, kriging_vals)

    def test_rkhs_returns_hilbert_kernel(self):
        self.intp.n_neighbors = 20
        self.intp.fit(self.source, self.target)
        src_grid = self.intp._prepare_grid(self.source)
        src_points = src_grid.values[self.intp._locs, :]
        prepare_matrix = np.ones((
            self.intp._locs.shape[0], self.intp.n_neighbors+1,
            self.intp.n_neighbors+1
        ))
        dist_matrix = self.intp._get_distance_matrix(src_points)
        kernel_matrix = self.intp.kernel(dist_matrix)
        diag_ind = np.diag_indices(self.intp.n_neighbors)
        kernel_matrix[..., diag_ind[0], diag_ind[1]] += DEFAULT_ALPHA
        prepare_matrix[:, :-1, :-1] = kernel_matrix
        prepare_matrix[:, -1, -1] = 0
        rkhs = self.intp._const_rkhs_matrix(src_points)
        np.testing.assert_equal(rkhs, prepare_matrix)


if __name__ == '__main__':
    unittest.main()

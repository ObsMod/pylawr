#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from copy import deepcopy

# External modules
from scipy.spatial import cKDTree
import numpy as np
import xarray as xr

# Internal modules
from pylawr.remap.tree import TreeRemap
from pylawr.remap.nearest import NearestNeighbor
from pylawr.grid import PolarGrid, CartesianGrid
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestTreeRemap(unittest.TestCase):
    def setUp(self):
        self.intp = TreeRemap()
        self.source = PolarGrid(nr_ranges=20, nr_azi=36)
        self.target = CartesianGrid(nr_points=40, start=0)

    def test_max_dist_returns_private(self):
        self.intp._max_dist = 122343467
        self.assertEqual(self.intp.max_dist, self.intp._max_dist)

    def test_max_dist_sets_private(self):
        self.intp._max_dist = None
        self.intp.max_dist = 12345
        self.assertIsNotNone(self.intp._max_dist)
        self.assertEqual(12345, self.intp._max_dist)

    def test_max_dist_converts_none_into_infinity(self):
        self.intp.max_dist = None
        self.assertEqual(self.intp.max_dist, np.inf)

    def test_max_dist_raises_type_error_if_not_none_or_number(self):
        with self.assertRaises(TypeError):
            self.intp.max_dist = 'test'

    def test_fit_sets_grids(self):
        self.assertIsNone(self.intp._grid_in)
        self.assertIsNone(self.intp._grid_out)
        self.intp.fit(self.source, self.target)
        self.assertIsNotNone(self.intp._grid_in)
        self.assertIsNotNone(self.intp._grid_out)
        self.assertEqual(id(self.intp._grid_in), id(self.source))
        self.assertEqual(id(self.intp._grid_out), id(self.target))

    def test_fit_builds_up_tree_with_cartesian_source(self):
        self.assertIsNone(self.intp._tree)
        cartesian_src = self.intp._prepare_grid(self.source)
        source_tree = cKDTree(cartesian_src)

        self.intp.fit(self.source, self.target)
        self.assertIsInstance(self.intp._tree, cKDTree)
        np.testing.assert_equal(self.intp._tree.data, source_tree.data)

    def test_fit_queries_with_target_grid(self):
        cartesian_src = self.intp._prepare_grid(self.source)
        cartesian_trg = self.intp._prepare_grid(self.target)
        source_tree = cKDTree(cartesian_src)
        dists, locs = source_tree.query(cartesian_trg, self.intp.n_neighbors)

        self.intp.fit(self.source, self.target)
        self.assertIsInstance(self.intp._dists, np.ndarray)
        self.assertIsInstance(self.intp._locs, np.ndarray)
        np.testing.assert_equal(self.intp._dists, dists)
        np.testing.assert_equal(self.intp._locs, locs)

    def test_fit_queries_with_max_distance(self):
        max_dist = 100
        cartesian_src = self.intp._prepare_grid(self.source)
        cartesian_trg = self.intp._prepare_grid(self.target)
        source_tree = cKDTree(cartesian_src)
        dists, _ = source_tree.query(cartesian_trg, self.intp.n_neighbors,
                                     distance_upper_bound=max_dist)

        self.intp.max_dist = max_dist
        self.intp.fit(self.source, self.target)
        self.assertIsInstance(self.intp._dists, np.ndarray)
        np.testing.assert_equal(self.intp._dists, dists)

    def test_fit_uses_nearest_neighbours_for_query(self):
        self.intp.fit(self.source, self.target)
        self.assertTupleEqual(self.intp._dists.shape, (1600, 20))
        self.intp.n_neighbors = 2
        self.intp.fit(self.source, self.target)
        self.assertTupleEqual(self.intp._dists.shape, (1600, 2))

    def test_fit_sets_at_least_2d_array(self):
        self.intp.n_neighbors = 1
        self.intp.fit(self.source, self.target)
        self.assertEqual(self.intp._dists.ndim, 2)
        self.assertEqual(self.intp._locs.ndim, 2)
        self.assertEqual(self.intp._locs.shape[1], 1)
        self.assertEqual(self.intp._locs.shape[1], 1)

    def test_fit_sets_oob_locs_to_minus_one(self):
        self.intp.n_neighbors = 2
        self.intp.max_dist = 100
        self.intp.fit(self.source, self.target)
        prep_source = self.intp._prepare_grid(self.source)
        np.testing.assert_array_less(self.intp._locs, prep_source.shape[0])
        np.testing.assert_array_equal(
            self.intp._locs[self.intp._dists == np.inf], -1
        )

    def test_out_of_bound_locs_returns_boolean_array(self):
        self.intp.n_neighbors = 2
        self.intp.max_dist = 100
        self.intp.fit(self.source, self.target)
        out_of_bounds = self.intp._dists == np.inf
        np.testing.assert_array_equal(out_of_bounds,
                                      self.intp._out_of_bound_locs)

    def test_fitted_checks_if_dists_and_locs_set(self):
        self.assertFalse(self.intp.fitted)
        self.intp._dists = 1
        self.assertFalse(self.intp.fitted)
        self.intp._dists = None
        self.intp._locs = 1
        self.assertFalse(self.intp.fitted)
        self.intp._dists = 1
        self.assertTrue(self.intp.fitted)
        self.intp = TreeRemap()
        self.assertFalse(self.intp.fitted)
        self.intp.fit(self.source, self.target)
        self.assertTrue(self.intp.fitted)


class TestNearest(unittest.TestCase):
    def setUp(self):
        self.intp = NearestNeighbor()
        self.source = PolarGrid()
        self.target = CartesianGrid(nr_points=40, start=0)
        self.array = create_array(self.source, 30.)
        self.array = self.array.lawr.set_grid_coordinates(self.source)
        self.intp.fit(self.source, self.target)

    def test_remap_method_select_nearest_neighbours_based_on_loc(self):
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        nearest_array = stacked_array.values[..., self.intp._locs][..., 0]
        returned_array = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(nearest_array, returned_array)

    def test_remap_method_remaps_with_average_of_neighbors(self):
        self.intp.n_neighbors = 2
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neighbor_vals = stacked_array.values[..., self.intp._locs]

        returned_array = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(np.median(neighbor_vals, axis=-1),
                                returned_array)

    def test_check_if_all_neighbors_nan_checks_nans(self):
        self.intp.n_neighbors = 10
        self.intp.max_dist = 100
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neighbor_vals = stacked_array.values[..., self.intp._locs]
        neighbor_vals[..., self.intp._out_of_bound_locs] = np.nan
        nan_sum_array = np.sum(np.isnan(neighbor_vals).astype(int), axis=-1) > 0

        np.testing.assert_array_equal(
            nan_sum_array, self.intp._all_neighbors_nan(neighbor_vals)
        )
    #
    # def test_remap_method_remaps_with_half_dist(self):
    #     self.intp.n_neighbors = 4
    #     self.intp.max_dist = 100
    #     self.intp._inner_max_dist = 0.5
    #     self.intp._inner_neighbors = 0.75
    #     self.source = CartesianGrid(resolution=50, nr_points=2, start=0)
    #     self.target = CartesianGrid(resolution=0, nr_points=1, start=0)
    #     new_array = xr.DataArray(
    #         data=np.ones((1, 2, 2)),
    #         dims=['time', 'x', 'y']
    #     )
    #     new_array['time'] = self.array['time']
    #     new_array = new_array.lawr.set_grid_coordinates(self.source)
    #     stacked_array = new_array.stack(grid=('x', 'y'))
    #     self.intp.fit(self.source, self.target)
    #     returned_array = self.intp._remap_method(stacked_array)
    #     np.testing.assert_equal(np.isnan(returned_array), False)
    #     self.source = CartesianGrid(resolution=(50, 55), nr_points=2, start=0)
    #     self.intp.fit(self.source, self.target)
    #     returned_array = self.intp._remap_method(stacked_array)
    #     np.testing.assert_equal(np.isnan(returned_array), True)

    def test_remap_method_remaps_with_max_distance(self):
        self.intp.n_neighbors = 10
        self.intp.max_dist = 100
        self.intp.fit(self.source, self.target)
        stack_dims = self.source.coord_names
        stacked_array = self.array.stack(grid=stack_dims)
        neighbor_vals = stacked_array.values[..., self.intp._locs]
        neighbor_vals[..., self.intp._out_of_bound_locs] = np.nan
        neighbor_median = np.nanmedian(neighbor_vals, axis=-1)
        neighbor_median[
            self.intp._neighbors_not_available(neighbor_vals)
        ] = np.nan

        returned_array = self.intp._remap_method(stacked_array)
        np.testing.assert_equal(neighbor_median, returned_array)

    def test_remap_remaps_field_with_nearest_neighbour(self):
        prepared_data = self.intp._stack_grid_coords(self.array)
        stacked_out = self.intp._remap_method(prepared_data)
        right_interpolated = self.intp._array_postprocess(stacked_out,
                                                          prepared_data)

        returned_interpolated = self.intp.remap(self.array)
        xr.testing.assert_identical(returned_interpolated, right_interpolated)


if __name__ == '__main__':
    unittest.main()

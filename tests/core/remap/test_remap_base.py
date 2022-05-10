#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import PropertyMock, patch

# External modules
import xarray as xr
import numpy as np

# Internal modules
import pylawr.functions.grid as grid_funcs
from pylawr.remap.base import BaseRemap, NotImprovableError, NotFittedError
from pylawr.grid import PolarGrid, CartesianGrid
from pylawr.grid.unstructured import UnstructuredGrid
from pylawr.utilities.helpers import create_array


logger = logging.getLogger(__name__)
logger.level = int(os.environ.get('PYLAWR_TEST_LOGLEVEL', logging.CRITICAL))

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestBaseRemap(unittest.TestCase):
    def setUp(self):
        self.int_obj = BaseRemap()
        self.grid_in = PolarGrid()
        self.grid_out = CartesianGrid()
        self.array = create_array(self.grid_in, 30.)

    @staticmethod
    def grid_to_unstructured(grid):
        lat_lon = np.array(grid.lat_lon).reshape(2, -1).transpose()
        unstructured = UnstructuredGrid(lat_lon)
        return unstructured

    def test_optimize_raises_not_implemented_error(self):
        with self.assertRaises(NotImprovableError) as e:
            _ = self.int_obj.optimize(self.array, False)
        self.assertEqual(
            str(e.exception), 'This interpolation cannot be optimized!'
        )

    def test_remap_raises_not_fitted_error_if_not_fitted(self):
        with patch('pylawr.remap.base.BaseRemap.fitted',
                   new_callable=PropertyMock, return_value=False) as f_patch:
            with self.assertRaises(NotFittedError) as e:
                _ = self.int_obj.remap(self.array)
            self.assertEqual(
                str(e.exception), 'This interpolation was not fitted yet!'
            )

    def test_get_cartesian_checks_grid(self):
        error_msg = 'The given grid is no valid grid and cannot be used!'

        with self.assertRaises(TypeError) as e:
            _ = grid_funcs.get_cartesian(self.array)
        self.assertEqual(str(e.exception), error_msg)

    def test_get_cartesian_calculates_cartesian_coords_use_altitude(self):
        cartesian = grid_funcs.get_cartesian(self.grid_in, use_altitude=True)

        lat = np.deg2rad(self.grid_in.get_lat_lon()['lat'])
        lon = np.deg2rad(self.grid_in.get_lat_lon()['lon'])
        x = self.grid_in.earth_radius * np.cos(lat) * np.cos(lon)
        y = self.grid_in.earth_radius * np.cos(lat) * np.sin(lon)
        z = self.grid_in.earth_radius * np.sin(lat)
        z += self.grid_in.get_altitude()
        np.testing.assert_equal(cartesian[..., 0].values, x.values)
        np.testing.assert_equal(cartesian[..., 1].values, y.values)
        np.testing.assert_equal(cartesian[..., 2].values, z.values)

    def test_get_cartesian_calculates_cartesian_coords_omit_altitude(self):
        cartesian = grid_funcs.get_cartesian(self.grid_in, use_altitude=False)

        lat = np.deg2rad(self.grid_in.get_lat_lon()['lat'])
        lon = np.deg2rad(self.grid_in.get_lat_lon()['lon'])
        x = self.grid_in.earth_radius * np.cos(lat) * np.cos(lon)
        y = self.grid_in.earth_radius * np.cos(lat) * np.sin(lon)
        z = self.grid_in.earth_radius * np.sin(lat)

        np.testing.assert_equal(cartesian[..., 0].values, x.values)
        np.testing.assert_equal(cartesian[..., 1].values, y.values)
        np.testing.assert_equal(cartesian[..., 2].values, z.values)

    def test_get_cartesian_calculates_cartesian_coords_check_default(self):
        cartesian_left = grid_funcs.get_cartesian(self.grid_in,
                                                  use_altitude=False)
        cartesian_right = grid_funcs.get_cartesian(self.grid_in)

        xr.testing.assert_equal(cartesian_left, cartesian_right)

    def test_get_cartesian_array_shape(self):
        cartesian = grid_funcs.get_cartesian(self.grid_in)
        self.assertIsInstance(cartesian, xr.DataArray)
        self.assertListEqual(
            list(cartesian.shape), list(self.grid_in._data_shape) + [3, ]
        )

    def test_get_cartesian_uses_grid(self):
        cartesian_one = grid_funcs.get_cartesian(self.grid_in)
        cartesian_two = grid_funcs.get_cartesian(self.grid_out)
        with self.assertRaises(AssertionError):
            xr.testing.assert_equal(
                cartesian_one, cartesian_two
            )

    def test_prepare_grid_stacks_grid(self):
        cartesian_one = grid_funcs.get_cartesian(self.grid_in)
        stack_dims = cartesian_one.dims[:-1]
        stacked_one = cartesian_one.stack(stacked=stack_dims).transpose()
        returned_prepared = self.int_obj._prepare_grid(self.grid_in)
        xr.testing.assert_identical(stacked_one, returned_prepared)

    def test_stack_data_stacks_data_based_on_grid_coords(self):
        self.int_obj._grid_in = self.grid_in
        self.array = self.array.lawr.set_grid_coordinates(self.grid_in)
        stacked_array = self.array.stack(grid=self.grid_in.coord_names)
        returned_array = self.int_obj._stack_grid_coords(self.array)
        xr.testing.assert_equal(stacked_array, returned_array)

    def test_prepare_grid_works_with_single_grid_dim(self):
        unstructured_grid = self.grid_to_unstructured(self.grid_in)
        cartesian_un = grid_funcs.get_cartesian(unstructured_grid)
        returned_prepared = grid_funcs.prepare_grid(unstructured_grid)
        xr.testing.assert_equal(cartesian_un, returned_prepared)


if __name__ == '__main__':
    unittest.main()

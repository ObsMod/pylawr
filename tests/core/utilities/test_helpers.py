#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
import datetime

# External modules
import numpy as np
import xarray as xr

# Internal modules
from pylawr.utilities.helpers import polar_padding, create_array
from pylawr.grid import PolarGrid

logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class TestCreateArray(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()

    def test_has_given_constant_value(self):
        array = create_array(self.grid, 0)
        np.testing.assert_equal(array.values, 0)
        array = create_array(self.grid, 10)
        np.testing.assert_equal(array.values, 10)

    def test_array_has_right_dimensions(self):
        right_dimensions = ['time', *self.grid.coord_names]
        array = create_array(self.grid)
        self.assertListEqual(right_dimensions, list(array.dims))

    def test_right_array(self):
        dims = ['time', *self.grid.coord_names]
        coords = self.grid.get_coordinates()
        timestep = datetime.datetime(1970, 1, 1)
        coords['time'] = [timestep]
        shape = (1, *self.grid.grid_shape)
        data = np.ones(shape=shape, dtype=float) * 99
        right_da = xr.DataArray(
            data=data,
            coords=coords,
            dims=dims
        )
        returned_da = create_array(self.grid, 99, timestamp=timestep)
        xr.testing.assert_equal(returned_da, right_da)

    def test_set_timestamp(self):
        timestep = datetime.datetime(1995, 1, 22)
        returned_da = create_array(self.grid, 99, timestamp=timestep)
        self.assertEqual(np.datetime64(timestep), returned_da.time.values)

    def test_has_same_grid(self):
        returned_array = create_array(self.grid, 99)
        self.assertEqual(id(self.grid), id(returned_array.lawr.grid))

    def test_has_testing_tag(self):
        returned_array = create_array(self.grid, 99)
        self.assertIn('testing', returned_array.attrs['tags'])


class TestPolarPadding(unittest.TestCase):
    def setUp(self):
        self.reflectivity = np.arange(1, 13).reshape(3, 4)
        self.padded_array = np.array([
            [2, 3, 4, 1, 2, 3],
            [4, 1, 2, 3, 4, 1],
            [8, 5, 6, 7, 8, 5],
            [12, 9, 10, 11, 12, 9],
            [12, 9, 10, 11, 12, 9]
        ])

    def test_functional_padding(self):
        padded_array = polar_padding(self.reflectivity, (1, 1))
        np.testing.assert_equal(padded_array, self.padded_array)

    def test_padding_appends_lower_pad(self):
        lower_pad = np.array([2, 3, 4, 1, 2, 3])
        padded_array = polar_padding(self.reflectivity, (1, 1))
        np.testing.assert_equal(padded_array[0], lower_pad)

    def test_padding_appends_upper_pad(self):
        upper_pad = np.array([12, 9, 10, 11, 12, 9])
        padded_array = polar_padding(self.reflectivity, (1, 1))
        np.testing.assert_equal(padded_array[-1], upper_pad)

    def test_padding_appends_left_pad(self):
        left_pad = np.array([2, 4, 8, 12, 12])
        padded_array = polar_padding(self.reflectivity, (1, 1))
        np.testing.assert_equal(padded_array[:, 0], left_pad)

    def test_padding_appends_right_pad(self):
        right_pad = np.array([3, 1, 5, 9, 9])
        padded_array = polar_padding(self.reflectivity, (1, 1))
        np.testing.assert_equal(padded_array[:, -1], right_pad)

    def test_padding_uses_pad_size(self):
        test_array = np.pad(self.padded_array[:, 1:-1], ((0, 0), (2, 2)),
                            mode='wrap')
        padded_array = polar_padding(self.reflectivity, (1, 2))
        np.testing.assert_equal(padded_array, test_array)


if __name__ == '__main__':
    unittest.main()

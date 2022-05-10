#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import unittest
import logging
import os
from unittest.mock import patch, PropertyMock
import math

# External modules
import xarray as xr
import numpy as np

# Internal modules
from pylawr.grid import PolarGrid, LatLonGrid
from pylawr.functions.grid import remap_data, get_latlon_grid
from pylawr.remap import NearestNeighbor
from pylawr.utilities.helpers import create_array


logging.basicConfig(level=logging.DEBUG)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_PATH), 'data')


class TestGridFunctions(unittest.TestCase):
    def setUp(self):
        self.grid = PolarGrid()
        self.data = create_array(self.grid, 20.)

    def test_remap_data_returns_remaps_data(self):
        new_grid = PolarGrid(range_res=120)
        remapper = NearestNeighbor(1)
        remapper.fit(self.grid, new_grid)
        remapped_data = remapper.remap(self.data)
        returned_data, returned_remapper = remap_data(
            self.data, self.grid, new_grid
        )
        xr.testing.assert_equal(remapped_data, returned_data)
        self.assertNotEqual(id(remapper), id(returned_remapper))

    def test_remap_data_uses_given_remapper(self):
        new_grid = PolarGrid(range_res=120)
        remapper = NearestNeighbor(5)
        remapper.fit(self.grid, new_grid)
        remapped_data = remapper.remap(self.data)
        returned_data, returned_remapper = remap_data(
            self.data, self.grid, new_grid, remapper=remapper
        )
        xr.testing.assert_equal(remapped_data, returned_data)
        self.assertEqual(id(remapper), id(returned_remapper))

    def test_get_lat_lon_grid_returns_lat_lon_grid(self):
        returned_grid = get_latlon_grid(self.grid)
        self.assertIsInstance(returned_grid, LatLonGrid)

    def test_get_lat_lon_grid_uses_lat_lon_from_orig_grid(self):
        orig_lat_lon = self.grid.lat_lon
        with patch('pylawr.grid.polar.BaseGrid.lat_lon',
                   new_callable=PropertyMock,
                   return_value=orig_lat_lon) as grid_patch:
            _ = get_latlon_grid(self.grid)
        grid_patch.assert_called_once()

    def test_get_lat_lon_uses_start_point_as_min_lat_lon(self):
        lat, lon = self.grid.lat_lon
        start_lat, start_lon = get_latlon_grid(self.grid).start
        self.assertEqual(start_lat, np.min(lat))
        self.assertEqual(start_lon, np.min(lon))

    def test_get_lat_lon_uses_median_steps_as_resolution(self):
        lat, lon = self.grid.lat_lon
        mean_lat = np.median(np.abs(np.diff(lat, axis=-1)))
        mean_lon = np.median(np.abs(np.diff(lon, axis=0)))
        res_lat, res_lon = get_latlon_grid(self.grid).resolution
        self.assertEqual(res_lat, mean_lat)
        self.assertEqual(res_lon, mean_lon)

    def test_get_lat_lon_steps_as_end_start_res(self):
        lat, lon = self.grid.lat_lon
        diff_lat = np.max(lat) - np.min(lat)
        diff_lon = np.max(lon) - np.min(lon)
        res_lat = np.median(np.abs(np.diff(lat, axis=-1)))
        res_lon = np.median(np.abs(np.diff(lon, axis=0)))
        steps_lat = math.ceil(diff_lat/res_lat)
        steps_lon = math.ceil(diff_lon/res_lon)
        nr_lat, nr_lon = get_latlon_grid(self.grid).nr_points
        self.assertEqual(steps_lat, nr_lat)
        self.assertEqual(steps_lon, nr_lon)

    def test_get_lat_lon_sets_center_to_grid_center(self):
        self.grid.center = (48, 8, 100)
        ret_center = get_latlon_grid(self.grid).center
        self.assertTupleEqual(self.grid.center, ret_center)


if __name__ == '__main__':
    unittest.main()
